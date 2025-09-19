import time
from attack.data_conversion import hamming_distance
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import argparse
import os,sys,copy
import torch.nn.functional as F
import utils.change_relu as cr
import models
from attack.BFA import *
from utils.function import GABO
from models.quantization import quan_Conv2d, quan_Linear, quantize,quan_fixed_Conv2d

parser = argparse.ArgumentParser(
    description='modify the ReLU to Brelu',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',
                    default='/home/liaolei.pan/data/temp',
                    type=str,
                    help='Path to dataset')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')

## dataset
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10', 'mnist'],
    help='Choose between Cifar10/100 and ImageNet.')

parser.add_argument('--save_path',
                    type=str,
                    default='./save/',
                    help='Folder to save logs.')

parser.add_argument('--boundary_path',
                    type=str,
                    default='/home/liaolei.pan/code1/Brelu/boundary',
                    help='Folder to save boundaries.')

parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet18_quan'
                    )

parser.add_argument('--attack_sample_size',
                    type=int,
                    default=128,
                    help='attack sample size')

parser.add_argument('--n_iter',
                    type=int,
                    default=20,
                    help='number of attack iterations')

parser.add_argument(
    '--k_top',
    type=int,
    default=5,
    help='k weight with top ranking gradient used for bit-level gradient check.'
)

parser.add_argument('--test_batch_size',
                    type=int,
                    default=256,
                    help='Batch size.')

parser.add_argument('--workers',
                    type=int,
                    default=4,
                    help='number of data loading workers (default: 2)')
parser.add_argument('--find_boundaries',
                    action='store_true',
                    help='use the training data to find the boundaries of Brelu')
parser.add_argument('--fixed',
                    action='store_true',
                    help='replace the Brelu with fixed boundaries')


parser.add_argument(
    '--reset_weight',
    dest='reset_weight',
    action='store_true',
    help='enable the weight replacement with the quantized weight')
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')

args=parser.parse_args()

args.use_cuda = args.ngpu > 0 and torch.cuda.is_available() 

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_acc_history(save_path, history,log):
      
        import csv  
        log_dir = os.path.dirname(save_path)     
        if args.fixed:  
            log_filename = time.strftime("%Y-%m-%d-%H:%M")+ '-fixed.csv'
        else:
            log_filename = time.strftime("%Y-%m-%d-%H:%M")+ '.csv'
        csv_path = os.path.join(log_dir, log_filename)
        print(f"csvPath: {csv_path}")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Test_Top1_Accuracy(%)']) 
            writer.writerows(history)  
        
        print_log(f"Accuracy history saved to: {csv_path}", log)
def perform_attack(attacker, model, model_clean, train_loader, test_loader,
                   N_iter, log):
  
    model.eval()
    losses = AverageMeter()
    iter_time = AverageMeter()
    attack_time = AverageMeter()
    layer_list=[]

    layer_map={}

    # attempt to use the training data to conduct BFA
    for _, (data, target) in enumerate(train_loader):
        if args.use_cuda:
            target = target.cuda()
            data = data.cuda()
        # Override the target to prevent label leaking
        _, target = model(data).data.max(1)
        break

    val_acc_top1, val_acc_top5, val_loss = validate(test_loader, model,
                                                    attacker.criterion, log)
    history = []        # record the accuracy of the attack
    print_log('k_top is set to {}'.format(args.k_top), log)
    print_log('Attack sample size is {}'.format(data.size()[0]), log)
    end = time.time()
    for i_iter in range(N_iter):
        print_log('**********************************', log)
        attacker.progressive_bit_search(model, data, target,layer_list,layer_map)

        # measure data loading time
        attack_time.update(time.time() - end)
        end = time.time()

        h_dist = hamming_distance(model, model_clean)

        # record the loss
        losses.update(attacker.loss_max, data.size(0))

        print_log(
            'Iteration: [{:03d}/{:03d}]   '
            'Attack Time {attack_time.val:.3f} ({attack_time.avg:.3f})  '.
            format((i_iter + 1),
                   N_iter,
                   attack_time=attack_time,
                   iter_time=iter_time) + time_string(), log)

        print_log('loss before attack: {:.4f}'.format(attacker.loss.item()),
                  log)
        print_log('loss after attack: {:.4f}'.format(attacker.loss_max), log)
        print_log('bit flips: {:.0f}'.format(attacker.bit_counter), log)
        print_log('hamming_dist: {:.0f}'.format(h_dist), log)

        val_acc_top1, val_acc_top5, val_loss = validate(
            test_loader, model, attacker.criterion, log)
        history.append((i_iter + 1, val_acc_top1))
    
        # measure elapsed time
        iter_time.update(time.time() - end)
        print_log(
            'iteration Time {iter_time.val:.3f} ({iter_time.avg:.3f})'.format(
                iter_time=iter_time), log)
        end = time.time()
    log_file_path = log.name
    save_acc_history(log_file_path, history,log)
    return layer_list,layer_map

def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.use_cuda:
                target = target.cuda()
                input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

        print_log(
            '  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'
            .format(top1=top1, top5=top5, error1=100 - top1.avg), log)

    return top1.avg, top5.avg, losses.avg
def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(
        time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string

def load_boundaries(file_path):
    boundaries = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Boundary"):
                value_str = line.split(": ")[1]
                try:
                    value = float(value_str) 
                    boundaries.append(value)
                except:
                    boundaries.append(value_str)
    return boundaries

def main():
    # Init logger6
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.isdir(args.boundary_path):
        os.makedirs(args.boundary_path)
    log = open(
        os.path.join(args.save_path,
                     'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')),
              log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()),
              log)



    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif args.dataset == 'svhn':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'mnist':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    if args.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])  # here is actually the validation dataset
    else:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
 
    if args.dataset == 'mnist':
        train_data = dset.MNIST(args.data_path,
                                train=True,
                                transform=train_transform,
                                download=True)
        test_data = dset.MNIST(args.data_path,
                               train=False,
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path,
                                  train=True,
                                  transform=train_transform,
                                  download=True)
        test_data = dset.CIFAR10(args.data_path,
                                 train=False,
                                 transform=test_transform,
                                 download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path,
                                   train=True,
                                   transform=train_transform,
                                   download=True)
        test_data = dset.CIFAR100(args.data_path,
                                  train=False,
                                  transform=test_transform,
                                  download=True)
        num_classes = 100
    elif args.dataset == 'svhn':
        train_data = dset.SVHN(args.data_path,
                               split='train',
                               transform=train_transform,
                               download=True)
        test_data = dset.SVHN(args.data_path,
                              split='test',
                              transform=test_transform,
                              download=True)
        num_classes = 10
    elif args.dataset == 'stl10':
        train_data = dset.STL10(args.data_path,
                                split='train',
                                transform=train_transform,
                                download=True)
        test_data = dset.STL10(args.data_path,
                               split='test',
                               transform=test_transform,
                               download=True)
        num_classes = 10
    elif args.dataset == 'imagenet':
        train_dir = os.path.join('/opt/dataset/imagenet', 'val')
        test_dir = os.path.join('/opt/dataset/imagenet', 'val')
        train_data = dset.ImageFolder(train_dir, transform=train_transform)
        test_data = dset.ImageFolder(test_dir, transform=test_transform)
        num_classes = 1000
    else:
        assert False, 'Do not support dataset : {}'.format(args.dataset)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.attack_sample_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.test_batch_size,
                                              shuffle=True,
                                              num_workers=args.workers,
                                              pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)

    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)
    print_log("=> network :\n {}".format(net), log)

    if args.use_cuda:
        if args.ngpu > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    if args.find_boundaries:
        relu_idx=[]
        for i,(_,module) in enumerate(net.named_modules()):
            if isinstance(module,nn.ReLU):
                relu_idx.append(i)
        conv_idx=[x-1 for x in relu_idx]
        boundaries=[]
        
        for idx in conv_idx:
            print(idx)
            boundary=GABO(net,test_data,idx)
            boundaries.append(boundary)

  
        log_filename = time.strftime("%Y-%m-%d-%H:%M")+ '.txt'
        txt_path = os.path.join(args.boundary_path, log_filename)

        with open(txt_path, 'w') as f:
            for i, boundary in enumerate(boundaries):
                f.write(f"Boundary {i}: {boundary}\n")
    
        print("Boundaries successfully dump to boundaries.txt")
        return


    if args.fixed:
        filte_path=os.path.join(args.boundary_path,os.listdir(args.boundary_path)[-1])
        boundaries = load_boundaries(filte_path)
        cr.fix(net,boundaries,'BRelu')
        print(net)
 
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()


    # block for weight reset
    if args.reset_weight:
        for m in net.modules():
            if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
                m.__reset_weight__()
                # print(m.weight)

    attacker = BFA(criterion, args.k_top)
    net_clean = copy.deepcopy(net)
    # weight_conversion(net)

    validate(test_loader, net, criterion, log)
    perform_attack(attacker, net, net_clean, train_loader, test_loader,args.n_iter, log)

    log.close()
if __name__ == '__main__':
    main()
