import sys

import torch

from anytree import Node
from .build_tree import makeTree,get_relu_leaf_name,get_entity




#每一层
def calculate_boundary(dataset, model,b_type):

    root = Node('root')
    makeTree(model, root, '')
    acts = get_relu_leaf_name(model, root)
    if len(acts) == 0:
        return []
    print("需要修改的层数：{}".format(len(acts)))
    maxlist = find_brelu_gate(dataset, model, acts,b_type)
    return maxlist

#每个通道
def calculate_ch_boundary(dataset, model,b_type):


    root = Node('root')
    makeTree(model, root, '')
    acts = get_relu_leaf_name(model,root)
    if len(acts) == 0:
        return []

    print("需要修改的层数：{}".format(len(acts)))
    return find_chrelu_gate(dataset, model, acts,b_type)

def get_previous_layer(model, relu_index):
    layers = list(model.children())
    return layers[relu_index - 1] if relu_index > 0 else None
def for_hook(inpt):
    def hook(model,input,output):
        inp=torch.tensor(*input)
        inpt.append(inp)
    return hook

# 得到relu层之前的输出，也就是rulu层的输入，找到最大值
def conv_output(model, layer, img):
    input_data = torch.as_tensor(img,dtype=torch.float32,device='cuda')
    # Temporarily remove the layers after the target layer
    inpt=[]
    entity=get_entity(model,layer)
    index = layer.split('_')
    index = index[len(index) - 1]
    entity = getattr(entity, index)     #返回relu层
    hook=for_hook(inpt)
    handle=entity.register_forward_hook(hook)
    #冻结各层参数运行
    with torch.no_grad():
        model(input_data)
    handle.remove()

    return inpt


def find_brelu_gate(dataset, model, acts,b_type):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    change_num = len(acts)
    max_list = [-10000] * change_num

    count = 0


    # assuming dataset is a DataLoader or a list of PyTorch tensors
    for i, data in enumerate(dataset):

        count += 1
    
        data = data.permute(0,3,1,2).to(device)
        for j in range(change_num):
            out = conv_output(model, acts[j], data)
            # Flatten and sort the activations
            #索引所有input找到最大的，放入maxList中
            for item in out:
                out_flat = torch.tensor(item).flatten()
                if b_type=='最大值':
                    max_val=out_flat.max()
                else:
                    k = int(0.995 * len(out_flat))
                    high_values = torch.topk(out_flat, k,largest=False)
                    max_val = high_values.values.max()
                if max_val > max_list[j]:
                    max_list[j] = max_val.item()  # storing as a standard Python number

        # print(f'this is {i}')
    return max_list

def find_chrelu_gate(dataset, model, acts,b_type):


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    change_num = len(acts)
    max_list = [[-10000000 for i in range(5000)] for j in range(change_num)]

    total = len(dataset)
    count = 0
    # assuming dataset is a DataLoader or a list of PyTorch tensors
    for i, data in enumerate(dataset):

        count += 1
 
        print("epoch:", i)
        data = data.permute(0, 3, 1, 2).to(device)
        data = data.to(device)
        for j in range(change_num):
            out = conv_output(model, acts[j], data)
            for item in out:
                # max_list[j][0] =out.shape[-1]
                if item.shape[1]>5000:
                    return False
                #批组数
                for m in range(item.shape[0]):
                    #通道数
                    for n in range(item.shape[1]):
                        #卷积层
                        if len(item.shape)==4:
                            if b_type=='最大值':
                                max_num = torch.max(item[m, n, :, :].flatten())
                            else:
                                out_flat = torch.topk(item[m, n, :, :].flatten(),int(0.995 * len(item[m, n, :, :].flatten())), largest=False)
                                max_num = out_flat.values.max()
                            max_list[j][n]=max(max_num,max_list[j][n])
                        #全连接层
                        elif  len(item.shape)==2:
                            max_list[j][n]=max(max_list[j][n],item[m][n])
                        else:
                            return False

    # tensor,
    length = []
    mean = []

    # df = pd.DataFrame()
    # for i in range(len(max_list)):
    #     count = 0
    #     total = 0
    #     value = []
    #     for item in max_list[i]:
    #         if isinstance(item, torch.Tensor):
    #             total += item.item()
    #             value.append(item.item())
    #             count += 1
    #         else:
    #             value.append(item)
    #     length.append(count)
    #     mean.append(total/count)
    #     df[f'layer_{i}'] = value
    # print(length)
    # print(mean)

    # # 将DataFrame写入Excel文件
    # df.to_excel('data/output.xlsx', index=False)

    return max_list
