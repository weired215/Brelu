from anytree import Node, RenderTree, PreOrderIter

import torch
import torch.nn as nn
from torchvision import models
from torchvision import models

class ComplexModel(nn.Module):
    def __init__(self):
        super(ComplexModel, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # 使用预训练的ResNet50作为基础模型
        num_ftrs = self.resnet.fc.in_features
        # 冻结ResNet的参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 替换ResNet的全连接层
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 假设输出类别数为10
        )

    def forward(self, x):
        return self.resnet(x)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.flatten=nn.Flatten()
        self.fc1 = nn.Linear(in_features=256, out_features=120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.mp1(self.relu1(self.conv1(x)))
        x = self.conv2(x)
        x = self.mp2(self.relu2(x))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x



# 判断是否是
def is_not_basic_module(entity):

    count = 0
    for _, _ in enumerate(entity.named_children()):
        count = count + 1
        return True

    return False


# 例如嵌套关系为 resnet_layer1_0_conv1
# 根据逻辑此时的 model 一定是 resnet_layer1_0 即上一层
# 因此只需要 getattr(entity,name[length - 1])
def get_next_entity(model,name):

    name = name.split('_')
    length = len(name)

    entity = model
    entity = getattr(entity,name[length - 1])
    return entity

# 以嵌套关系resnet_layer1_0_conv1为例
# 返回实体为 resnet_layer1_0
# 后续可通过 setattr(entity,'conv1',chRelu()) 修改最后一层的激活函数
def get_entity(model,name):

    name = name.split('_')
    length = len(name)

    entity = model

    for i in range(length-1):
        entity = getattr(entity,name[i])
    return entity



def makeTree(model,node,name):

    name_list = []

    # 获取同一层兄弟节点
    for _, tmp in enumerate(model.named_children()):
        name_list.append(tmp[0])

    for i in range(len(name_list)):

        if name == "":
            node_name = name_list[i]
        else:
            node_name = name+'_'+name_list[i]
        cur_node = Node(node_name,parent=node)

        entity = get_next_entity(model,node_name)

        if is_not_basic_module(entity):
            makeTree(entity,cur_node,node_name)



def get_relu_leaf_name(model,root):

    acts = []
    leaves = root.leaves
    for node in leaves:

        entity = get_entity(model,node.name)

        index = node.name.split('_')
        index = index[len(index) - 1]
        entity = getattr(entity, index)
        if type(entity) is nn.ReLU :
            acts.append(node.name)
    return acts

def get_leaf_name(model,root):

    acts = []
    leaves = root.leaves
    for node in leaves:

        entity = get_entity(model,node.name)

        index = node.name.split('_')
        index = index[len(index) - 1]
        entity = getattr(entity, index)
        if is_not_basic_module(entity) == False:
            acts.append(node.name)
    return acts

# if __name__ == '__main__':

    # root = Node('root')
    # model_path = "C:/Users/ChenPanda/Desktop/origin_inject_models.pth"
    # model = torch.load(model_path, map_location='cpu')
    #
    # makeTree(model,root,'')
    #
    # for pre, fill, node in RenderTree(root):
    #     print("%s%s" % (pre, node.name))


