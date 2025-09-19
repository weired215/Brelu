# import tensorflow.compat.v1 as tf
# from keras import backend as K
# import os
# import keras
# from keras.layers import Activation, Lambda, ReLU
# from keras.models import Model
from brelu.buld_tree import makeTree, is_not_basic_module, get_next_entity, get_entity, get_relu_leaf_name
import torch
import torch.nn as nn
import numpy as np
from anytree import Node, RenderTree, PreOrderIter


def get_previous_layer(model, relu_index):
    layers = list(model.children())
    return layers[relu_index - 1] if relu_index > 0 else None


class chRelu(nn.Module):

    def __init__(self, boundary):
        super(chRelu, self).__init__()
        self.boundary = boundary

    def forward(self, x):
        length = x.shape[1]

        if len(x.shape) == 4:
            for j in range(length):
                x[:, j, :, :] = x[:, j, :, :].clamp(0, self.boundary[j])
        if len(x.shape) == 2:
            for j in range(length):
                x[:, j] = x[:, j].clamp(0, self.boundary[j])
        return x


class BRelu(nn.Module):

    def __init__(self, boundary):
        super(BRelu, self).__init__()
        self.boundary = boundary

    def forward(self, x):
        x = torch.clamp(x, 0, self.boundary)
        return x


class FRelu(nn.Module):

    def __init__(self, boundary):
        super(FRelu, self).__init__()
        self.boundary = boundary
        self.Tmax = self.boundary * pow(2, 64)

    def threshold_tensor(self, tensor, boundary):
        # 使用torch.clamp函数将张量中的值限制在[0, boundary]的范围内
        # 然后用torch.where函数将小于boundary的值替换为0
        result_tensor = torch.where(tensor > boundary, torch.tensor(0.0), tensor)
        return result_tensor

    def forward(self, x):
        x = torch.where((self.boundary * 2 > x) & (x > self.boundary), torch.tensor(self.boundary), x)
        x = torch.where((self.Tmax > x) & (x > self.boundary * 2), torch.tensor(2), x)
        x = torch.where(x > self.Tmax, torch.tensor(0.0), x)
        x = torch.where(x < 0, torch.tensor(0.0), x)
        return x


class Clipper(nn.Module):
    def __init__(self, boundary):
        super(Clipper, self).__init__()
        self.boundary = boundary

    def forward(self, x):
        x = torch.where(x > self.boundary, torch.tensor(0.), x)
        x = torch.where(x < 0, torch.tensor(0.), x)
        return x


def fix(model, boundarys: list, method):
    root = Node('root')
    makeTree(model, root, '')
    acts = get_relu_leaf_name(model, root)
    return replace_intermediate_layer_in_pytorch(model, acts, boundarys, method)


def ch_hook(boundarys, i):
    def ch_activation(model, input, output):
        length = output.shape[1]
        output = torch.cat([output[:, j, :, :].clamp(0, boundarys[i][j]) for j in range(length)], dim=1)
        return output

    return ch_activation


def hook(boundarys, i):
    def activation(model, input, output):
        output = torch.clamp(output, 0, boundarys[i])
        return output

    return activation


def replace_intermediate_layer_in_pytorch(model, acts, boundarys, method):  # 替代某层
    layers = list(model.children())
    length = len(layers)
    length2 = len(acts)
    new_model = model
    # 这里需要判断layer本身是否像Sequential(conv,relu,conv,relu)一样存在嵌套关系
    # 目前支持多嵌套
    # resnet 为例
    # 例如嵌套关系 resnet_layer1_0_conv1

    if len(torch.tensor(boundarys).shape) == 2:
        print("说明是通道置换")
        for i in range(length2):
            index = acts[i].split('_')
            index = index[len(index) - 1]
            entity = get_entity(new_model, acts[i])
            setattr(entity, index, chRelu(boundarys[i]))

    else:
        print("说明是层置换")
        for i in range(length2):
            index = acts[i].split('_')
            index = index[len(index) - 1]
            entity = get_entity(new_model, acts[i])

            if method == 'FRelu':
                setattr(entity, index, FRelu(boundarys[i]))
            if method == 'BRelu':
                setattr(entity, index, BRelu(boundarys[i]))
            if method == 'Clipper':
                setattr(entity, index, Clipper(boundarys[i]))
    return new_model  # 由于只修改了


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    Args:
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    Returns:
        A tensor.
    """
    return torch.as_tensor(x, dtype=dtype)
