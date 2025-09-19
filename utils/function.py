import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from anytree import Node
from .build_tree import makeTree,get_relu_leaf_name,get_entity




def get_layer_output(model, layer_idx, input_data):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    layers = list(model.modules())

    hook = layers[layer_idx].register_forward_hook(get_activation('layer_output'))

    with torch.no_grad():
        model(input_data)

    hook.remove()

    return activation['layer_output']


def compute_perturbed_layer_output(model, layer_idx, input_data, epsilon=1e-5):
    """计算受扰动后的层输出"""
    layers = list(model.modules())

    original_weight = layers[layer_idx].weight.detach().clone()
    original_bias = layers[layer_idx].bias.detach().clone() if layers[layer_idx].bias is not None else None

    # 添加扰动
    with torch.no_grad():
        layers[layer_idx].weight.add_(epsilon * torch.randn_like(layers[layer_idx].weight))
        if layers[layer_idx].bias is not None:
            layers[layer_idx].bias.add_(epsilon * torch.randn_like(layers[layer_idx].bias))

    perturbed_output = get_layer_output(model, layer_idx, input_data)

    with torch.no_grad():
        layers[layer_idx].weight.copy_(original_weight)
        if layers[layer_idx].bias is not None:
            layers[layer_idx].bias.copy_(original_bias)
    return perturbed_output
def get_class_scores(model, input_data):
    with torch.no_grad():
        return model(input_data)


def compute_lambda_k(model, dataset, layer_idx, alpha=0.05, beta=5, N=1, epsilon=1e-5, batch_size=32,
                     proximity_factor=0.1, min_margin=1e-4):
    """
    参数:
        model: 神经网络模型
        dataset: 数据集
        layer_idx: 要计算的层索引
        alpha: 学习率
        beta: 耐心参数
        N: 最大迭代次数
        epsilon: 扰动强度
        batch_size: 批处理大小
        proximity_factor: 接近因子，控制向brelu靠近的强度
        min_margin: lambda_result 与 max_output 之间的最小差值
    """
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device('cpu')

    max_omega_hat_k = 0.0
    global_max_output = -float('inf')
    all_original_outputs = []
    all_perturbed_outputs = []


    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        original_output = get_layer_output(model, layer_idx, inputs)
        perturbed_output = compute_perturbed_layer_output(model, layer_idx, inputs, epsilon)

        batch_max = torch.max(original_output).item()
        if batch_max > global_max_output:
            global_max_output = batch_max

        delta = torch.norm(perturbed_output - original_output, p=2, dim=1).max().item()
        if delta > max_omega_hat_k:
            max_omega_hat_k = delta

        all_original_outputs.append(original_output)
        all_perturbed_outputs.append(perturbed_output)

    initial_offset = max(min_margin, proximity_factor * max_omega_hat_k)
    lambda_k = global_max_output + initial_offset
    bound_stack = [lambda_k]
    lr = alpha
    patience = beta

    IterCount = 1
    sum_stack = []

    while IterCount <= N:
        sum_val = 1.0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            original_output = all_original_outputs[batch_idx]
            batch_max_output = torch.max(original_output).item()

            boundary_gradient = 1.0

            deviation = lambda_k - batch_max_output
            class_scores = get_class_scores(model, inputs)

            batch_size = inputs.size(0)
            for i in range(batch_size):
                true_label = labels[i].item()
                true_score = class_scores[i, true_label].item()

                max_error_score = -float('inf')
                for j in range(class_scores.size(1)):
                    if j != true_label and class_scores[i, j].item() > max_error_score:
                        max_error_score = class_scores[i, j].item()

                result = (true_score + boundary_gradient * deviation) - max_error_score
                sum_val *= result

        sum_stack.append(sum_val)

        if len(sum_stack) > 1:
            if sum_stack[-1] < sum_stack[-2]:
                patience -= 1
                if patience == 0:
                    final_lambda = max(bound_stack[0], global_max_output + min_margin)
                    return final_lambda

        h = 1e-5
        lambda_k_plus_h = lambda_k + h
        sum_plus_h = 1.0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            original_output = all_original_outputs[batch_idx]
            batch_max_output = torch.max(original_output).item()
            deviation = lambda_k_plus_h - batch_max_output

            class_scores = get_class_scores(model, inputs)

            batch_size = inputs.size(0)
            for i in range(batch_size):
                true_label = labels[i].item()
                true_score = class_scores[i, true_label].item()

                max_error_score = -float('inf')
                for j in range(class_scores.size(1)):
                    if j != true_label and class_scores[i, j].item() > max_error_score:
                        max_error_score = class_scores[i, j].item()

                result = (true_score + boundary_gradient * deviation) - max_error_score
                sum_plus_h *= result

        objective_gradient = (sum_plus_h - sum_val) / h
        lambda_k += objective_gradient * lr


        bound_stack.append(lambda_k)
        IterCount += 1

    return bound_stack[0]

def GABO(model,dataset,layer_idx):
    root = Node('root')
    makeTree(model, root, '')
    acts = get_relu_leaf_name(model, root)
    if len(acts) == 0:
        return []
    print("需要修改的层数：{}".format(len(acts)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    lambda_result = compute_lambda_k(model, dataset, layer_idx=layer_idx, min_margin=1e-4)
    return lambda_result


