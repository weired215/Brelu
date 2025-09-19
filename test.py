import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_layer_output(model, layer_idx, input_data):
    """获取指定层的输出结果"""
    # 存储中间层输出的钩子函数
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # 注册钩子
    layers = list(model.children())
    hook = layers[layer_idx].register_forward_hook(get_activation('layer_output'))

    # 前向传播以获取输出
    with torch.no_grad():
        model(input_data)

    # 移除钩子
    hook.remove()

    return activation['layer_output']


def compute_perturbed_layer_output(model, layer_idx, input_data, epsilon=1e-5):
    """计算受扰动后的层输出"""
    # 保存原始权重用于恢复
    layers = list(model.children())
    original_weight = layers[layer_idx].weight.detach().clone()
    original_bias = layers[layer_idx].bias.detach().clone() if layers[layer_idx].bias is not None else None

    # 添加扰动
    with torch.no_grad():
        layers[layer_idx].weight.add_(epsilon * torch.randn_like(layers[layer_idx].weight))
        if layers[layer_idx].bias is not None:
            layers[layer_idx].bias.add_(epsilon * torch.randn_like(layers[layer_idx].bias))

    # 获取扰动后的输出
    perturbed_output = get_layer_output(model, layer_idx, input_data)

    # 恢复原始权重
    with torch.no_grad():
        layers[layer_idx].weight.copy_(original_weight)
        if layers[layer_idx].bias is not None:
            layers[layer_idx].bias.copy_(original_bias)

    return perturbed_output


def get_class_scores(model, input_data):
    """获取模型的分类分数"""
    with torch.no_grad():
        return model(input_data)


def compute_lambda_k(model, dataset, layer_idx, alpha=0.05, beta=5, N=100, epsilon=1e-5, batch_size=32,
                     proximity_factor=0.1):
    """
    计算第k层的上界lambda^(k)，结果更靠近输出最大值

    参数:
        model: 神经网络模型
        dataset: 数据集
        layer_idx: 要计算的层索引
        alpha: 学习率（减小以避免远离最大值）
        beta: 耐心参数
        N: 最大迭代次数
        epsilon: 扰动强度
        batch_size: 批处理大小
        proximity_factor: 接近因子，控制向最大值靠近的强度
    """
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 确定设备
    device = next(model.parameters()).device

    # 计算最大扰动差和全局最大输出值
    max_omega_hat_k = 0.0
    global_max_output = -float('inf')  # 记录整个数据集上该层的最大输出值
    all_original_outputs = []
    all_perturbed_outputs = []

    for inputs, _ in dataloader:
        inputs = inputs.to(device)

        # 获取原始输出和扰动输出
        original_output = get_layer_output(model, layer_idx, inputs)
        perturbed_output = compute_perturbed_layer_output(model, layer_idx, inputs, epsilon)

        # 计算当前批次的最大值
        batch_max = torch.max(original_output).item()
        if batch_max > global_max_output:
            global_max_output = batch_max

        # 计算扰动差
        delta = torch.norm(perturbed_output - original_output, p=2, dim=1).max().item()
        if delta > max_omega_hat_k:
            max_omega_hat_k = delta

        all_original_outputs.append(original_output)
        all_perturbed_outputs.append(perturbed_output)

    # 初始化lambda_k，使其更接近全局最大值
    # 结合全局最大值和扰动差，偏向于最大值
    lambda_k = global_max_output + proximity_factor * max_omega_hat_k
    bound_stack = [lambda_k]
    lr = alpha
    patience = beta

    # 迭代优化lambda_k
    IterCount = 1
    sum_stack = []

    while IterCount <= N:
        sum_val = 1.0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 获取当前批次的原始输出
            original_output = all_original_outputs[batch_idx]
            batch_max_output = torch.max(original_output).item()

            # 计算边界梯度
            boundary_gradient = 1.0

            # 计算偏差
            deviation = lambda_k - batch_max_output

            # 获取分类分数
            class_scores = get_class_scores(model, inputs)

            # 计算正确类别与最大错误类别的差值
            batch_size = inputs.size(0)
            for i in range(batch_size):
                true_label = labels[i].item()
                true_score = class_scores[i, true_label].item()

                # 找到最大的错误类别分数
                max_error_score = -float('inf')
                for j in range(class_scores.size(1)):
                    if j != true_label and class_scores[i, j].item() > max_error_score:
                        max_error_score = class_scores[i, j].item()

                # 计算带扰动的差值
                result = (true_score + boundary_gradient * deviation) - max_error_score
                sum_val *= result

        sum_stack.append(sum_val)

        # 检查收敛情况
        if len(sum_stack) > 1:
            if sum_stack[-1] < sum_stack[-2]:
                patience -= 1
                if patience == 0:
                    return bound_stack[0]

        # 计算目标梯度 (使用数值方法)
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

        # 更新lambda_k，同时添加向全局最大值靠近的约束
        objective_gradient = (sum_plus_h - sum_val) / h
        lambda_k += objective_gradient * lr

        # 添加正则化项，使lambda_k保持在全局最大值附近
        # 如果远离全局最大值，会被拉回
        distance_from_max = lambda_k - global_max_output
        if abs(distance_from_max) > max_omega_hat_k:
            lambda_k = global_max_output + np.sign(distance_from_max) * max_omega_hat_k

        bound_stack.append(lambda_k)
        IterCount += 1

    return lambda_k


# 使用示例
if __name__ == "__main__":
    device='cuda'
    # 创建示例模型和数据集
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(20, 10)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Linear(10, 5)
            self.relu2 = nn.ReLU()
            self.fc = nn.Linear(5, 3)

        def forward(self, x):
            x = self.relu1(self.layer1(x))
            x = self.relu2(self.layer2(x))
            x = self.fc(x)
            return x


    # 创建随机数据集
    class RandomDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.data = torch.randn(size, 20)
            self.labels = torch.randint(0, 3, (size,))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]


    # 初始化模型和数据集
    model = SimpleModel()
    dataset = RandomDataset()

    # 计算第1层(relu1)的上界
    lambda_result = compute_lambda_k(model, dataset, layer_idx=4)

    # 计算该层的最大输出值用于对比
    sample_input = torch.randn(1, 20)
    layer_output = get_layer_output(model, 1, sample_input)
    max_output = torch.max(layer_output).item()

    print(f"计算得到的第1层上界lambda^(1): {lambda_result}")
    print(f"该层对应的brelu: {max_output}")
    print(f"差值: {abs(lambda_result - max_output)}")
