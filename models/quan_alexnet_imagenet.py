import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .quantization import *

__all__ = ['AlexNet', 'alexnet_quan']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):  # 修改输出类别数为10
        super(AlexNet, self).__init__()
        # 修改特征提取层结构
        self.features = nn.Sequential(
            quan_Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 减小卷积核和步长
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 减小池化核
            quan_Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            quan_Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            quan_Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            quan_Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 修改为全局平均池化
        self.classifier = nn.Sequential(
            nn.Dropout(),
            quan_Linear(256, 4096),  # 输入维度匹配全局池化输出
            nn.ReLU(inplace=True),
            nn.Dropout(),
            quan_Linear(4096, 4096),
            nn.ReLU(inplace=True),
            quan_Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 自动展平
        x = self.classifier(x)
        return x

def alexnet_quan(pretrained=False, **kwargs):  # 默认不加载ImageNet预训练权重
    model = AlexNet(**kwargs)
    if pretrained:
        # pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        pretrained_dict=torch.load('/home/liaolei.pan/code1/PBS/models/pth/alexnet.pth')
        model_dict = model.state_dict()
        # 只加载能匹配的权重（可能部分层不匹配）
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                         if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model