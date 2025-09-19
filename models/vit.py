import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from .quantization  import quan_Conv2d,quan_Linear
import matplotlib.pyplot as plt

__all__ = [
    "ViT"
]

# # 1. 数据准备
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# train_dataset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# test_dataset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)

# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
# test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


# 2. 定义ViT模型 (简化版)
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, N]
        x = x.transpose(1, 2)  # [B, N, E]
        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=64, depth=6, n_heads=8, mlp_ratio=4,
                 num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # 类别token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # 1. 分块嵌入
        x = self.patch_embed(x)  # [B, N, E]

        # 2. 添加类别token
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, E]
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, E]

        # 3. 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)

        # 4. Transformer编码
        x = x.transpose(0, 1)  # [N+1, B, E] (PyTorch Transformer要求序列维度在前)
        x = self.encoder(x)
        x = x.transpose(0, 1)  # [B, N+1, E]

        # 5. 分类
        x = self.norm(x)
        cls_token_final = x[:, 0]  # 取类别token
        logits = self.head(cls_token_final)

        return logits


# # 3. 初始化模型、优化器和损失函数
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ViT(
#     img_size=32,
#     patch_size=4,
#     embed_dim=128,
#     depth=6,
#     n_heads=8,
#     mlp_ratio=4,
#     num_classes=10
# )
# pretrained_dict = torch.load('models/pth/vit_train_2.pth')
# model_dict = model.state_dict()
# pretrained_dict = {
#     k: v
#     for k, v in pretrained_dict.items() if k in model_dict
# }
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)
# for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d)  :
#             attributes = name.split('.')
#             sub_module = model
#             for attr in attributes[:-1]:  # 遍历到倒数第二个属性
#                 sub_module = getattr(sub_module, attr)
#             print(sub_module)
#             setattr(sub_module, attributes[-1],quan_Conv2d(module.in_channels,module.out_channels,module.kernel_size,
#                                              stride=module.stride,padding=module.padding,
#                                              dilation=module.dilation,groups=module.groups))
#
# for name, module in model.named_modules():
#         if isinstance(module, nn.Linear)  :
#             attributes = name.split('.')
#             sub_module = model
#             for attr in attributes[:-1]:  # 遍历到倒数第二个属性
#                 sub_module = getattr(sub_module, attr)
#             print(sub_module)
#             setattr(sub_module, attributes[-1],quan_Linear(module.in_features,module.out_features))


# model=model.to(device)
# optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
# criterion = nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# # 4. 训练函数
# def train(epoch):
#     model.train()
#     train_loss = 0
#     correct = 0
#     total = 0

#     for batch_idx, (inputs, targets) in enumerate(train_loader):
#         inputs, targets = inputs.to(device), targets.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#         if batch_idx % 100 == 0:
#             print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} '
#                   f'| Loss: {loss.item():.4f} | Acc: {100. * correct / total:.2f}%')

#     return train_loss / (batch_idx + 1), 100. * correct / total


# # 5. 测试函数
# def test(epoch):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(test_loader):
#             inputs, targets = inputs.to(device), targets.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     acc = 100. * correct / total
#     print(f'Test Epoch: {epoch} | Loss: {test_loss / (batch_idx + 1):.4f} | Acc: {acc:.2f}%')
#     return test_loss / (batch_idx + 1), acc

if __name__ == '__main__':

    # 6. 训练循环
    epochs = 15
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    test(1)
    # for epoch in range(epochs):
    #     loss, acc = train(epoch)
    #     train_losses.append(loss)
    #     train_accs.append(acc)
    #
    #     loss, acc = test(epoch)
    #     test_losses.append(loss)
    #     test_accs.append(acc)
    #
    #     scheduler.step()
    # torch.save(model.state_dict(),"vit_train_2.pth")
