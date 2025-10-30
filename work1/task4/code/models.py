from typing import Type, Union, List, Optional
import torch
import torch.nn as nn
from torch import Tensor


# 基础残差块 (ResNet-18/34)
class BasicBlock(nn.Module):
    expansion: int = 1  # 输出通道扩展系数

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接处理
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 跳跃连接
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接 + 激活
        out += identity
        out = self.relu(out)

        return out

# 瓶颈残差块 (ResNet-50/101/152)
class Bottleneck(nn.Module):
    expansion: int = 4  # 输出通道扩展系数

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        # 1x1 降维卷积
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 空间卷积
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 升维卷积
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,  # 通道数扩展4倍
            kernel_size=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)    # 特别注意：这里没有将out_channels的值*4

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 跳跃连接
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

# ResNet 主网络
class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000
    ) -> None:
        super().__init__()
        self.in_channels = 64

        # 初始卷积层 (步长2下采样)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)     # 全连接层

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        # 需要下采样或通道数变化时创建跳跃连接
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        # 第一个块处理步长变化
        layers.append(block(
            self.in_channels,
            out_channels,
            stride,     # 除第一层外stride = 2
            downsample
        ))
        self.in_channels = out_channels * block.expansion

        # 后续块
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels
            ))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 预定义 ResNet 变种
def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes: int = 1000) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

def resnet101(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

def resnet152(num_classes: int = 1000) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)

# 测试代码
if __name__ == '__main__':
    model = resnet50()
    input_tensor = torch.randn(1, 3, 224, 224)  # 模拟ImageNet输入
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
