import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DoubleConv(nn.Module):
    """双卷积块：两个卷积层，每个后面跟着ReLU激活函数"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化双卷积块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.conv(x)


class EncoderBlock(nn.Module):
    """编码器块：双卷积 + 最大池化"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        初始化编码器块
        
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super(EncoderBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            (池化后特征, 池化前特征)，池化前特征用于跳跃连接
        """
        features = self.double_conv(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """解码器块：上采样 + 连接 + 双卷积"""
    
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        初始化解码器块
        
        Args:
            in_channels: 输入通道数
            skip_channels: 跳跃连接通道数
            out_channels: 输出通道数
        """
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.double_conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)
        
    def forward(self, x: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征
            skip_features: 来自编码器的跳跃连接特征
        """
        x = self.up(x)
        x = torch.cat([x, skip_features], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net模型用于图像分割
    
    输入为2通道图像（超声图像 + 预测掩码），输出为1通道二值分割掩码
    """
    
    def __init__(self, in_channels: int = 2, out_channels: int = 1, features: List[int] = [64, 128, 256, 512]):
        """
        初始化U-Net模型
        
        Args:
            in_channels: 输入通道数，默认为2
            out_channels: 输出通道数，默认为1（二值分割）
            features: 每个编码器层的特征通道数列表
        """
        super(UNet, self).__init__()
        
        # 编码器部分
        self.encoder_blocks = nn.ModuleList()
        for feature in features:
            self.encoder_blocks.append(EncoderBlock(in_channels, feature))
            in_channels = feature
        
        # 底部（最深层）
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        
        # 解码器部分
        self.decoder_blocks = nn.ModuleList()
        features = features[::-1]  # 反转特征列表
        for i, feature in enumerate(features):
            self.decoder_blocks.append(
                DecoderBlock(
                    feature * 2 if i == 0 else feature * 2,
                    feature if i == len(features) - 1 else features[i + 1],
                    feature
                )
            )
        
        # 最终输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像，形状为[B, 2, H, W]
            
        Returns:
            分割掩码，形状为[B, 1, H, W]
        """
        # 存储编码器特征用于跳跃连接
        skip_connections = []
        
        # 编码器前向传播
        for encoder in self.encoder_blocks:
            x, features = encoder(x)
            skip_connections.append(features)
        
        # 底部处理
        x = self.bottleneck(x)
        
        # 反转跳跃连接列表以匹配解码器顺序
        skip_connections = skip_connections[::-1]
        
        # 解码器前向传播
        for idx, decoder in enumerate(self.decoder_blocks):
            x = decoder(x, skip_connections[idx])
        
        # 最终输出，不包含激活函数（返回logits）
        return self.final_conv(x)

