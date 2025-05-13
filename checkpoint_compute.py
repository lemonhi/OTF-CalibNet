import torch
from OTF_CalibNet import OTF_CalibNet
import torch
import torch.nn as nn

# 示例：定义一个简单的 PyTorch 模型


# 创建模型实例
model = OTF_CalibNet()

# 计算参数总数
total_params = sum(param.numel() for param in model.parameters())
print(f"Total number of parameters: {total_params}")