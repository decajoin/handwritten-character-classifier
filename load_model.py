import torch
import torch.nn as nn
from torchvision.models import resnet18


def load_model(model_path):
    """
    加载训练好的模型

    Args:
        model_path (str): 模型文件路径，如'base.pth'

    Returns:
        torch.nn.Module: 加载好的PyTorch模型
    """
    # 创建与训练时相同的模型结构
    NUM_CLASSES = 501
    model = resnet18(num_classes=NUM_CLASSES)

    # 应用CIFAR风格的修改
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # 加载权重
    try:
        # 尝试加载权重
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"成功加载模型权重: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        raise

    return model


# 测试加载函数
model = load_model('best.pth')
print(model)  # 打印模型结构
print("模型加载成功!")