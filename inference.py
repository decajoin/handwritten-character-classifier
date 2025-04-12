import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms, datasets
from PIL import Image
import os
import json

# ========================== 基本设置 ==========================
IMAGE_SIZE = 96                      # 输入图像的尺寸
NUM_CLASSES = 501                   # 分类类别数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU

# ========================== 加载字符映射字典 ==========================
def load_char_dict(json_path):
    # 加载用于索引转字符的字典（如 '703' → '天'）
    with open(json_path, 'r', encoding='utf-8') as f:
        char_dict = json.load(f)
    return char_dict

# ========================== 图像预处理操作 ==========================
def get_transform():
    # 推理时使用的图像处理方式，与验证阶段保持一致
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[227.439 / 255] * 3, std=[49.3361 / 255] * 3)
    ])

# ========================== 构建模型并加载权重 ==========================
def load_model(weights_path):
    # 创建 ResNet18 模型并加载预训练权重
    model = resnet18(num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 替换第一层以适配小图
    model.maxpool = nn.Identity()  # 移除最大池化层
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()  # 设置为评估模式
    return model

# ========================== 获取类别索引到目录名的映射 ==========================
def get_class_mapping(data_dir):
    # 利用 ImageFolder 的 class_to_idx 构建 idx → class 名字的映射
    dataset = datasets.ImageFolder(root=data_dir)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    return idx_to_class

# ========================== 推理函数 ==========================
def predict(image_path, model, transform, idx_to_class=None):
    # 对单张图片进行推理，输出预测类别和置信度
    image = Image.open(image_path).convert('RGB')  # 打开图像并转为 RGB
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # 增加 batch 维度并送入设备

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)  # 获取分类概率
        top_prob, top_idx = torch.max(probs, dim=1)

    pred_idx = top_idx.item()
    confidence = top_prob.item()
    pred_class = idx_to_class[pred_idx] if idx_to_class else str(pred_idx)

    return pred_class, confidence

# ========================== 主程序入口 ==========================
def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Image Classification Inference")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--weights', type=str, default='best.pth', help="Path to model weights")
    parser.add_argument('--data_dir', type=str, default='data/train', help="Path to dataset for class names")
    args = parser.parse_args()

    # --------------------- 路径合法性检查 ---------------------
    assert os.path.exists(args.image), f"Image not found: {args.image}"
    assert os.path.exists(args.weights), f"Weights not found: {args.weights}"

    # --------------------- 模型加载和预测流程 ---------------------
    transform = get_transform()
    model = load_model(args.weights)
    idx_to_class = get_class_mapping(args.data_dir)

    pred_class, confidence = predict(args.image, model, transform, idx_to_class)

    # --------------------- 加载并查找对应的字符 ---------------------
    char_dict = load_char_dict('Char_dict.json')  # 读取字符映射字典

    class_key = str(int(pred_class))  # 去除前导零，例如 "00703" → "703"
    unicode_char = char_dict.get(class_key, "[UNK]")  # 查不到返回 [UNK]
    clean_char = unicode_char.replace('\u0000', '')  # 去除异常 NULL 字符

    # --------------------- 打印推理结果 ---------------------
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence Score: {confidence:.4f}")
    print(f"Predicted Character: {clean_char}")

# ========================== 程序启动 ==========================
if __name__ == "__main__":
    main()
