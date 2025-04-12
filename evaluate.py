import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader

# ========================== 超参数设置 ==========================
BATCH_SIZE = 128                # 批大小
NUM_CLASSES = 501              # 类别数
IMAGE_SIZE = 96                # 输入图像尺寸
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU

# ========================== Top-k准确率计算函数 ==========================
def top_k_accuracy(output, target, k=1):
    """计算Top-k准确率"""
    _, pred = output.topk(k, 1, True, True)  # 获取前k个预测结果的索引
    correct = pred.eq(target.view(-1, 1).expand_as(pred))  # 判断预测是否与真实标签一致（支持广播）
    correct_k = correct.sum().item()  # 累加正确预测数量
    return correct_k

# ========================== 模型评估函数 ==========================
def evaluate(model, dataloader):
    """在给定dataloader上评估模型的Top-1和Top-5准确率"""
    model.eval()  # 设置为评估模式
    total = 0
    top1_correct = 0
    top5_correct = 0

    with torch.no_grad():  # 推理阶段不计算梯度，节省内存
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)

            # 计算Top-1和Top-5正确个数
            top1_correct += top_k_accuracy(outputs, labels, k=1)
            top5_correct += top_k_accuracy(outputs, labels, k=5)

            total += labels.size(0)  # 统计总样本数

    # 计算准确率
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    return top1_acc, top5_acc

# ========================== 加载模型 ==========================
def load_model(path):
    """构建模型并加载训练好的权重"""
    model = resnet18(num_classes=501)  # 创建 ResNet18 模型（输出类别数为501）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 修改输入层适应小图（96x96）
    model.maxpool = nn.Identity()  # 移除最大池化层（CIFAR风格）
    model.load_state_dict(torch.load(path, map_location=DEVICE))  # 加载权重
    model.to(DEVICE)
    model.eval()
    return model

# ========================== 主程序入口 ==========================
def main():
    # -------- 数据预处理 --------
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 缩放图像
        transforms.ToTensor(),                        # 转为Tensor
        transforms.Normalize(mean=[227.439/255]*3, std=[49.3361/255]*3),  # 标准化（使用训练集统计量）
    ])

    # -------- 加载测试数据 --------
    test_dataset = datasets.ImageFolder(root='data/test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # -------- 加载模型 --------
    model = load_model('best.pth')

    # -------- 评估模型 --------
    top1_acc, top5_acc = evaluate(model, test_loader)

    # -------- 打印结果 --------
    print(f"Final Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Final Top-5 Accuracy: {top5_acc:.4f}")

# ========================== 启动程序 ==========================
if __name__ == "__main__":
    main()
