import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# ------------------------ 超参数设置 ------------------------
BATCH_SIZE = 512        # 每个 batch 的图像数量
NUM_CLASSES = 501       # 分类数（目标类别总数）
NUM_EPOCHS = 100        # 总训练轮数
LR = 0.001              # 初始学习率
IMAGE_SIZE = 96         # 图像缩放尺寸
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU（若可用）

# ------------------------ 日志保存函数 ------------------------
def save_log(log_data, log_file='training_log.json'):
    # 将每轮训练的日志数据追加保存到 JSON 文件中
    with open(log_file, 'a') as f:
        json.dump(log_data, f)
        f.write("\n")

# ------------------------ 模型评估函数 ------------------------
def evaluate(model, dataloader):
    # 模型评估阶段不进行梯度计算
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)  # 取最大概率对应的类别作为预测
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total  # 返回准确率

# ------------------------ 主训练函数 ------------------------
def main():
    # 数据增强（训练集专用）
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),               # 缩放图像
        transforms.RandomCrop(IMAGE_SIZE, padding=4),              # 随机裁剪并填充
        transforms.RandomHorizontalFlip(),                         # 随机水平翻转
        transforms.RandomRotation(15),                             # 随机旋转
        transforms.ToTensor(),                                     # 转为 Tensor
        transforms.Normalize(mean=[227.439/255]*3, std=[49.3361/255]*3),  # 归一化
    ])

    # 验证集 / 测试集的数据预处理（无数据增强）
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[227.439/255]*3, std=[49.3361/255]*3),
    ])

    # 加载数据集（ImageFolder 目录结构应为 data/train/class_x/*.jpg）
    train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='data/val', transform=test_transform)
    test_dataset = datasets.ImageFolder(root='data/test', transform=test_transform)

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # ------------------------ 模型构建 ------------------------
    model = resnet18(num_classes=NUM_CLASSES)  # 加载预设 ResNet18 并替换输出层
    # 修改第一层卷积核以适应小图（96x96）
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # 去除下采样最大池化层
    model.to(DEVICE)  # 模型移动到 GPU

    # ------------------------ 损失函数与优化器 ------------------------
    criterion = nn.CrossEntropyLoss(reduction='mean')  # 交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)  # SGD 优化器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)  # 学习率调度器

    # 日志文件初始化
    log_file = 'training_log.json'

    # ------------------------ 开始训练 ------------------------
    best_acc = 0.0  # 保存最佳准确率
    for epoch in range(NUM_EPOCHS):
        model.train()  # 切换到训练模式
        running_loss = 0.0  # 当前 epoch 的累计 loss

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()           # 梯度清零
            outputs = model(imgs)           # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()                 # 反向传播
            optimizer.step()                # 更新参数

            running_loss += loss.item()     # 累加 loss

        scheduler.step()  # 更新学习率

        # 每轮结束后在验证集上评估
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存当前 epoch 的训练日志
        epoch_log_data = {
            "lr": optimizer.param_groups[0]['lr'],
            "loss": running_loss,
            "accuracy/top1": val_acc * 100,
            "epoch": epoch + 1,
            "iter": (epoch + 1) * len(train_loader)
        }
        save_log(epoch_log_data, log_file)

        # 保存验证集上最好的模型参数
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'base.pth')
            print("Saved Best Model")

    # ------------------------ 最终测试 ------------------------
    model.load_state_dict(torch.load('base.pth'))  # 加载最优模型
    test_acc = evaluate(model, test_loader)              # 测试集评估
    print(f"Final Test Accuracy: {test_acc:.4f}")

# 入口函数
if __name__ == "__main__":
    main()
