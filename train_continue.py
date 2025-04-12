import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# ========================== 超参数设置 ==========================
BATCH_SIZE = 8               # 批次大小
NUM_CLASSES = 501            # 分类类别数量
START_EPOCH = 100            # 从第 100 轮开始继续训练
END_EPOCH = 160              # 训练到第 160 轮为止
LR = 0.01                    # 学习率
IMAGE_SIZE = 96              # 输入图像大小
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU

# ========================== 日志保存函数 ==========================
def save_log(log_data, log_file='training_log_continue.json'):
    # 将每轮训练的日志信息追加写入 JSON 文件
    with open(log_file, 'a') as f:
        json.dump(log_data, f)
        f.write("\n")

# ========================== 验证函数 ==========================
def evaluate(model, dataloader):
    # 在验证/测试集上评估模型准确率
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ========================== 主函数 ==========================
def main():
    # --------------------- 数据增强（训练集） ---------------------
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[227.439/255]*3, std=[49.3361/255]*3),
    ])

    # --------------------- 验证/测试集预处理 ---------------------
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[227.439/255]*3, std=[49.3361/255]*3),
    ])

    # --------------------- 加载数据集 ---------------------
    train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
    val_dataset = datasets.ImageFolder(root='data/val', transform=test_transform)
    test_dataset = datasets.ImageFolder(root='data/test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # --------------------- 构建模型并修改结构 ---------------------
    model = resnet18(num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR风格修改
    model.maxpool = nn.Identity()  # 移除最大池化
    model.to(DEVICE)

    # --------------------- 加载已有最优模型参数 ---------------------
    model.load_state_dict(torch.load('base.pth'))
    print("Loaded base.pth")

    # --------------------- 设置损失函数、优化器和调度器 ---------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[130, 140], gamma=0.1)  # 学习率衰减点

    best_acc = 0.0  # 记录最优准确率
    for epoch in range(START_EPOCH, END_EPOCH):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{END_EPOCH}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()  # 更新学习率
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Val Acc: {val_acc:.4f}")

        # --------------------- 保存日志 ---------------------
        log_data = {
            "lr": optimizer.param_groups[0]['lr'],
            "loss": running_loss,
            "accuracy/top1": val_acc * 100,
            "epoch": epoch + 1,
            "iter": (epoch + 1) * len(train_loader)
        }
        save_log(log_data)

        # --------------------- 保存最优模型参数 ---------------------
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best.pth')
            print("Saved Improved Model")

    # --------------------- 最终测试精度 ---------------------
    model.load_state_dict(torch.load('best.pth'))
    test_acc = evaluate(model, test_loader)
    print(f"Final Test Accuracy: {test_acc:.4f}")

# 运行主函数
if __name__ == "__main__":
    main()
