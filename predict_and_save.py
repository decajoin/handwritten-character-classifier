import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

# 参数配置
NUM_CLASSES = 501
IMAGE_SIZE = 96
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 模型定义（保持和训练时一致）
def load_model(path):
    model = resnet18(num_classes=NUM_CLASSES)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# 数据加载和预处理
def load_test_loader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[227.439 / 255] * 3, std=[49.3361 / 255] * 3),
    ])
    dataset = datasets.ImageFolder(root='data/test', transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset.classes


# 获取预测结果
def get_predictions(model, loader):
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    return (
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        np.concatenate(all_labels)
    )


def main():
    print("[*] 加载模型...")
    model = load_model("best.pth")

    print("[*] 加载测试数据...")
    loader, class_names = load_test_loader()

    print("[*] 开始预测...")
    preds, probs, labels = get_predictions(model, loader)

    # 将预测结果保存到文件
    results = {
        'predictions': preds,
        'probabilities': probs,
        'true_labels': labels,
        'class_names': class_names
    }

    with open('prediction_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    print(f"[✔] 预测结果已保存至 prediction_results.pkl")
    print(f"    - 样本数量: {len(labels)}")
    print(f"    - 准确率: {np.mean(preds == labels) * 100:.2f}%")


if __name__ == "__main__":
    main()