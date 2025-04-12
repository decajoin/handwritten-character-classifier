import json
import matplotlib.pyplot as plt


# ========================== 日志加载函数 ==========================
def load_log(file_path):
    epochs = []
    train_losses = []
    val_accuracies = []

    with open(file_path, 'r') as f:
        for line in f:
            log_data = json.loads(line.strip())  # 解析 JSON
            if 'loss' in log_data:  # 确保该条日志记录了损失
                epochs.append(log_data['epoch'])  # 当前 epoch 编号
                train_losses.append(log_data['loss'])  # 当前训练损失
                val_accuracies.append(log_data['accuracy/top1'])  # 当前验证准确率

    return epochs, train_losses, val_accuracies


# ========================== 可视化绘图函数 ==========================
def plot_training_curve(epochs, train_losses, val_accuracies, save_path):
    plt.figure(figsize=(12, 6))  # 设置画布大小

    # -------- 绘制训练损失曲线 --------
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    # -------- 绘制验证准确率曲线 --------
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.grid(True)

    # -------- 图像保存 --------
    plt.tight_layout()
    plt.savefig(f'{save_path}', dpi=300)
    print(f"[✔] loss 曲线已保存：{save_path}")
    # plt.show()  # 如果需要显示图像，可取消注释


# ========================== 主函数入口 ==========================
def main():
    # 从 JSON 日志中读取数据
    epochs, train_losses, val_accuracies = load_log('training_log.json')

    # 绘图并保存
    plot_training_curve(epochs, train_losses, val_accuracies, 'loss_and_val_loss_1.png')


# ========================== 程序启动 ==========================
if __name__ == "__main__":
    main()
