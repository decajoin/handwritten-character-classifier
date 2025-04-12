import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from sklearn.metrics import precision_recall_fscore_support


# 设置中文字体支持
def set_chinese_font():
    # 尝试设置系统中可能存在的中文字体
    try:
        # Windows系统可能的中文字体
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',  # 黑体
            'C:/Windows/Fonts/simsun.ttc',  # 宋体
            'C:/Windows/Fonts/msyh.ttc',  # 微软雅黑
            # Linux/Mac可能的中文字体
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
        ]

        for font_path in font_paths:
            try:
                font_prop = FontProperties(fname=font_path)
                mpl.rcParams['font.family'] = font_prop.get_name()
                print(f"[✓] 成功加载中文字体: {font_path}")
                return True
            except:
                continue

        # 如果找不到特定的中文字体文件，尝试使用系统内置的sans-serif字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS',
                                           'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("[!] 未找到指定中文字体文件，尝试使用系统默认sans-serif字体")
        return True

    except Exception as e:
        print(f"[!] 设置中文字体失败: {e}")
        print("[!] 将使用默认字体，中文可能无法正确显示")
        return False


def load_results(file_path='prediction_results.pkl'):
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    return results


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names=None, n_classes=5, save_path='confusion_matrix.png'):
    # 只选取前n_classes类的样本
    mask = (y_true < n_classes) & (y_pred < n_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    class_names_filtered = class_names[:n_classes] if class_names else None

    # 如果类名太长，截断以避免重叠
    if class_names_filtered:
        class_names_filtered = [name[:15] + '...' if len(name) > 15 else name for name in class_names_filtered]

    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    plt.figure(figsize=(16, 14))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_filtered)
    disp.plot(cmap='Blues', xticks_rotation=90)
    plt.title(f"Confusion Matrix (前 {n_classes} 类)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✔] 混淆矩阵已保存：{save_path}")
    plt.close()



# 绘制Top-K准确率
def plot_top_k_accuracy(y_true, y_probs, save_path='top_k_accuracy.png', max_k=10):
    top_k_accuracy = []
    k_values = list(range(1, max_k + 1))

    for k in k_values:
        # 获取每个样本的top-k预测类别
        top_k_preds = np.argsort(-y_probs, axis=1)[:, :k]
        # 检查真实标签是否在top-k预测中
        correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
        # 计算top-k准确率
        accuracy = np.mean(correct) * 100
        top_k_accuracy.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, top_k_accuracy, marker='o', linestyle='-', linewidth=2)
    plt.xlabel('K Value')
    plt.ylabel('Accuracy (%)')
    plt.title('Top-K Accuracy')
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✔] Top-K准确率图已保存：{save_path}")
    plt.close()

# 绘制 Precision / Recall / F1-score 图表（前 n_classes）
def plot_precision_recall_f1(y_true, y_pred, class_names=None, n_classes=25, save_path='prf_scores.png'):
    # 只保留前n_classes类
    mask = (y_true < n_classes) & (y_pred < n_classes)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    class_names_filtered = class_names[:n_classes] if class_names else [f"Class {i}" for i in range(n_classes)]
    class_names_filtered = [name[:15] + '...' if len(name) > 15 else name for name in class_names_filtered]

    precision, recall, f1, _ = precision_recall_fscore_support(y_true_filtered, y_pred_filtered, labels=np.arange(n_classes), zero_division=0)

    x = np.arange(n_classes)
    width = 0.25

    plt.figure(figsize=(18, 8))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')

    plt.xlabel('类别')
    plt.ylabel('得分')
    plt.title('前 %d 类 Precision / Recall / F1-score' % n_classes)
    plt.xticks(x, class_names_filtered, rotation=90)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[✔] Precision / Recall / F1-score 图已保存：{save_path}")
    plt.close()


# 分析误分类最多的类别
def analyze_misclassifications(y_true, y_pred, class_names=None):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 获取对角线元素（正确分类的样本数）
    correct_samples = np.diag(cm)

    # 计算每个类别的总样本数
    class_samples = np.sum(cm, axis=1)

    # 计算每个类别的错误率
    error_rates = 1 - correct_samples / class_samples

    # 获取错误率最高的十个类别
    top_error_indices = np.argsort(error_rates)[-10:][::-1]

    print("\n[*] 错误率最高的 10 个类别:")
    print(f"{'类别索引':<10}{'类别名称':<20}{'错误率 (%)':<15}{'样本数':<10}")
    print("-" * 55)

    for idx in top_error_indices:
        class_name = class_names[idx] if class_names and idx < len(class_names) else f"Class {idx}"
        if len(class_name) > 18:
            class_name = class_name[:15] + "..."
        print(f"{idx:<10}{class_name:<20}{error_rates[idx] * 100:<15.2f}{class_samples[idx]:<10}")

def main():
    # 设置中文字体支持
    set_chinese_font()

    print("[*] 加载预测结果...")
    results = load_results()

    preds = results['predictions']
    probs = results['probabilities']
    labels = results['true_labels']
    class_names = results.get('class_names', None)

    print(f"[*] 样本数量: {len(labels)}")
    print(f"[*] 准确率: {np.mean(preds == labels) * 100:.2f}%")

    # 绘制混淆矩阵
    print("[*] 绘制混淆矩阵（前 25 类）...")
    plot_confusion_matrix(labels, preds, class_names, n_classes=25)

    # 绘制Top-K准确率
    print("[*] 绘制Top-K准确率...")
    plot_top_k_accuracy(labels, probs)

    # 绘制 Precision / Recall / F1-score 图
    print("[*] 绘制 Precision / Recall / F1-score 图（前 10 类）...")
    plot_precision_recall_f1(labels, preds, class_names, n_classes=10)

    # 输出混淆矩阵中的前十个误分类最多的类别
    print("[*] 分析误分类情况...")
    analyze_misclassifications(labels, preds, class_names)



if __name__ == "__main__":
    main()