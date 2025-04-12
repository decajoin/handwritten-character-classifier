# README

[数据集下载](https://pan.baidu.com/s/1ZO33bzb6GoFw4e-b8hAEMQ?pwd=a2bj)

## 1. 代码文件说明

1. `split_dataset.py` 用于将原始图像分类数据集按照指定比例划分为训练集、验证集和测试集，训练集、验证集、测试集的比例为 8:1:1。

2. `train.py` 用于训练一个适用于 **501 类图像分类任务** 的轻量级 ResNet18 模型，支持训练、验证和测试流程，基于 PyTorch 实现。

3. `train_continue.py` 用于在前面训练的模型基础上进行继续训练， **从第 100 轮继续训练到第 160 轮**，并在验证集上继续提升性能。

4. `evaluate.py ` 用于对测试集图像进行推理评估，输出模型在 **Top-1 和 Top-5** 下的准确率。包含 **load_model** 函数，可以用来进行未公开测试集的准确率验证，只需要修改数据集的文件路径即可。
5. `inference.py` 用于对单张图像文件进行分类预测，可输出**预测类别**，**预测类别对应的具体汉字**（从 `Char_dict.json` 映射），**预测类别的置信度**。
6. `plot_log.py` 用于读取**训练过程**中的 JSON 格式日志文件，绘制：训练损失（loss）曲线，验证集准确率（Top-1 accuracy）曲线。
7. `predict_and_save.py` 用于加载训练好的图像分类模型，对**测试集**图像进行批量预测，并将结果保存为 `prediction_results.pkl`，供后续评估与可视化使用。

8. `evaluate_and_plot.py` 用于模型在**测试集**上进行系统性分析与可视化，评估维度包括：总体准确率，混淆矩阵（Confusion Matrix），Top-K 准确率，Precision / Recall / F1-score 对比柱状图，误分类率最高的类别分析。

## 2. 模型文件说明

1. `base.pth` 训练到 100 epoch 保存的模型文件，在测试集的准确率是 88.0%，是在 GeForce RTX 4090 上进行的训练，batchsize 设置为 512。
2. `best.pth` 在`base.pth`基础上继续训练到 160 epoch 保存的模型文件，在测试集的准确率是 95.14%，是在 GeForce RTX 3060 上进行的训练，batchsize 设置为 8。

## 3. 模型性能

验证集准确率达到 95.14%，测试集 Top-1 准确率为 **93.76%**，Top-6 准确率达到 100%。

![1](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/loss_and_val_loss_2.png)

![2](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20250412152343319.png)

![3](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/prf_scores.png)

![4](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20250412152349044.png)

使用 `evaluate.py ` 对测试集验证示意图如下，输出测试集的准确率。

![5](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20250412153025095.png)

使用 `inference.py` 进行推理示意图如下，输出预测类别和对应置信度分数。

![6](https://yeyi0003.oss-cn-hangzhou.aliyuncs.com/image-20250412152250553.png)