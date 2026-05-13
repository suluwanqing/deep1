from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在无GUI环境下报错
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, confusion_matrix, classification_report
)
from tqdm import tqdm


class FocalLoss(nn.Module):
    """
    Focal Loss 焦点损失函数
    公式: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    核心思想：通过降低易分类样本的权重，使模型聚焦于难分类样本
    - gamma参数控制调制强度，gamma越大，易分类样本的权重衰减越严重
    - alpha参数可选，用于平衡正负样本的类别权重
    适用于类别不均衡或难易样本差异大的场景
    """
    def __init__(self, alpha: float = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        参数:
            alpha: 类别权重，可以是浮点数（统一权重）或列表（每类权重）
            gamma: 调制因子，控制易分类样本的降权程度，默认2.0
            reduction: 损失聚合方式，'mean'取均值，'sum'求和，'none'不聚合
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算Focal Loss
        参数:
            logits: 模型输出的未归一化概率，形状 (batch, num_classes)
            targets: 真实标签，形状 (batch,)
        返回:
            计算得到的损失值
        """
        # 先计算标准交叉熵损失（不聚合）
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        # 计算模型对正确类别的预测概率 p_t
        pt = torch.exp(-ce_loss)
        # 计算焦点调制权重 (1 - p_t)^gamma，易分类样本p_t接近1，权重接近0
        focal_weight = (1 - pt) ** self.gamma

        # 如果提供了alpha权重，乘以类别权重
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                # alpha为列表时，根据每个样本的真实标签索引对应权重
                alpha_t = torch.tensor([self.alpha[t] for t in targets], device=logits.device)
            else:
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        # 最终损失 = 调制权重 × 交叉熵损失
        loss = focal_weight * ce_loss
        # 根据reduction参数聚合损失
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    将硬标签（如 [1, 0]）软化为软标签（如 [0.9, 0.1]），
    防止模型过度自信，从而提升泛化能力。
    平滑方式：正确类别的概率为 1 - smoothing，其余类别均分 smoothing 概率
    """
    def __init__(self, num_classes: int = 2, smoothing: float = 0.1):
        """
        参数:
            num_classes: 分类类别数
            smoothing: 平滑系数ε，正确类别概率为 1-ε，其余每个类别概率为 ε/(C-1)
        """
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        # 真实类别的目标概率
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算标签平滑交叉熵
        参数:
            logits: 模型输出的logits，形状 (batch, num_classes)
            targets: 真实标签，形状 (batch,)
        返回:
            标签平滑交叉熵损失值
        """
        # 计算log_softmax，数值更稳定的softmax对数
        log_probs = F.log_softmax(logits, dim=-1)
        # 初始化平滑目标分布：所有位置填充 ε/(C-1)
        smooth_targets = torch.zeros_like(log_probs).fill_(self.smoothing / (self.num_classes - 1))
        # 在真实类别位置填入 1-ε 的概率
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        # 计算交叉熵：-Σ smooth_target * log_prob
        return (-smooth_targets * log_probs).sum(dim=-1).mean()


class EMA:
    """
    指数移动平均（Exponential Moving Average）
    维护模型参数的滑动平均副本（影子权重），推理时使用影子权重通常能获得更好的泛化性能。
    工作流程：
      - 训练时每步更新影子权重：shadow = decay * shadow + (1 - decay) * current
      - 评估时临时将模型权重替换为影子权重
      - 评估后恢复原始权重继续训练
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        参数:
            model: 需要进行EMA的模型
            decay: EMA衰减率，越接近1表示历史权重占比越大，更新越平滑
        """
        self.decay = decay
        # 初始化影子权重为模型当前参数的深拷贝
        self.shadow = {name: param.clone().detach()
                       for name, param in model.named_parameters()
                       if param.requires_grad}
        # 备份字典，用于评估时临时替换权重后恢复
        self.backup = {}

    def update(self, model: nn.Module) -> None:
        """更新影子权重: new_shadow = decay * old_shadow + (1 - decay) * current_param"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> dict:
        """
        临时将模型参数替换为影子权重，并返回原始权重的备份
        用于评估阶段：评估时使用平滑后的影子权重可获得更稳定的结果
        """
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # 备份当前训练权重
                    self.backup[name] = param.data.clone()
                    # 替换为影子权重
                    param.data.copy_(self.shadow[name])
        return self.backup

    def restore(self, model: nn.Module) -> None:
        """恢复原始训练权重，评估结束后必须调用此方法以继续正常训练"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.backup:
                    param.data.copy_(self.backup[name])
        self.backup = {}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 5.0,
    ema: EMA = None,
    accumulation_steps: int = 1,
    scheduler = None
) -> Tuple[float, float]:
    """
    执行一个epoch的训练
    支持功能：梯度累积、梯度裁剪、EMA更新、每步学习率调度

    参数:
        model: 待训练的模型
        loader: 训练数据的DataLoader
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        clip_grad: 梯度裁剪的最大范数，防止梯度爆炸
        ema: EMA实例，若提供则每步更新影子权重
        accumulation_steps: 梯度累积步数，等效增大batch_size
        scheduler: 可选的每步学习率调度器

    返回:
        (平均损失, 训练集准确率)
    """
    model.train()  # 设置为训练模式，启用Dropout和BatchNorm的训练行为
    total_loss = 0.0
    all_preds, all_labels = [], []

    optimizer.zero_grad()  # 在epoch开始时清零梯度
    pbar = tqdm(loader, desc="[训练中]", leave=False, ncols=100)  # 进度条

    for step, (texts, labels) in enumerate(pbar):
        # 将数据移动到计算设备
        texts, labels = texts.to(device), labels.to(device)
        # 前向传播
        logits = model(texts)
        # 计算损失
        loss = criterion(logits, labels)

        # 梯度累积：损失除以累积步数，使得累积后的梯度等价于大batch
        if accumulation_steps > 1:
            loss = loss / accumulation_steps

        # 反向传播，累积梯度（不立即清零）
        loss.backward()

        # 达到累积步数或到达最后一个batch时，执行参数更新
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            # 梯度裁剪：限制梯度范数，防止梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            # 优化器更新参数
            optimizer.step()
            # 清零梯度，为下一轮累积做准备
            optimizer.zero_grad()
            # 如果提供了每步调度器，则每步更新学习率
            if scheduler is not None:
                scheduler.step()

        # 记录损失（乘回累积步数以得到真实损失值）
        total_loss += loss.item() * accumulation_steps
        # 收集预测结果用于计算准确率
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # 每步更新EMA影子权重
        if ema is not None:
            ema.update(model)

        # 更新进度条显示的当前损失
        pbar.set_postfix(Loss=f"{loss.item() * accumulation_steps:.4f}")

    # 返回平均损失和训练准确率
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float, List[int], List[int]]:
    """
    在给定数据集上评估模型性能

    参数:
        model: 待评估的模型
        loader: 评估数据的DataLoader
        criterion: 损失函数
        device: 计算设备

    返回:
        (平均损失, 准确率, F1分数, 精确率, 召回率, 所有预测标签, 所有真实标签)
    """
    model.eval()  # 设置为评估模式，关闭Dropout，BatchNorm使用运行时统计量
    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="[评估中]", leave=False, ncols=100)
    with torch.no_grad():  # 禁用梯度计算，节省显存和加速推理
        for texts, labels in pbar:
            texts, labels = texts.to(device), labels.to(device)
            # 前向传播
            logits = model(texts)
            # 计算损失
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # 收集预测结果
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(Loss=f"{loss.item():.4f}")

    # 计算各项评估指标
    avg_loss = total_loss / len(loader)
    return (avg_loss,
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='binary'),
            precision_score(all_labels, all_preds, average='binary', zero_division=0),
            recall_score(all_labels, all_preds, average='binary', zero_division=0),
            all_preds, all_labels)


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path: str) -> None:
    """
    绘制训练曲线：左图损失曲线，右图准确率曲线
    用于直观观察模型训练过程中的收敛情况和是否过拟合

    参数:
        train_losses: 每轮训练损失列表
        val_losses: 每轮验证损失列表
        train_accs: 每轮训练准确率列表
        val_accs: 每轮验证准确率列表
        save_path: 图片保存路径
    """
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # 左图：训练/验证损失随epoch变化
    axes[0].plot(epochs, train_losses, 'b-o', markersize=4, label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-o', markersize=4, label='Val Loss')
    axes[0].set_xlabel('Epoch'), axes[0].set_ylabel('Loss'), axes[0].set_title('Loss Curve')
    axes[0].legend(), axes[0].grid(True, alpha=0.3)

    # 右图：训练/验证准确率随epoch变化（转换为百分比显示）
    axes[1].plot(epochs, [a * 100 for a in train_accs], 'b-o', markersize=4, label='Train Acc')
    axes[1].plot(epochs, [a * 100 for a in val_accs], 'r-o', markersize=4, label='Val Acc')
    axes[1].set_xlabel('Epoch'), axes[1].set_ylabel('Accuracy (%)'), axes[1].set_title('Accuracy Curve')
    axes[1].legend(), axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion(labels: List[int], preds: List[int], save_path: str,
                   class_names: List[str] = ['Negative', 'Positive']) -> None:
    """
    绘制混淆矩阵：左图为原始计数，右图为归一化比例
    混淆矩阵可以直观展示模型在各类别上的分类情况，包括正确分类和误分类的分布

    参数:
        labels: 真实标签列表
        preds: 预测标签列表
        save_path: 图片保存路径
        class_names: 类别名称列表，默认为 ['Negative', 'Positive']
    """
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    # 按行归一化，得到每个真实类别被预测为各类别的比例
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 同时绘制计数版和归一化版混淆矩阵
    for ax, data, title, fmt in zip(axes, [cm, cm_norm],
                                     ['Confusion Matrix (Count)', 'Confusion Matrix (Normalized)'],
                                     ['d', '.2f']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, linewidths=0.5)
        ax.set_title(title), ax.set_ylabel('True Label'), ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_classification_report(labels: List[int], preds: List[int]) -> None:
    """
    打印详细的分类报告，包含每个类别的精确率、召回率、F1分数和样本数
    输出内容为中文标注版本，便于阅读

    参数:
        labels: 真实标签列表
        preds: 预测标签列表
    """
    # 使用sklearn生成分类报告
    report = classification_report(labels, preds, target_names=['负面 (Negative)', '正面 (Positive)'])
    # 将英文指标名替换为中文，提升可读性
    report = report.replace('precision', '精确率 (Precision)')
    report = report.replace('recall', '召回率 (Recall)')
    report = report.replace('f1-score', 'F1分数 (F1-Score)')
    report = report.replace('support', '样本数 (Support)')
    report = report.replace('accuracy', '准确率 (Accuracy)')
    report = report.replace('macro avg', '宏平均 (Macro Avg)')
    report = report.replace('weighted avg', '加权平均 (Weighted Avg)')
    print("\n" + "=" * 50)
    print("详细分类报告")
    print("=" * 50)
    print(report)
