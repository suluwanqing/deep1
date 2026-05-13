import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import TextCNN, BiLSTMClassifier
from data_utils import load_csv_data
from trainer import train_one_epoch, evaluate, plot_curves, plot_confusion, print_classification_report
from trainer import FocalLoss, LabelSmoothingCrossEntropy, EMA


def parse_args() -> argparse.Namespace:
    """解析命令行参数，定义模型结构、训练超参数、路径等所有可配置项"""

    parser = argparse.ArgumentParser(description='中文情感分析训练脚本')

    # ---- 模型与训练基本参数 ----
    parser.add_argument('--model', type=str, default='textcnn', choices=['textcnn', 'bilstm'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=1.0)

    # ---- 模型结构参数 ----
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_filters', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--max_len', type=int, default=128)

    # ---- 早停 ----
    parser.add_argument('--patience', type=int, default=5)

    # ---- 损失函数选择与参数 ----
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'ce', 'label_smooth'],
                        help='损失函数: focal(FocalLoss) / ce(交叉熵+标签平滑) / label_smooth(自定义标签平滑)')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='FocalLoss的gamma参数')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='标签平滑系数')

    # ---- EMA指数移动平均参数 ----
    parser.add_argument('--use_ema', action='store_true', default=True, help='启用EMA指数移动平均')
    parser.add_argument('--no_ema', action='store_false', dest='use_ema', help='禁用EMA')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA衰减率')

    # ---- 梯度累积参数 ----
    parser.add_argument('--accumulation_steps', type=int, default=2, help='梯度累积步数(等效batch=batch_size*accumulation_steps)')

    # ---- 文件路径参数 ----
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--fig_dir', type=str, default='figures')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--data_dir', type=str, default='csv_data')
    parser.add_argument('--dataset', type=str, default='merged')

    return parser.parse_args()


def build_model(args: argparse.Namespace, vocab_size: int) -> nn.Module:
    """
    根据参数构建对应的模型实例
    参数:
        args: 命令行参数，包含模型类型及结构超参数
        vocab_size: 词汇表大小，用于嵌入层的输入维度
    返回:
        构建好的PyTorch模型（TextCNN或BiLSTMClassifier）
    """
    if args.model == 'textcnn':
        # TextCNN: 多尺度卷积核 [2,3,4,5] 捕获不同长度的n-gram特征
        return TextCNN(
            vocab_size=vocab_size, embed_dim=args.embed_dim, num_classes=2,
            num_filters=args.num_filters, kernel_sizes=[2, 3, 4, 5], dropout=args.dropout
        )
    else:
        # BiLSTM: 双向LSTM捕获上下文语义，2层堆叠增强表达能力
        return BiLSTMClassifier(
            vocab_size=vocab_size, embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
            num_classes=2, num_layers=2, dropout=args.dropout
        )


def main():
    """主训练流程：参数解析 → 数据加载 → 模型构建 → 训练循环 → 评估与可视化"""

    # 解析命令行参数
    args = parse_args()

    # 自动选择计算设备，优先使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[设备] {device}")

    # 创建输出目录（模型保存目录和图片保存目录）
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    # 检查数据目录是否存在
    if not os.path.exists(args.data_dir):
        print(f"[错误] 数据目录不存在: {args.data_dir}，请先运行 python download_dataset.py")
        return

    # 加载数据：返回训练/验证/测试的DataLoader以及词汇表对象
    train_loader, val_loader, test_loader, vocab = load_csv_data(
        batch_size=args.batch_size, max_len=args.max_len,
        data_dir=args.data_dir, dataset_name=args.dataset
    )

    # 将词汇表序列化保存到磁盘，推理时需要加载使用
    with open(os.path.join(args.save_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    # 根据参数构建模型并移动到指定设备
    model = build_model(args, len(vocab)).to(device)
    print(f"[模型] {args.model.upper()} | 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ---- 构建损失函数 ----
    if args.loss == 'focal':
        # FocalLoss: 降低易分类样本权重，聚焦难分类样本，适合类别不均衡场景
        criterion = FocalLoss(gamma=args.focal_gamma)
        print(f"[损失] FocalLoss (gamma={args.focal_gamma})")
    elif args.loss == 'label_smooth':
        # 自定义标签平滑交叉熵：将硬标签软化为软标签，防止模型过自信
        criterion = LabelSmoothingCrossEntropy(num_classes=2, smoothing=args.label_smoothing)
        print(f"[损失] LabelSmoothingCrossEntropy (smoothing={args.label_smoothing})")
    else:
        # PyTorch原生交叉熵+内置标签平滑
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print(f"[损失] CrossEntropyLoss + LabelSmoothing (ε={args.label_smoothing})")

    # AdamW优化器：解耦权重衰减的Adam，正则化效果更好
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- 学习率调度策略 ----
    # warmup阶段：前3个epoch线性提升学习率，避免初始大学习率导致训练不稳定
    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e + 1) / 3 if e < 3 else 1.0)
    # plateau阶段：监控验证集准确率，若连续2轮无提升则将学习率减半
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # ---- EMA 指数移动平均 ----
    # 维护模型参数的滑动平均副本，推理时使用平均权重可提升泛化性能
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema:
        print(f"[EMA] 已启用 (decay={args.ema_decay})")

    # ---- 梯度累积 ----
    # 当GPU显存不足时，可通过梯度累积模拟更大的batch_size
    if args.accumulation_steps > 1:
        print(f"[梯度累积] 步数={args.accumulation_steps}, 等效batch_size={args.batch_size * args.accumulation_steps}")

    # TensorBoard日志记录器，用于可视化训练过程
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, args.model))

    # 训练状态跟踪变量
    best_val_acc, no_improve = 0.0, 0           # 最优验证准确率 & 连续无提升轮数
    best_ckpt_path = os.path.join(args.save_dir, f'{args.model}_best.pth')  # 最优模型保存路径
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  # 训练历史记录

    # ==================== 训练循环 ====================
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch:02d}/{args.epochs}] lr={optimizer.param_groups[0]['lr']:.2e}")

        # 执行一个epoch的训练，支持梯度裁剪、EMA、梯度累积
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            clip_grad=args.clip_grad, ema=ema, accumulation_steps=args.accumulation_steps
        )

        # 评估时临时将模型权重替换为EMA影子权重（如果启用了EMA）
        if ema is not None:
            ema.apply_shadow(model)

        # 在验证集上评估当前模型性能
        val_loss, val_acc, val_f1, _, _, _, _ = evaluate(model, val_loader, criterion, device)

        # 评估完毕后恢复原始权重，继续下一轮训练
        if ema is not None:
            ema.restore(model)

        # ---- 学习率调度器更新 ----
        # 前3个epoch使用warmup调度器线性提升学习率
        if epoch <= 3:
            warmup_scheduler.step()
        else:
            # 之后使用plateau调度器根据验证准确率自适应调整学习率
            plateau_scheduler.step(val_acc)

        # 记录本轮训练和验证的损失与准确率
        train_losses.append(train_loss), val_losses.append(val_loss)
        train_accs.append(train_acc), val_accs.append(val_acc)

        # 将指标写入TensorBoard日志
        writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train': train_acc, 'Val': val_acc}, epoch)

        print(f"  训练 Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"  验证 Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")

        # ---- 早停机制与最优模型保存 ----
        if val_acc > best_val_acc:
            # 验证准确率创新高，重置无提升计数器
            best_val_acc = val_acc
            no_improve = 0
            # 保存模型：如果启用了EMA，则使用EMA影子权重保存，同时保留BatchNorm等非训练参数
            save_state = model.state_dict()
            if ema is not None:
                for name in ema.shadow:
                    save_state[name] = ema.shadow[name]
            torch.save({'epoch': epoch, 'model_state': save_state,
                        'optimizer_state': optimizer.state_dict(), 'val_acc': val_acc,
                        'args': vars(args)}, best_ckpt_path)
            print(f"  [保存] 最优模型，验证准确率: {val_acc:.4f}")
        else:
            # 验证准确率未提升，累计无提升轮数
            no_improve += 1
            if no_improve >= args.patience:
                # 连续patience轮无提升，触发早停
                print(f"[早停] 连续{args.patience}轮无提升，终止训练")
                break

    # ==================== 训练结束后的后处理 ====================

    # 绘制训练曲线（损失和准确率随epoch变化图）
    plot_curves(train_losses, val_losses, train_accs, val_accs,
                os.path.join(args.fig_dir, f'{args.model}_training_curves.png'))

    # 加载最优模型，在测试集上进行最终评估
    print("\n[测试] 加载最优模型进行评估...")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_loss, test_acc, test_f1, test_prec, test_rec, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    # 打印测试集结果和详细分类报告
    print(f"\n测试集结果: 准确率={test_acc:.4f} F1={test_f1:.4f}")
    print_classification_report(test_labels, test_preds)

    # 绘制混淆矩阵（计数版和归一化版）
    plot_confusion(test_labels, test_preds, os.path.join(args.fig_dir, f'{args.model}_confusion_matrix.png'))

    # 关闭TensorBoard写入器
    writer.close()


if __name__ == '__main__':
    main()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意如果需要切换模型需要手动处理
