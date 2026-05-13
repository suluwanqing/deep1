import os
import argparse
import pickle
from typing import Optional

import torch
import torch.nn.functional as F

from model import TextCNN, BiLSTMClassifier
from data_utils import Vocabulary


class SentimentPredictor:
    """
    情感分析推理引擎
    封装模型加载、文本编码和预测的完整流程，
    用户只需调用 predict() 方法即可获取文本的情感分析结果。
    """

    # 标签到中文的映射字典
    LABEL_CN = {0: '负面', 1: '正面'}

    def __init__(self, checkpoint: str, vocab_path: Optional[str] = None, device: str = 'auto'):
        """
        初始化预测器：加载模型权重和词汇表

        参数:
            checkpoint: 模型权重文件路径（.pth文件）
            vocab_path: 词汇表文件路径（.pkl文件），若为None则自动查找
            device: 计算设备，'auto'自动选择，也可指定'cuda'或'cpu'
        """
        # 自动选择计算设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 检查模型权重文件是否存在
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint}")

        # 加载模型权重文件
        print(f"[加载] 模型权重: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=self.device)
        # 从权重文件中提取训练时的参数配置
        args = ckpt.get('args', {})

        # 加载词汇表（用于将文本转换为模型可接受的数字ID序列）
        self.vocab = self._load_vocab(vocab_path, ckpt)
        # 获取训练时的最大序列长度
        self.max_len = args.get('max_len', 128)

        # 根据参数中的模型类型构建模型结构
        model_type = args.get('model', 'textcnn')
        self.model = self._build_model(model_type, args, len(self.vocab))
        # 加载模型权重
        self.model.load_state_dict(ckpt['model_state'])
        # 将模型移动到计算设备
        self.model.to(self.device)
        # 设置为评估模式（关闭Dropout，BatchNorm使用运行时统计量）
        self.model.eval()

        print(f"[就绪] 词汇表大小: {len(self.vocab)} | 最大长度: {self.max_len}")

    def _load_vocab(self, vocab_path, ckpt) -> Vocabulary:
        """
        加载词汇表，按优先级依次尝试多种来源

        参数:
            vocab_path: 用户指定的词汇表路径
            ckpt: 模型权重字典，可能包含内嵌的词汇表

        返回:
            加载的Vocabulary对象
        """
        # 优先使用用户指定的路径
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                return pickle.load(f)
        # 其次检查权重文件中是否保存了词汇表
        if 'vocab' in ckpt:
            return ckpt['vocab']
        # 最后尝试默认路径
        for candidate in ['checkpoints/vocab.pkl', 'vocab.pkl']:
            if os.path.exists(candidate):
                with open(candidate, 'rb') as f:
                    return pickle.load(f)
        raise FileNotFoundError("找不到词汇表 vocab.pkl")

    def _build_model(self, model_type: str, args: dict, vocab_size: int):
        """
        根据模型类型和参数构建模型实例
        注意：推理时dropout设为0.0，因为模型已在eval模式下运行

        参数:
            model_type: 模型类型，'textcnn'或'bilstm'
            args: 训练时保存的参数字典
            vocab_size: 词汇表大小

        返回:
            构建好的模型实例
        """
        if model_type == 'textcnn':
            return TextCNN(
                vocab_size=vocab_size,
                embed_dim=args.get('embed_dim', 256),
                num_classes=2,
                num_filters=args.get('num_filters', 256),
                kernel_sizes=[2, 3, 4, 5],
                dropout=0.0  # 推理时不需要Dropout
            )
        else:
            return BiLSTMClassifier(
                vocab_size=vocab_size,
                embed_dim=args.get('embed_dim', 256),
                hidden_dim=args.get('hidden_dim', 256),
                num_classes=2,
                num_layers=2,
                dropout=0.0  # 推理时不需要Dropout
            )

    def _encode(self, text: str) -> torch.Tensor:
        """
        将原始文本编码为模型输入张量

        参数:
            text: 待预测的中文文本

        返回:
            形状为 (1, max_len) 的LongTensor，包含词ID序列
        """
        ids = self.vocab.encode(text, self.max_len)
        return torch.tensor([ids], dtype=torch.long)

    def predict(self, text: str) -> tuple:
        """
        预测单条文本的情感倾向

        参数:
            text: 待预测的中文文本

        返回:
            (标签, 置信度, 负面概率, 正面概率)
            - 标签: 0=负面, 1=正面
            - 置信度: 预测类别的概率值
            - 负面概率: 属于负面类别的概率
            - 正面概率: 属于正面类别的概率
        """
        tensor = self._encode(text).to(self.device)
        with torch.no_grad():  # 禁用梯度计算，节省显存
            logits = self.model(tensor)
            # softmax将logits转换为概率分布
            probs = F.softmax(logits, dim=1)[0]
        # 取概率最大的类别作为预测标签
        label = int(probs.argmax().item())
        # 该类别的概率即为置信度
        confidence = float(probs[label].item())
        return label, confidence, float(probs[0].item()), float(probs[1].item())


def format_bar(prob: float, width: int = 30) -> str:
    """
    生成可视化概率条
    用实心方块和空心方块直观展示概率大小

    参数:
        prob: 概率值，范围 [0, 1]
        width: 概率条的总宽度（字符数）

    返回:
        由 █ 和 ░ 组成的概率条字符串
    """
    filled = int(prob * width)
    return '█' * filled + '░' * (width - filled)


def main():
    """主函数：解析参数 → 初始化预测器 → 进入交互式对话模式"""

    parser = argparse.ArgumentParser(description='中文情感分析 - 正负面')
    parser.add_argument('--model', type=str, default='textcnn', help='模型类型: textcnn 或 bilstm')
    parser.add_argument('--checkpoint', type=str, default=None, help='模型权重路径')
    parser.add_argument('--vocab', type=str, default='checkpoints/vocab.pkl', help='词汇表路径')
    parser.add_argument('--device', type=str, default='auto', help='设备: auto/cuda/cpu')
    args = parser.parse_args()

    # 设置默认checkpoint路径（根据模型类型自动推断）
    if args.checkpoint is None:
        args.checkpoint = os.path.join('checkpoints', f'{args.model}_best.pth')

    # 检查模型权重文件是否存在
    if not os.path.exists(args.checkpoint):
        print(f"[错误] 模型不存在: {args.checkpoint}")
        print(f"[提示] 请先训练: python main.py --model {args.model}")
        return

    # 初始化预测器
    try:
        predictor = SentimentPredictor(args.checkpoint, args.vocab, args.device)
    except Exception as e:
        print(f"[错误] 初始化失败: {e}")
        return

    # ==================== 交互式对话模式 ====================
    print("\n" + "=" * 50)
    print("中文情感分析 - 正负面")
    print("=" * 50)
    print("输入中文文本进行情感分析，输入 q 退出\n")

    idx = 1  # 输入计数器
    while True:
        try:
            text = input(f"[{idx}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            # 处理Ctrl+C或Ctrl+D等中断信号
            print("\n再见")
            break

        if not text:
            # 忽略空输入
            continue

        if text.lower() in ('q', 'quit', 'exit'):
            # 退出命令
            print("再见")
            break

        # 执行情感预测
        label, confidence, prob_neg, prob_pos = predictor.predict(text)
        # 根据标签确定情感词
        sentiment = "正面" if label == 1 else "负面"
        # 使用ANSI转义码为输出着色：正面绿色，负面红色
        color = "\033[92m" if label == 1 else "\033[91m"
        reset = "\033[0m"

        # 格式化输出预测结果
        print(f"\n  判定: {color}{sentiment}{reset} (置信度: {confidence*100:.1f}%)")
        print(f"  负面: {format_bar(prob_neg)} {prob_neg*100:.1f}%")
        print(f"  正面: {format_bar(prob_pos)} {prob_pos*100:.1f}%\n")
        idx += 1


if __name__ == '__main__':
    main()
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!注意如果需要切换模型需要手动处理
# 由于数据集合偏向生活日志而不是生活用于,建议与餐饮评价和酒店评价相关,更多可以先参见查看csv数据。
