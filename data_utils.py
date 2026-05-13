import re
import os
import pandas as pd
from collections import Counter
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import jieba


class Vocabulary:
    """
    词汇表类，负责文本到数字ID的双向映射
    包含两个特殊标记：
      - <PAD> (ID=0)：填充标记，用于将不同长度的序列补齐到相同长度
      - <UNK> (ID=1)：未知词标记，处理词汇表外的词
    """
    PAD_TOKEN, UNK_TOKEN = '<PAD>', '<UNK>'

    def __init__(self, max_size: int = 30000, min_freq: int = 2):
        """
        参数:
            max_size: 词汇表最大容量，限制嵌入层大小以控制模型参数量
            min_freq: 最低词频阈值，低于此频率的词将被过滤，减少噪声
        """
        self.max_size, self.min_freq = max_size, min_freq
        # 初始化词到ID的映射，预留PAD和UNK
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        # 初始化ID到词的反向映射
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}

    def _clean(self, text: str) -> str:
        """
        清洗文本：去除特殊符号，只保留中文字符、英文字母和数字
        连续空白字符合并为单个空格

        参数:
            text: 原始文本

        返回:
            清洗后的文本
        """
        # [标记]：清洗文本，去除特殊符号，保留中英文数字
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        # 合并连续空白为一个空格，并去除首尾空白
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词：先清洗再使用jieba进行中文分词

        参数:
            text: 原始中文文本

        返回:
            分词后的词列表
        """
        return jieba.lcut(self._clean(text))

    def build(self, texts: List[str]) -> None:
        """
        从训练文本构建词汇表
        统计所有词的频率，按频率降序排列，过滤低频词后加入词汇表

        参数:
            texts: 训练集文本列表
        """
        # 从训练集语料库统计词频，过滤低频词
        counter = Counter()
        for text in texts: counter.update(self.tokenize(text))
        # 按频率降序取前max_size个词，跳过低于min_freq的词
        for word, freq in counter.most_common(self.max_size):
            if freq < self.min_freq: break
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        """
        将文本转换为固定长度的ID序列
        处理流程：分词 → 查表转换为ID → 截断/填充到max_len

        参数:
            text: 待编码的文本
            max_len: 目标序列长度，超出截断，不足用PAD(0)填充

        返回:
            长度为max_len的整数列表
        """
        # 将文本转换为ID序列，进行截断和填充
        tokens = self.tokenize(text)[:max_len]  # 截断超长文本
        ids = [self.word2idx.get(t, 1) for t in tokens]  # 未知词映射为UNK(ID=1)
        # 右侧填充PAD(ID=0)到固定长度
        return ids + [0] * (max_len - len(ids))

    def __len__(self) -> int:
        """返回词汇表大小（包含PAD和UNK）"""
        return len(self.word2idx)


class SentimentDataset(Dataset):
    """
    PyTorch数据集封装类
    将文本数据转换为模型可消费的(张量, 标签)对
    """
    def __init__(self, texts, labels, vocab, max_len):
        """
        参数:
            texts: 文本列表
            labels: 对应标签列表
            vocab: Vocabulary词汇表对象
            max_len: 序列最大长度
        """
        self.texts, self.labels, self.vocab, self.max_len = texts, labels, vocab, max_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单条数据样本
        将文本编码为ID序列张量，标签转为长整型张量

        参数:
            idx: 数据索引

        返回:
            (文本ID张量, 标签张量)
        """
        ids = self.vocab.encode(self.texts[idx], self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def load_csv_data(batch_size: int = 64, max_len: int = 128, data_dir: str = "csv_data", dataset_name: str = "merged"):
    """
    从CSV文件加载情感分析数据集的完整流程
    流程：读取CSV → 提取文本和标签 → 构建词汇表 → 封装为DataLoader

    参数:
        batch_size: 每个batch的样本数
        max_len: 文本序列的最大长度
        data_dir: CSV文件所在目录
        dataset_name: 数据集名称前缀（如"merged"对应merged_train.csv等）

    返回:
        (train_loader, val_loader, test_loader, vocab)
        - train_loader: 训练集DataLoader（打乱顺序）
        - val_loader: 验证集DataLoader
        - test_loader: 测试集DataLoader
        - vocab: 构建好的词汇表对象
    """
    # 构造CSV文件路径
    train_path = os.path.join(data_dir, f"{dataset_name}_train.csv")
    val_path = os.path.join(data_dir, f"{dataset_name}_val.csv")
    test_path = os.path.join(data_dir, f"{dataset_name}_test.csv")

    # 检查数据文件完整性
    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p): raise FileNotFoundError(f"文件未找到: {p}")

    # 读取三个CSV文件为DataFrame
    df_tr, df_va, df_te = pd.read_csv(train_path), pd.read_csv(val_path), pd.read_csv(test_path)

    # 从DataFrame提取文本和标签
    # CSV格式：第一列为标签(label)，第二列为评论内容(review)
    def process_df(df):
        return df.iloc[:, 1].astype(str).tolist(), df.iloc[:, 0].astype(int).tolist()

    train_texts, train_labels = process_df(df_tr)
    val_texts, val_labels = process_df(df_va)
    test_texts, test_labels = process_df(df_te)

    # 构建词表，仅使用训练集构建，避免数据泄漏
    vocab = Vocabulary()
    vocab.build(train_texts)

    # 创建DataLoader：训练集打乱顺序，验证集和测试集不打乱
    train_loader = DataLoader(SentimentDataset(train_texts, train_labels, vocab, max_len), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(SentimentDataset(val_texts, val_labels, vocab, max_len), batch_size=batch_size,
                            shuffle=False)
    test_loader = DataLoader(SentimentDataset(test_texts, test_labels, vocab, max_len), batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader, vocab
