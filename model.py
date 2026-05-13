import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    TextCNN文本分类模型
    核心思想：使用多种尺寸的卷积核提取不同长度的n-gram特征，
    通过最大池化选取每个卷积核响应最强烈的特征，拼接后进行分类。
    结构：Embedding → Dropout → 多尺度Conv1d → BatchNorm → ReLU → 自适应最大池化 → 拼接 → Dropout → 全连接
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int,
                 num_filters: int, kernel_sizes: list, dropout: float = 0.4):
        """
        参数:
            vocab_size: 词汇表大小，决定嵌入层矩阵的行数
            embed_dim: 词向量维度，每个词被映射到的向量长度
            num_classes: 分类类别数（本任务为2：正面/负面）
            num_filters: 每种尺寸卷积核的数量
            kernel_sizes: 卷积核尺寸列表，例如[2,3,4,5]分别捕获bigram/trigram/4-gram/5-gram特征
            dropout: 全连接层前的Dropout比率，防止过拟合
        """
        super(TextCNN, self).__init__()

        # 词嵌入层：将词ID映射为稠密向量，padding_idx=0表示PAD词的嵌入始终为零向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 嵌入层后的Dropout，以0.2的概率随机置零，增加鲁棒性
        self.embed_drop = nn.Dropout(0.2)

        # 多尺度卷积层：每种尺寸的卷积核独立提取特征，padding=k//2保持序列长度不变
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2) for k in kernel_sizes
        ])
        # 批归一化层：每个卷积核对应一个BN层，加速收敛并提供正则化效果
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])

        # 全连接层前的Dropout
        self.fc_drop = nn.Dropout(dropout)
        # 全连接分类层：输入维度为各卷积核数量之和，输出维度为类别数
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

        # 初始化模型权重
        self._init_weights()

    def _init_weights(self):
        """自定义权重初始化：卷积层使用Kaiming初始化适配ReLU，全连接层使用Xavier初始化"""
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
            nn.init.zeros_(conv.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: 输入文本的词ID序列，形状为 (batch_size, seq_len)
        返回:
            分类logits，形状为 (batch_size, num_classes)
        """
        # 词嵌入 + Dropout → (batch, seq_len, embed_dim)
        embedded = self.embed_drop(self.embedding(x))
        # 转置为 (batch, embed_dim, seq_len) 适配PyTorch Conv1d的输入格式
        embedded = embedded.permute(0, 2, 1)

        # 对每种尺寸的卷积核分别提取特征
        pooled = []
        for conv, bn in zip(self.convs, self.bns):
            # 卷积 → 批归一化 → ReLU激活 → 自适应最大池化
            # 自适应最大池化将变长序列压缩为单个值，取每个通道中最强的情感特征
            c = F.relu(bn(conv(embedded)))
            c = F.adaptive_max_pool1d(c, output_size=1).squeeze(2)  # (batch, num_filters)
            pooled.append(c)

        # 拼接所有卷积核的输出，形成完整的特征表示
        cat = torch.cat(pooled, dim=1)  # (batch, num_filters * len(kernel_sizes))
        return self.fc(self.fc_drop(cat))


class BiLSTMClassifier(nn.Module):
    """
    双向LSTM文本分类模型
    核心思想：利用双向LSTM捕获文本的前向和后向上下文信息，
    通过全局最大池化提取序列中最显著的特征进行分类。
    结构：Embedding → Dropout → BiLSTM → LayerNorm → 全局最大池化 → Dropout → 全连接
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_classes: int, num_layers: int = 2, dropout: float = 0.4):
        """
        参数:
            vocab_size: 词汇表大小
            embed_dim: 词向量维度
            hidden_dim: LSTM隐藏层维度（单向），双向LSTM实际输出维度为 hidden_dim * 2
            num_classes: 分类类别数
            num_layers: LSTM堆叠层数，多层堆叠增强模型表达能力
            dropout: Dropout比率（LSTM层间和全连接层前使用）
        """
        super(BiLSTMClassifier, self).__init__()

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 嵌入层后的Dropout
        self.embed_drop = nn.Dropout(0.2)

        # 双向LSTM层
        # batch_first=True使输入输出形状为(batch, seq, feature)
        # 多层LSTM的层间Dropout仅在num_layers>1时生效
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # 层归一化：对LSTM输出进行归一化，稳定RNN训练过程中的梯度流
        self.ln = nn.LayerNorm(hidden_dim * 2)
        # 全连接层前的Dropout
        self.fc_drop = nn.Dropout(dropout)
        # 全连接分类层：输入维度为双向LSTM输出维度(2 * hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        参数:
            x: 输入文本的词ID序列，形状为 (batch_size, seq_len)
        返回:
            分类logits，形状为 (batch_size, num_classes)
        """
        # 词嵌入 + Dropout → (batch, seq_len, embed_dim)
        embedded = self.embed_drop(self.embedding(x))
        # 双向LSTM编码，out形状为 (batch, seq_len, hidden_dim * 2)
        out, _ = self.lstm(embedded)
        # 层归一化，稳定输出分布
        out = self.ln(out)

        # 全局最大池化：沿时间维度取每个通道的最大值
        # 过滤padding位置的零向量影响，提取序列中最强的情感特征词
        out = out.permute(0, 2, 1)  # (batch, hidden_dim*2, seq_len)
        hidden = F.adaptive_max_pool1d(out, output_size=1).squeeze(2)  # (batch, hidden_dim*2)

        return self.fc(self.fc_drop(hidden))
