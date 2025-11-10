import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRUAttention(nn.Module):
    """
    轻量化BiGRU：减少复杂度，避免过拟合
    新增：返回注意力权重用于可视化
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.4):
        super(BiGRUAttention, self).__init__()
        self.bigru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.residual = nn.Linear(input_dim, hidden_dim * 2)
        self.attn_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # 对序列维度（dim=1）归一化，得到每个时间步的权重
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim] → 输入时间序列
        gru_out, _ = self.bigru(x)  # [batch, seq_len, 2*hidden_dim] → GRU捕捉时序特征
        gru_out = gru_out + self.residual(x)  # 残差连接：缓解梯度消失
        attn_weights = self.attn_layer(gru_out)  # [batch, seq_len, 1] → 每个时间步的注意力权重
        weighted_feat = torch.sum(attn_weights * gru_out, dim=1)  # 加权求和：聚焦关键时间步
        
        # 新增：返回加权特征 + 注意力权重（用于可视化）
        return weighted_feat, attn_weights  
