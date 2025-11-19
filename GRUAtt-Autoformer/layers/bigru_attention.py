import torch
import torch.nn as nn
import torch.nn.functional as F

class BiGRUAttention(nn.Module):

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
            nn.Softmax(dim=1)  
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim] 
        gru_out, _ = self.bigru(x)  # [batch, seq_len, 2*hidden_dim] 
        gru_out = gru_out + self.residual(x)  
        attn_weights = self.attn_layer(gru_out)  # [batch, seq_len, 1] 
        weighted_feat = torch.sum(attn_weights * gru_out, dim=1)  
        
        return weighted_feat, attn_weights  

