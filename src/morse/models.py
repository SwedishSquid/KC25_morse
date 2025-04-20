from torch import nn
import torch
import math


class ResBlock(nn.Module):
    def __init__(self, size, p_dropout):
        super().__init__()
        self.cell = nn.Sequential(
            nn.Conv1d(size, size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(size),
            nn.Dropout(p=p_dropout),
            nn.Conv1d(size, size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(size),
            nn.Dropout(p=p_dropout),
        )
        self.activation = nn.ReLU()
        pass

    def forward(self, x):
        return self.activation(x + self.cell(x))
    pass


class MySomething(nn.Module):
    def __init__(self, n_pooled_blocks = 3, n_head_blocks = 2, pooled_blocks_thickness=1, 
                 input_size = 64, inner_size = 64, output_size = 5, p_dropout = 0.1):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Conv1d(input_size, inner_size, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(inner_size),
            nn.Dropout(),
            *[
                 nn.Sequential(
                    *[ResBlock(inner_size, p_dropout) for i_ in range(pooled_blocks_thickness)],
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    ) for _ in range(n_pooled_blocks)
            ],
            *[ResBlock(inner_size, p_dropout) for _ in range(n_head_blocks)],
            nn.Conv1d(inner_size, output_size, kernel_size=3),
        )
        pass

    def forward(self, x):
        return self.estimator(x)





# class CNNResidualBlockCasualNormDropoutOrder(nn.Module):
#     def __init__(self, d_model, d_inner, dropout=0.1, apply_post_residual_nonlinearity=False):
#         super().__init__()
#         self.apply_post_residual_nonlinearity = apply_post_residual_nonlinearity
#         self.cell = nn.Sequential(
#             nn.Conv1d(d_model, d_inner, kernel_size=3, padding=1),
#             nn.BatchNorm1d(d_inner),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#             nn.Conv1d(d_inner, d_model, kernel_size=3, padding=1),
#             nn.BatchNorm1d(d_model),
#             nn.ReLU(),
#             nn.Dropout(p=dropout),
#         )
#         self.post_residual_nonlinearity = nn.Sequential(
#             nn.BatchNorm1d(d_model),
#         )
#         pass

#     def forward(self, x):
#         # [batch, channels, seq_len]
#         out = x + self.cell(x)
#         if self.apply_post_residual_nonlinearity:
#             return self.post_residual_nonlinearity(out)
#         return out



class CNNResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1, apply_post_norm=False):
        super().__init__()
        self.apply_post_norm = apply_post_norm
        
        # Main path (pre-norm version)
        self.cell = nn.Sequential(
            # First conv block
            nn.BatchNorm1d(d_model),  # Pre-norm
            nn.Conv1d(d_model, d_inner, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            
            # Second conv block
            nn.BatchNorm1d(d_inner),  # Pre-norm
            nn.Conv1d(d_inner, d_model, kernel_size=3, padding=1),
            nn.Dropout(p=dropout)
        )
        
        # Optional post-normalization
        self.post_norm = nn.BatchNorm1d(d_model) if apply_post_norm else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.cell(x)
        out = residual + out
        return self.post_norm(out)


class TransformerResidualBlock(nn.Module):
    def __init__(self, d_model, d_ffn, num_heads=4, dropout=0.1, apply_post_norm = False):
        super().__init__()
        self.apply_post_norm = apply_post_norm
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        
        self.post_norm = nn.LayerNorm(d_model)
        pass

    def forward(self, x):
        # x shape: (seq_len, batch, features)
        attn_input = self.norm1(x)
        attn_out, _ = self.attention(attn_input, attn_input, attn_input)
        attn_out = self.attention_dropout(attn_out)
        pre_ffn = x + attn_out
        ffn_input = self.norm2(pre_ffn)
        ffn_out = self.ffn(ffn_input)
        out = ffn_out + pre_ffn

        if self.apply_post_norm:
            return self.post_norm(out)
        return out
    
    def calculate_attention_entropy(self, x):
        '''for debugging'''
        attn_input = self.norm1(x)
        _, attn_weights = self.attention(attn_input, attn_input, attn_input)
        return torch.sum(-attn_weights * torch.log(attn_weights + 1e-10), dim=-1).mean().item()


class PoolingTransition(nn.Module):
    def __init__(self, overlap=False):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1) if overlap else nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        return self.pool(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), None, :]
        return self.dropout(x)



class CTCHead(nn.Module):
    def __init__(self, d_model, d_output):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_output, kernel_size=1)
        )
    
    def forward(self, x):
        return self.estimator(x)



class CNNTransformer(nn.Module):
    def __init__(self, d_input, d_model, 
                 n_pools, n_blocks_before_pool, n_transformer_blocks,
                 head_block: nn.Module,
                 make_cnn_block = lambda: CNNResidualBlock(),
                 make_transformer_block = lambda: TransformerResidualBlock(), 
                 pooling_overlap=False,
                 dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            *[
                nn.Sequential(
                    *[make_cnn_block() for j in range(n_blocks_before_pool)],
                    PoolingTransition(overlap=pooling_overlap)
                )
                for i in range(n_pools)
            ]
        )

        # self.cnn_to_transformer = nn.Sequential(
        #     nn.Conv1d(d_model, d_model, 1),
        #     nn.LayerNorm(d_model),
        #     nn.Dropout(dropout)
        # )

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        self.transformer = nn.Sequential(
            *[make_transformer_block() for k in range(n_transformer_blocks)],
        )

        self.head = head_block
    
    def forward(self, x: torch.Tensor):
        # [batch, channels, seq_len]
        x = self.cnn(x)

        x = x.permute(2, 0, 1) # [seq_len, batch, channels]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)

        # [batch, channels, seq_len]
        out = self.head(x)
        return out



class SimpleCNN(nn.Module):
    def __init__(self, d_input, d_model, d_inner, d_output,
                 n_pools, n_blocks_before_pool,
                 pooling_overlap=False,
                 dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            *[
                nn.Sequential(
                    *[CNNResidualBlock(d_model=d_model, d_inner=d_inner, dropout=dropout) for j in range(n_blocks_before_pool)],
                    PoolingTransition(overlap=pooling_overlap)
                )
                for i in range(n_pools)
            ]
        )
        self.head = CTCHead(d_model, d_output)
    
    def forward(self, x: torch.Tensor):
        # [batch, channels, seq_len]
        x = self.cnn(x)
        # [batch, channels, seq_len]
        out = self.head(x)
        return out



