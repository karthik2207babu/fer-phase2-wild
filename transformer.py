import torch
import torch.nn as nn

class FRITTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,      
        num_local_layers=2,  
        num_classes=7,
        dropout=0.5       
    ):
        super(FRITTransformer, self).__init__()

        self.num_patches = 9
        self.pos_drop = nn.Dropout(p=dropout)
        
        self.local_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        local_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.local_transformer = nn.TransformerEncoder(local_layer, num_layers=num_local_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        self.aux_global_head = nn.Linear(embed_dim, num_classes)
        self.aux_local_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        global_feat = x.mean(dim=[2, 3]) 
        aux_global_logits = self.aux_global_head(global_feat)

        patch_size = 12
        stride = 8
        regions = []
        for i in range(3):
            for j in range(3):
                h_start, w_start = i * stride, j * stride
                patch = x[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size]
                regions.append(patch.mean(dim=[2, 3]))
        
        regional_tokens = torch.stack(regions, dim=1) 

        T_local = self.pos_drop(regional_tokens + self.local_pos_embed)
        T_local_out = self.local_transformer(T_local) 
        
        local_feat = T_local_out.mean(dim=1)
        aux_local_logits = self.aux_local_head(local_feat)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=cls_tokens, key=T_local_out, value=T_local_out)
        
        cls_tokens = self.norm1(cls_tokens + attn_out)
        ffn_out = self.ffn(cls_tokens)
        cls_tokens = self.norm2(cls_tokens + ffn_out)
        
        cls_out = cls_tokens.squeeze(1) 
        logits = self.head(cls_out)

        return logits, cls_out, aux_global_logits, aux_local_logits