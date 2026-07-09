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

        # Restored to 9 patches (12x12 window, stride 8)
        self.num_patches = 9
        self.pos_drop = nn.Dropout(p=dropout)
        
        # ==========================================
        # 1. LOCAL RELATION TRANSFORMER (Self-Attention)
        # ==========================================
        self.local_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        local_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.local_transformer = nn.TransformerEncoder(local_layer, num_layers=num_local_layers)

        # ==========================================
        # 2. GLOBAL-LOCAL FUSION (Cross-Attention)
        # ==========================================
        # Native PyTorch Cross-Attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed Forward Network for the Cross-Attention block
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

        # ==========================================
        # 3. JOINT OPTIMIZATION HEADS
        # ==========================================
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        self.aux_global_head = nn.Linear(embed_dim, num_classes)
        self.aux_local_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- A. TRUE GLOBAL TOKEN EXTRACTION (MRAN: x^G) ---
        global_feat = x.mean(dim=[2, 3]) 
        global_token = global_feat.unsqueeze(1) # Shape: (B, 1, 128) - Token #1
        aux_global_logits = self.aux_global_head(global_feat)

        # --- B. OVERLAPPING PATCH EXTRACTION (9 Tokens) ---
        # Window: 12x12, Stride: 8
        patch_size = 12
        stride = 8
        regions = []
        for i in range(3):
            for j in range(3):
                h_start, w_start = i * stride, j * stride
                patch = x[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size]
                regions.append(patch.mean(dim=[2, 3]))
        
        regional_tokens = torch.stack(regions, dim=1) # Shape: (B, 9, 128) - Tokens #2-10

        # --- C. REGION RELATION TRANSFORMER (Local-Local Self Attention) ---
        T_local = self.pos_drop(regional_tokens + self.local_pos_embed)
        T_local_out = self.local_transformer(T_local) # Shape: (B, 9, 128)
        
        local_feat = T_local_out.mean(dim=1)
        aux_local_logits = self.aux_local_head(local_feat)

        # --- D. GLOBAL-LOCAL RELATION TRANSFORMER (MRAN Cross-Attention) ---
        # Combine 1 Global Token + 9 Local Tokens = 10 Tokens total
        kv_tokens = torch.cat([global_token, T_local_out], dim=1) # Shape: (B, 10, 128)
        
        # Query (Q) = Global Token
        # Keys (K) & Values (V) = All 10 Tokens
        attn_out, _ = self.cross_attn(query=global_token, key=kv_tokens, value=kv_tokens)
        
        # Add & Norm
        global_token = self.norm1(global_token + attn_out)
        
        # Feed-Forward & Norm
        ffn_out = self.ffn(global_token)
        global_token = self.norm2(global_token + ffn_out) # Shape: (B, 1, 128)
        
        # --- E. FINAL PREDICTION ---
        cls_out = global_token.squeeze(1) 
        logits = self.head(cls_out)

        return logits, cls_out, aux_global_logits, aux_local_logits