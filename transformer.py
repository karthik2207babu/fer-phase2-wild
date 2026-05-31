import torch
import torch.nn as nn

class FRITTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,      
        num_local_layers=2,  
        num_fusion_layers=2, 
        num_classes=7,
        dropout=0.5       
    ):
        super(FRITTransformer, self).__init__()

        self.num_patches = 9
        
        # 1. LOCAL RELATION TRANSFORMER
        self.local_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        local_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.local_transformer = nn.TransformerEncoder(local_layer, num_layers=num_local_layers)

        # 2. GLOBAL FUSION TRANSFORMER
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.fusion_pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=num_fusion_layers)

        self.pos_drop = nn.Dropout(p=dropout)
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        # 3. JOINT OPTIMIZATION HEADS
        self.aux_global_head = nn.Linear(embed_dim, num_classes)
        self.aux_local_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- A. GLOBAL FEATURE EXTRACTION ---
        global_feat = x.mean(dim=[2, 3]) 
        aux_global_logits = self.aux_global_head(global_feat)

        # --- B. PATCH EXTRACTION (28x28 map -> 9 patches) ---
        patch_size = 12
        stride = 8
        regions = []
        for i in range(3):
            for j in range(3):
                h_start, w_start = i * stride, j * stride
                patch = x[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size]
                regions.append(patch.mean(dim=[2, 3]))
        
        regional_tokens = torch.stack(regions, dim=1) 

        # --- C. LOCAL RELATION TRANSFORMER ---
        T_local = self.pos_drop(regional_tokens + self.local_pos_embed)
        T_local_out = self.local_transformer(T_local)
        
        local_feat = T_local_out.mean(dim=1)
        aux_local_logits = self.aux_local_head(local_feat)

        # --- D. GLOBAL FUSION TRANSFORMER ---
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        T_fusion = torch.cat([cls_tokens, T_local_out], dim=1) 
        T_fusion = self.pos_drop(T_fusion + self.fusion_pos_embed)
        
        out = self.fusion_transformer(T_fusion)
        
        cls_out = out[:, 0, :]
        logits = self.head(cls_out)

        return logits, cls_out, aux_global_logits, aux_local_logits