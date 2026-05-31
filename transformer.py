import torch
import torch.nn as nn

class FRITTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,      
        num_layers=4,     
        num_classes=7,
        dropout=0.4       
    ):
        super(FRITTransformer, self).__init__()

        # 9 local regions + 1 CLS token = 10 tokens
        self.num_tokens = 10
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

        # JOINT OPTIMIZATION Auxiliary Heads
        self.aux_global_head = nn.Linear(embed_dim, num_classes)
        self.aux_local_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Global Auxiliary Prediction
        global_feat = x.mean(dim=[2, 3])
        aux_global_logits = self.aux_global_head(global_feat)

        # 2. 3x3 OVERLAPPING REGIONAL PARTITION (9 Patches)
        # 12x12 patches with a stride of 8 fits a 28x28 map perfectly.
        patch_size = 12
        stride = 8
        regions = []
        
        for i in range(3):
            for j in range(3):
                h_start = i * stride
                w_start = j * stride
                patch = x[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size]
                regions.append(patch.mean(dim=[2, 3]))

        # Stack into (B, 9, 128)
        regional_tokens = torch.stack(regions, dim=1)

        # 3. Local Auxiliary Prediction (Averaging all 9 tokens)
        local_feat = regional_tokens.mean(dim=1)
        aux_local_logits = self.aux_local_head(local_feat)

        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Concatenate CLS with the 9 regional tokens -> (B, 10, 128)
        T = torch.cat([cls_tokens, regional_tokens], dim=1)

        T = T + self.pos_embed
        T = self.pos_drop(T)

        out = self.transformer(T)
        
        # Only extract the CLS token for classification
        cls_out = out[:, 0, :]
        logits = self.head(cls_out)

        return logits, cls_out, aux_global_logits, aux_local_logits