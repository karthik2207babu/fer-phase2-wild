import torch
import torch.nn as nn

class FRITTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,      # 👇 CHANGED: Default back to 4
        num_layers=2,     # 👇 CHANGED: Default back to 2
        num_classes=7,
        dropout=0.3       # 👇 CHANGED: Default relaxed to 0.3
    ):
        super(FRITTransformer, self).__init__()

        # -------------------------------------------------
        # 4 regional tokens + 1 learnable CLS token = 5 tokens
        # -------------------------------------------------
        self.num_tokens = 5
        
        # Learnable CLS token
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

    def forward(self, x):

        B, C, H, W = x.shape

        # =================================================
        # HARD 2x2 REGIONAL PARTITION (Non-overlapping)
        # =================================================
        h_mid = H // 2
        w_mid = W // 2

        regions = [
            x[:, :, :h_mid, :w_mid],   # top-left
            x[:, :, :h_mid, w_mid:],   # top-right
            x[:, :, h_mid:, :w_mid],   # bottom-left
            x[:, :, h_mid:, w_mid:]    # bottom-right
        ]

        regional_tokens = []

        for region in regions:
            token = region.mean(dim=[2, 3])
            regional_tokens.append(token)

        regional_tokens = torch.stack(regional_tokens, dim=1)
        # Shape: (B, 4, 128)

        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # Shape: (B, 1, 128)

        T = torch.cat([cls_tokens, regional_tokens], dim=1)
        # Shape: (B, 5, 128)

        T = T + self.pos_embed
        T = self.pos_drop(T)

        out = self.transformer(T)
        
        # Only extract the CLS token for classification
        cls_out = out[:, 0, :]
        
        logits = self.head(cls_out)

        return logits, cls_out