import torch
import torch.nn as nn

class FRITTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, num_classes=7, dropout=0.5):
        super(FRITTransformer, self).__init__()
        
        # Now we only have 5 tokens: 1 global + 4 regions
        self.num_tokens = 5
        
        # Positional embedding for 5 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head (same as before)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Input: (B, 128, 28, 28)
        B, C, H, W = x.shape

        # -----------------------------
        # 1. Split into 4 regions
        # -----------------------------
        tl = x[:, :, :14, :14]   # top-left
        tr = x[:, :, :14, 14:]   # top-right
        bl = x[:, :, 14:, :14]   # bottom-left
        br = x[:, :, 14:, 14:]   # bottom-right

        # -----------------------------
        # 2. Global Average Pooling
        # -----------------------------
        def gap(t):
            return t.mean(dim=[2, 3])  # (B, 128)

        x1 = gap(tl)
        x2 = gap(tr)
        x3 = gap(bl)
        x4 = gap(br)

        # -----------------------------
        # 3. Global token
        # -----------------------------
        xg = x.mean(dim=[2, 3])  # (B, 128)

        # -----------------------------
        # 4. Stack tokens → (B, 5, 128)
        # Order: [global, tl, tr, bl, br]
        # -----------------------------
        T = torch.stack([xg, x1, x2, x3, x4], dim=1)

        # -----------------------------
        # 5. Add positional embedding
        # -----------------------------
        T = T + self.pos_embed
        T = self.pos_drop(T)

        # -----------------------------
        # 6. Transformer
        # -----------------------------
        T = self.transformer(T)  # (B, 5, 128)

        # -----------------------------
        # 7. Use GLOBAL token only
        # -----------------------------
        features = T[:, 0, :]  # (B, 128)

        # -----------------------------
        # 8. Classification
        # -----------------------------
        logits = self.head(features)

        return logits, features


if __name__ == "__main__":
    dummy_input = torch.randn(2, 128, 28, 28)
    model = FRITTransformer()
    logits, features = model(dummy_input)

    print("Logits shape:", logits.shape)     # (2, 7)
    print("Features shape:", features.shape) # (2, 128)