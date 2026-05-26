import torch
import torch.nn as nn

class FRITTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        num_classes=7,
        dropout=0.5
    ):
        super(FRITTransformer, self).__init__()

        # -------------------------------------------------
        # 4 regional tokens + 1 global token = 5 tokens
        # -------------------------------------------------
        self.num_tokens = 5

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
        # 2×2 regional partition
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
        # Shape: (B,4,128)

        global_token = x.mean(dim=[2, 3]).unsqueeze(1)
        # Shape: (B,1,128)

        T = torch.cat([global_token, regional_tokens], dim=1)
        # Shape: (B,5,128)

        T = T + self.pos_embed
        T = self.pos_drop(T)

        T = self.transformer(T)

        # Use global token
        features = T[:, 0, :]

        logits = self.head(features)

        return logits, features


if __name__ == "__main__":

    dummy_input = torch.randn(2, 128, 28, 28)

    model = FRITTransformer()

    logits, features = model(dummy_input)

    print("Logits shape:", logits.shape)
    print("Features shape:", features.shape)