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
        # 16 regional tokens + 1 global token = 17 tokens
        # -------------------------------------------------
        self.num_tokens = 17

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_tokens, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=dropout)

        # -------------------------------------------------
        # Transformer Encoder
        # -------------------------------------------------
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

        # -------------------------------------------------
        # Classification Head
        # -------------------------------------------------
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        """
        Input:
            x -> (B, 128, 28, 28)

        Output:
            logits   -> (B, num_classes)
            features -> (B, 128)
        """

        B, C, H, W = x.shape

        # =================================================
        # 1. Create 4×4 grid tokens
        # Each patch = 7×7
        # =================================================

        patch_size = 7
        tokens = []

        for i in range(4):
            for j in range(4):

                patch = x[
                    :,
                    :,
                    i*patch_size:(i+1)*patch_size,
                    j*patch_size:(j+1)*patch_size
                ]

                # Global Average Pool each patch
                token = patch.mean(dim=[2, 3])  # (B,128)

                tokens.append(token)

        # =================================================
        # 2. Stack regional tokens
        # =================================================

        regional_tokens = torch.stack(tokens, dim=1)
        # Shape: (B,16,128)

        # =================================================
        # 3. Global token
        # =================================================

        global_token = x.mean(dim=[2, 3]).unsqueeze(1)
        # Shape: (B,1,128)

        # =================================================
        # 4. Final token structure
        # [global | 16 regional]
        # =================================================

        T = torch.cat([global_token, regional_tokens], dim=1)
        # Shape: (B,17,128)

        # =================================================
        # 5. Add positional embeddings
        # =================================================

        T = T + self.pos_embed
        T = self.pos_drop(T)

        # =================================================
        # 6. Transformer Encoder
        # =================================================

        T = self.transformer(T)
        # Shape: (B,17,128)

        # =================================================
        # 7. Use GLOBAL token only
        # (Keeping minimal architectural change)
        # =================================================

        features = T[:, 0, :]
        # Shape: (B,128)

        # =================================================
        # 8. Classification
        # =================================================

        logits = self.head(features)

        return logits, features


if __name__ == "__main__":

    dummy_input = torch.randn(2, 128, 28, 28)

    model = FRITTransformer()

    logits, features = model(dummy_input)

    print("Logits shape:", logits.shape)
    print("Features shape:", features.shape)

    # Expected:
    # Logits   -> (2,7)
    # Features -> (2,128)