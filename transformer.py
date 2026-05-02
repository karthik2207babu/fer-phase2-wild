import torch
import torch.nn as nn

class FRITTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, num_classes=7, dropout=0.5):
        super(FRITTransformer, self).__init__()
        
        # Spatial dimensions from LFA/SAFM are 28x28
        self.num_patches = 28 * 28
        
        # [CLS] token for the final classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embeddings (784 patches + 1 CLS token)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Transformer Encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # Input: (Batch, 128, 28, 28) from SAFM
        B, C, H, W = x.shape
        
        # 1. Tokenization: Flatten 28x28 spatial grid into 784 sequence tokens
        # Reshape to (Batch, 128, 784) then transpose to (Batch, 784, 128)
        x = x.view(B, C, -1).permute(0, 2, 1)
        
        # 2. Append the [CLS] token to the sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: (Batch, 785, 128)
        
        # 3. Add positional embeddings so the Transformer knows where tokens came from
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Pass through Transformer layers
        x = self.transformer(x)
        
        # 5. Extract ONLY the [CLS] token for our final prediction
        features = x[:, 0, :]  # Shape: (Batch, 128)
        
        # 6. Final classification
        logits = self.head(features)
        
        # Return both for the CombinedFERLoss (CrossEntropy needs logits, Cluster needs features)
        return logits, features

if __name__ == "__main__":
    # Test tensor representing SAFM output: (Batch, 128, 28, 28)
    dummy_input = torch.randn(2, 128, 28, 28) 
    transformer = FRITTransformer()
    logits, features = transformer(dummy_input)
    
    print(f"Logits shape: {logits.shape}")       # Expected: [2, 7]
    print(f"Features shape: {features.shape}")   # Expected: [2, 128]