# import torch
# import torch.nn as nn

# class FRITTransformer(nn.Module):
#     def __init__(
#         self,
#         embed_dim=128,
#         num_heads=8,      
#         num_local_layers=2,  
#         num_classes=7,
#         dropout=0.5       
#     ):
#         super(FRITTransformer, self).__init__()

#         self.num_patches = 9
#         self.pos_drop = nn.Dropout(p=dropout)
        
#         self.local_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
#         local_layer = nn.TransformerEncoderLayer(
#             d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
#             dropout=dropout, activation='gelu', batch_first=True
#         )
#         self.local_transformer = nn.TransformerEncoder(local_layer, num_layers=num_local_layers)

#         self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
#         self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
        
#         self.ffn = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 4, embed_dim),
#             nn.Dropout(dropout)
#         )

#         self.head = nn.Sequential(
#             nn.LayerNorm(embed_dim),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim, num_classes)
#         )
#         self.aux_global_head = nn.Linear(embed_dim, num_classes)
#         self.aux_local_head = nn.Linear(embed_dim, num_classes)

#     def forward(self, x):
#         B, C, H, W = x.shape

#         global_feat = x.mean(dim=[2, 3]) 
#         aux_global_logits = self.aux_global_head(global_feat)

#         patch_size = 12
#         stride = 8
#         regions = []
#         for i in range(3):
#             for j in range(3):
#                 h_start, w_start = i * stride, j * stride
#                 patch = x[:, :, h_start:h_start+patch_size, w_start:w_start+patch_size]
#                 regions.append(patch.mean(dim=[2, 3]))
        
#         regional_tokens = torch.stack(regions, dim=1) 

#         T_local = self.pos_drop(regional_tokens + self.local_pos_embed)
#         T_local_out = self.local_transformer(T_local) 
        
#         local_feat = T_local_out.mean(dim=1)
#         aux_local_logits = self.aux_local_head(local_feat)

#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         attn_out, _ = self.cross_attn(query=cls_tokens, key=T_local_out, value=T_local_out)
        
#         cls_tokens = self.norm1(cls_tokens + attn_out)
#         ffn_out = self.ffn(cls_tokens)
#         cls_tokens = self.norm2(cls_tokens + ffn_out)
        
#         cls_out = cls_tokens.squeeze(1) 
#         logits = self.head(cls_out)

#         return logits, cls_out, aux_global_logits, aux_local_logits


import torch
import torch.nn as nn
import torch.nn.functional as F
from safm import SAFM


class FRITTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        num_local_layers=2,
        num_classes=8,
        dropout=0.5
    ):
        super().__init__()

        self.num_regions = 4

        self.local_safm = SAFM(
            kernel_size=7
        )

        self.local_pool_proj = nn.Sequential(
            nn.Linear(
                embed_dim * 2,
                embed_dim
            ),
            nn.LayerNorm(
                embed_dim
            )
        )

        self.global_pool_proj = nn.Sequential(
            nn.Linear(
                embed_dim * 2,
                embed_dim
            ),
            nn.LayerNorm(
                embed_dim
            )
        )

        self.pos_embed = nn.Parameter(
            torch.randn(
                1,
                5,
                embed_dim
            )
        )

        self.local_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True
            ),
            num_layers=num_local_layers
        )

        self.aux_global_head = nn.Linear(
            embed_dim,
            num_classes
        )

        self.aux_local_head = nn.Linear(
            embed_dim,
            num_classes
        )

    def _pool_local_region(self, region):
        avg_feat = F.adaptive_avg_pool2d(
            region,
            1
        ).flatten(1)

        max_feat = F.adaptive_max_pool2d(
            region,
            1
        ).flatten(1)

        pooled = torch.cat(
            [
                avg_feat,
                max_feat
            ],
            dim=1
        )

        return self.local_pool_proj(
            pooled
        )

    def forward(self, x):
        B, C, H, W = x.shape

        if H != 28 or W != 28:
            raise ValueError(
                f"Expected 28x28 feature map, got {H}x{W}"
            )

        global_avg = F.adaptive_avg_pool2d(
            x,
            1
        ).flatten(1)

        global_max = F.adaptive_max_pool2d(
            x,
            1
        ).flatten(1)

        global_feat = self.global_pool_proj(
            torch.cat(
                [
                    global_avg,
                    global_max
                ],
                dim=1
            )
        )

        aux_global_logits = self.aux_global_head(
            global_feat
        )

        tl = self.local_safm(
            x[:, :, :14, :14]
        )

        tr = self.local_safm(
            x[:, :, :14, 14:]
        )

        bl = self.local_safm(
            x[:, :, 14:, :14]
        )

        br = self.local_safm(
            x[:, :, 14:, 14:]
        )

        local_tokens = torch.stack(
            [
                self._pool_local_region(tl),
                self._pool_local_region(tr),
                self._pool_local_region(bl),
                self._pool_local_region(br)
            ],
            dim=1
        )

        tokens = torch.cat(
            [
                global_feat.unsqueeze(1),
                local_tokens
            ],
            dim=1
        )

        tokens = (
            tokens
            + self.pos_embed
        )

        relation = self.local_transformer(
            tokens
        )

        fused_global = relation[
            :, 0, :
        ]

        fused_local = relation[
            :, 1:, :
        ]

        aux_local_logits = self.aux_local_head(
            fused_local.mean(dim=1)
        )

        return (
            None,
            fused_global,
            aux_global_logits,
            aux_local_logits
        )