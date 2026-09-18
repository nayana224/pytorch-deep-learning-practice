"""Vision Transformer 실습 파일.

논문 구조와 핵심 메커니즘을 읽기 쉽게 따라가기 위한 공부용 코드다.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=384, patch_size=16, hidden_dim=768):
        super().__init__()
        self.projection = nn.Conv2d(
            3,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.projection(x)      # [B, D, H/P, W/P]
        x = x.flatten(2)            # [B, D, N]
        x = x.transpose(1, 2)       # [B, N, D]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=12, mlp_dim=3072):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x, return_attention=False):
        normalized = self.norm1(x)
        attention_output, attention_map = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + attention_output

        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output

        if return_attention:
            return x, attention_map
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=384,
        patch_size=16,
        num_classes=100,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size,
            patch_size,
            hidden_dim,
        )

        num_patches = (image_size // patch_size) ** 2

        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, hidden_dim)
        )

        self.encoder_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    hidden_dim,
                    num_heads,
                    mlp_dim,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        x,
        return_features=False,
        return_attention=False,
    ):
        patch_tokens = self.patch_embedding(x)

        cls_token = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls_token, patch_tokens], dim=1)
        tokens = tokens + self.position_embedding

        attention_maps = []

        for block in self.encoder_blocks:
            if return_attention:
                tokens, attention = block(
                    tokens,
                    return_attention=True,
                )
                attention_maps.append(attention)
            else:
                tokens = block(tokens)

        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])

        if return_features or return_attention:
            return logits, {
                "patches": patch_tokens,
                "tokens": tokens,
                "attentions": attention_maps,
            }

        return logits


def vit_b16(num_classes=100, image_size=384):
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        hidden_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
    )


def vit_tiny16(num_classes=100, image_size=224):
    """Scaled local variant with the same patch/token/encoder flow."""
    return VisionTransformer(
        image_size=image_size,
        patch_size=16,
        num_classes=num_classes,
        hidden_dim=192,
        depth=12,
        num_heads=3,
        mlp_dim=768,
    )
