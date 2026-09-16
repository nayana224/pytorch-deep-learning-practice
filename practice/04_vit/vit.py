import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=384, patch_size=16, dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x, return_attention=False):
        q = self.norm1(x)
        y, attn = self.attn(q, q, q, need_weights=return_attention, average_attn_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return (x, attn) if return_attention else x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=384, patch_size=16, num_classes=100, dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        self.patch = PatchEmbedding(image_size, patch_size, dim)
        n_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, return_features=False, return_attention=False):
        patches = self.patch(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos
        attentions = []
        for block in self.blocks:
            if return_attention:
                tokens, attn = block(tokens, True)
                attentions.append(attn)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)
        logits = self.head(tokens[:, 0])
        if return_features or return_attention:
            return logits, {"patches": patches, "tokens": tokens, "attentions": attentions}
        return logits


def vit_b16(num_classes=100, image_size=384):
    return VisionTransformer(image_size=image_size, patch_size=16, num_classes=num_classes, dim=768, depth=12, heads=12, mlp_dim=3072)


def vit_tiny16(num_classes=100, image_size=224):
    """Scaled local variant preserving the paper data flow."""
    return VisionTransformer(image_size=image_size, patch_size=16, num_classes=num_classes, dim=192, depth=12, heads=3, mlp_dim=768)
