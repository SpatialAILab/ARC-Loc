import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    """Spatial feature projection using convolution, normalization, and GELU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.layers(x)


class SpatialContextBlock(nn.Module):
    """Global spatial context aggregation over flattened image tokens."""

    def __init__(self, dim, num_heads=4, mlp_ratio=2):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        b, c, h, w = x.shape

        tokens = x.flatten(2).transpose(1, 2)

        norm_tokens = self.norm1(tokens)
        attended, _ = self.attention(
            norm_tokens,
            norm_tokens,
            norm_tokens,
            need_weights=False,
        )
        tokens = tokens + attended

        tokens = tokens + self.mlp(self.norm2(tokens))

        return tokens.transpose(1, 2).reshape(b, c, h, w)


class ARCDescriptorHead(nn.Module):
    """Generate dense L2-normalized descriptors from backbone features."""

    def __init__(self, in_channels=1024, descriptor_dim=128):
        super().__init__()

        self.projection = nn.Sequential(
            ConvNormAct(in_channels, 512),
            ConvNormAct(512, 256),
            ConvNormAct(256, 128),
        )

        self.context = SpatialContextBlock(
            dim=128,
            num_heads=4,
        )

        self.output = nn.Conv2d(
            128,
            descriptor_dim,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.context(x)
        x = self.output(x)

        return F.normalize(x, p=2, dim=1)


class ARCConfidenceHead(nn.Module):
    """Predict a dense confidence value for every spatial feature."""

    def __init__(self, in_channels=1024):
        super().__init__()

        self.projection = nn.Sequential(
            ConvNormAct(in_channels, 256),
            ConvNormAct(256, 128),
        )

        self.context = SpatialContextBlock(
            dim=128,
            num_heads=4,
        )

        self.output = nn.Conv2d(
            128,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x):
        x = self.projection(x)
        x = self.context(x)
        x = self.output(x)

        return torch.sigmoid(x)