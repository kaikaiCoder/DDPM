import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class AttentionBlock(nn.Module):
    def __init__(self, units, groups):
        super(AttentionBlock, self).__init__()
        self.units = units
        self.groups = groups
        self.norm = nn.GroupNorm(num_groups=groups, num_channels=units)
        self.q = nn.Conv2d(units, units, 1)
        self.k = nn.Conv2d(units, units, 1)
        self.v = nn.Conv2d(units, units, 1)
        self.proj = nn.Conv2d(units, units, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        # 计算注意力分数
        q = self.q(x).reshape(B, self.groups, -1, H * W)
        k = self.k(x).reshape(B, self.groups, -1, H * W)
        v = self.v(x).reshape(B, self.groups, -1, H * W)
        attn = torch.einsum("bgci,bgdi->bgcd", q, k) / (C**0.5)
        attn = F.softmax(attn, dim=-1)

        x = torch.einsum("bgcd,bgdi->bgci", attn, v).reshape(B, C, H, W)
        x = x + self.proj(x)
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.half_dim = dim // 2
        self.emb = math.log(10000) / (self.half_dim - 1)
        self.emb = torch.exp(
            torch.arange(self.half_dim, dtype=torch.float32) * -self.emb
        )

    def forward(self, x):
        x = x.float()
        emb = x[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_groups, t_dim, has_attention):
        super(ResidualBlock, self).__init__()
        self.norm_groups = norm_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hasAttention = has_attention
        self.attention = (
            AttentionBlock(groups=norm_groups, units=out_channels)
            if has_attention
            else None
        )
        self.norm1 = nn.GroupNorm(num_groups=norm_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Dense layer for time embedding
        self.time_dense = nn.Linear(t_dim, out_channels)

        # 1x1 convolution to match dimensions if needed
        self.conv1x1 = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x, t):
        residual = x

        if self.conv1x1 is not None:
            residual = self.conv1x1(x)

        t = self.time_dense(F.silu(t)).unsqueeze(2).unsqueeze(3)

        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = x + t

        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        x = x + residual

        if self.attention:
            x = self.attention(x)

        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.down = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, interpolation="nearest"):
        super(UpSample, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode=interpolation)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class MLP(nn.Module):
    def __init__(self, units):
        super(MLP, self).__init__()
        self.units = units
        self.fc1 = nn.Linear(units, units)
        self.fc2 = nn.Linear(units, units)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = self.fc2(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        widths,
        has_attention,
        in_channels=3,
        out_channels=3,
        num_res_blocks=1,
        norm_groups=8,
        interpolation="nearest",
    ):
        super(UNet, self).__init__()
        self.widths = widths
        self.first_conv_channels = widths[0]
        self.inital = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.first_conv_channels,
            kernel_size=3,
            padding=1,
        )
        self.t_dim = self.first_conv_channels * 4
        self.time_embedding = TimeEmbedding(self.t_dim)
        self.time_mlp = MLP(self.t_dim)

        # encoder
        self.enc1 = ResidualBlock(
            in_channels=self.first_conv_channels,
            out_channels=widths[0],
            norm_groups=norm_groups,
            t_dim=self.t_dim,
            has_attention=has_attention[0],
        )
        self.down1 = DownSample(widths[0])
        self.enc2 = ResidualBlock(
            in_channels=widths[0],
            out_channels=widths[1],
            norm_groups=norm_groups,
            t_dim=self.t_dim,
            has_attention=has_attention[1],
        )
        self.down2 = DownSample(widths[1])
        self.enc3 = ResidualBlock(
            in_channels=widths[1],
            out_channels=widths[2],
            norm_groups=norm_groups,
            t_dim=self.t_dim,
            has_attention=has_attention[2],
        )
        self.down3 = DownSample(widths[2])
        self.enc4 = ResidualBlock(
            in_channels=widths[2],
            out_channels=widths[3],
            norm_groups=norm_groups,
            t_dim=self.t_dim,
            has_attention=has_attention[3],
        )

        self.middle_blocks = nn.ModuleList()

        # decoder
        self.up3 = UpSample(
            in_channels=widths[3], out_channels=widths[2], interpolation=interpolation
        )
        self.dec3 = ResidualBlock(
            in_channels=widths[3],
            out_channels=widths[2],
            norm_groups=norm_groups,
            t_dim=self.t_dim,
            has_attention=has_attention[2],
        )
        self.up2 = UpSample(
            in_channels=widths[2], out_channels=widths[1], interpolation=interpolation
        )
        self.dec2 = ResidualBlock(
            in_channels=widths[2],
            out_channels=widths[1],
            norm_groups=norm_groups,
            t_dim=self.t_dim,
            has_attention=has_attention[1],
        )
        self.up1 = UpSample(
            in_channels=widths[1], out_channels=widths[0], interpolation=interpolation
        )
        self.dec1 = ResidualBlock(
            in_channels=widths[1],
            out_channels=widths[0],
            norm_groups=norm_groups,
            t_dim=self.t_dim,
            has_attention=has_attention[0],
        )

        self.final_norm = nn.GroupNorm(num_groups=norm_groups, num_channels=widths[0])
        self.final_conv = nn.Conv2d(
            widths[0], out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x, t):
        t = self.time_embedding(t)
        t = self.time_mlp(t)
        x = self.inital(x)

        # encoder
        x1 = self.enc1(x, t)  # [B, 64, H, W]
        x2 = self.enc2(self.down1(x1), t)  # [B, 128, H/2, W/2]
        x3 = self.enc3(self.down2(x2), t)  # [B, 256, H/4, W/4]
        x4 = self.enc4(self.down3(x3), t)  # [B, 512, H/8, W/8]

        # decoder
        x = self.up3(x4)  # [B, 256, H/4, W/4]
        x = self.dec3(torch.cat([x, x3], dim=1), t)  # [B, 256, H/4, W/4]
        x = self.up2(x)  # [B, 128, H/2, W/2]
        x = self.dec2(torch.cat([x, x2], dim=1), t)  # [B, 128, H/2, W/2]
        x = self.up1(x)  # [B, 64, H, W]
        x = self.dec1(torch.cat([x, x1], dim=1), t)  # [B, 64, H, W]

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)
        return x


class Sample_UNet(nn.Module):
    def __init__(self):
        super(Sample_UNet, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, t):
        x1 = self.down1(x)
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = F.interpolate(x2, scale_factor=2, mode="nearest")
        x4 = self.up1(x3)
        x5 = self.up2(x4 + x1)
        return x5
