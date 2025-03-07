import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TimeEmbedding(nn.Module):
    """
    时间步嵌入模块：将时间步转换为特征向量
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 创建时间嵌入的线性层
        self.linear_1 = nn.Linear(dim, dim * 4)
        self.linear_2 = nn.Linear(dim * 4, dim)

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # 计算位置嵌入
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        # 通过MLP处理
        embeddings = self.linear_1(embeddings)
        embeddings = F.silu(embeddings)
        embeddings = self.linear_2(embeddings)
        return embeddings


class CrossAttention(nn.Module):
    """
    交叉注意力模块：用于处理条件信息（如文本特征）
    """

    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8):
        super().__init__()
        inner_dim = query_dim
        context_dim = context_dim if context_dim is not None else query_dim

        self.heads = heads
        self.scale = (query_dim // heads) ** -0.5

        # 定义注意力的Q,K,V转换矩阵
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        context = context if context is not None else x

        # 计算Q,K,V
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # 重塑张量以进行多头注意力计算
        q = q.reshape(x.shape[0], -1, self.heads, q.shape[-1] // self.heads).permute(0, 2, 1, 3)
        k = k.reshape(context.shape[0], -1, self.heads, k.shape[-1] // self.heads).permute(0, 2, 1, 3)
        v = v.reshape(context.shape[0], -1, self.heads, v.shape[-1] // self.heads).permute(0, 2, 1, 3)

        # 计算注意力
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)

        # 应用注意力
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], -1, q.shape[-1] * self.heads)
        return self.to_out(out)


class ResnetBlock(nn.Module):
    """
    残差块：包含时间条件和可选的交叉注意力
    """

    def __init__(self, in_channels: int, out_channels: int, temb_channels: int,
                 groups: int = 32, use_attention: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 时间嵌入投影
        self.temb_proj = nn.Linear(temb_channels, out_channels)

        # 残差连接
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        if use_attention:
            self.attn = CrossAttention(out_channels)
        else:
            self.attn = None

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        # 添加时间嵌入
        temb = self.temb_proj(F.silu(temb))[:, :, None, None]
        h = h + temb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # 应用注意力（如果有）
        if self.attn is not None:
            h = h.reshape(h.shape[0], h.shape[1], -1).transpose(1, 2)
            h = self.attn(h)
            h = h.transpose(1, 2).reshape(x.shape[0], -1, x.shape[2], x.shape[3])

        return h + self.shortcut(x)


class UNet2DConditionModel(nn.Module):
    """
    条件UNet模型：用于生成扩散模型
    """

    def __init__(
            self,
            in_channels: int = 4,
            out_channels: int = 4,
            model_channels: int = 320,
            time_embed_dim: int = 1280,
            context_dim: int = 768,
            attention_levels: Tuple[bool, ...] = (False, True, True, True),
    ):
        super().__init__()

        # 时间嵌入
        self.time_embed = TimeEmbedding(model_channels)
        self.time_embed_mlp = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 初始卷积
        self.input_blocks = nn.ModuleList([
            nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        ])

        # 下采样块
        current_channels = model_channels
        channel_multipliers = [1, 2, 4, 4]
        for level, use_attention in enumerate(attention_levels):
            # 添加残差块
            for _ in range(2):
                layers = [ResnetBlock(
                    current_channels,
                    current_channels * channel_multipliers[level],
                    time_embed_dim,
                    use_attention=use_attention
                )]
                current_channels = current_channels * channel_multipliers[level]
                self.input_blocks.append(nn.ModuleList(layers))

            # 添加下采样
            if level != len(attention_levels) - 1:
                self.input_blocks.append(nn.Conv2d(
                    current_channels, current_channels,
                    kernel_size=3, stride=2, padding=1
                ))

        # 中间块
        self.middle_block = nn.ModuleList([
            ResnetBlock(current_channels, current_channels, time_embed_dim, use_attention=True),
            ResnetBlock(current_channels, current_channels, time_embed_dim, use_attention=False),
        ])

        # 上采样块
        self.output_blocks = nn.ModuleList([])
        for level, use_attention in enumerate(reversed(attention_levels)):
            for _ in range(3):
                layers = [ResnetBlock(
                    current_channels + self.input_blocks[-1].out_channels,
                    current_channels // 2,
                    time_embed_dim,
                    use_attention=use_attention
                )]
                current_channels = current_channels // 2
                self.output_blocks.append(nn.ModuleList(layers))

            # 添加上采样
            if level != len(attention_levels) - 1:
                self.output_blocks.append(nn.Upsample(
                    scale_factor=2, mode='nearest'
                ))

        # 最终输出层
        self.out = nn.Sequential(
            nn.GroupNorm(32, current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. 时间嵌入
        temb = self.time_embed(timesteps)
        temb = self.time_embed_mlp(temb)

        # 2. 下采样路径
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    h = layer(h, temb)
            else:
                h = module(h)
            hs.append(h)

        # 3. 中间块
        for module in self.middle_block:
            h = module(h, temb)

        # 4. 上采样路径
        for module in self.output_blocks:
            if isinstance(module, nn.ModuleList):
                h = torch.cat([h, hs.pop()], dim=1)
                for layer in module:
                    h = layer(h, temb)
            else:
                h = module(h)

        # 5. 输出
        return self.out(h)


# 使用示例
def test_unet():
    # 创建模型
    model = UNet2DConditionModel(
        in_channels=4,
        out_channels=4,
        model_channels=320,
        attention_levels=(False, True, True, True)
    )

    # 创建测试输入
    batch_size = 4
    x = torch.randn(batch_size, 4, 64, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    context = torch.randn(batch_size, 77, 768)  # 文本特征

    # 前向传播
    output = model(x, timesteps, context)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")


if __name__ == "__main__":
    test_unet()

