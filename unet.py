import torch
import torcn.nn as nn


def get_time_embedding(time_steps, t_emb_dim):
    # create a tensor of frequencies starting from 1 and increasing exponentially to 10000
    factor = 10000 ** ((torch.arange(
        start = 0, end = t_emb_dim // 2, device = time_steps.device) / (t_emb_dim // 2)
        ))
    # first transform 1D tensor to 2D tensor, then repeat until we reach the number of values in factor
    t_emb = time_steps[:, None].repeat(1, t_emb_dim // 2) / factor
    # take sine and cosine of above result, and concatenate across last (inner) dimension
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample # a boolean value
        # first convolution block. normalize, actiavtion function, and convolution layer
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        # inject positional embeddings after matching shape to first resnet block output
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        # second convolition block
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        # attention block
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        # residual skip connection
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # downsample
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4,
                                        stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = x
        # Resnet block
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h*w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn

        out = self.down_sample_conv(out)
        return out