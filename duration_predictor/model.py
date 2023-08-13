import copy
import math

import torch
import torch.nn as nn


def get_mask_from_lengths(lengths):
    max_length = torch.max(lengths)
    mask = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    mask = mask < lengths.unsqueeze(1)
    return mask


class FFTBlock(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        conv_channels,
        conv_kernel_size,
        attn_dropout,
        conv_dropout,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        self.conv1 = nn.Conv1d(
            d_model, conv_channels, conv_kernel_size, 1, (conv_kernel_size - 1) // 2
        )
        self.conv2 = nn.Conv1d(
            conv_channels, d_model, conv_kernel_size, 1, (conv_kernel_size - 1) // 2
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(attn_dropout)
        self.dropout2 = nn.Dropout(conv_dropout)

    def forward(self, x, mask):
        attn, _ = self.self_attn(x, x, x, key_padding_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout1(attn))

        x = x.masked_fill(mask.unsqueeze(-1), 0)

        conv = self.conv2(torch.relu(self.conv1(x.transpose(1, 2))))
        x = self.norm2(x + self.dropout2(conv.transpose(1, 2)))

        x = x.masked_fill(mask.unsqueeze(-1), 0)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, num_codes, embedding_dim, encoder_block, num_blocks):
        super().__init__()
        self.code_embedding = nn.Embedding(num_codes, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim)
        self.blocks = nn.ModuleList(
            [copy.deepcopy(encoder_block) for _ in range(num_blocks)]
        )

    def forward(self, x, mask):
        x = self.code_embedding(x)
        x = self.positional_encoding(x)

        x = x.masked_fill(mask.unsqueeze(-1), 0)
        for block in self.blocks:
            x = block(x, mask)
        return x


class DurationPredictor(nn.Module):
    def __init__(
        self,
        num_codes,
        embedding_dim,
        nhead,
        conv_channels,
        conv_kernel_size,
        attn_dropout,
        conv_dropout,
        num_blocks,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            encoder_block=FFTBlock(
                d_model=embedding_dim,
                nhead=nhead,
                conv_channels=conv_channels,
                conv_kernel_size=conv_kernel_size,
                attn_dropout=attn_dropout,
                conv_dropout=conv_dropout,
            ),
            num_blocks=num_blocks,
        )
        self.projection = nn.Linear(embedding_dim, 1)

    def forward(self, src, src_lengths):
        mask = ~get_mask_from_lengths(src_lengths)
        src = self.encoder(src, mask)
        src = src.masked_fill(mask.unsqueeze(-1), 0)
        log_durations = self.projection(src)
        return log_durations.squeeze(-1)

    @torch.inference_mode()
    def predict(self, src, src_lengths):
        log_durations = self.__call__(src, src_lengths)
        durations = torch.round((2**log_durations - 1))
        return durations.long()
