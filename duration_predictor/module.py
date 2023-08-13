import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import optim

from duration_predictor.model import DurationPredictor


class LitDurationPredictor(pl.LightningModule):
    def __init__(
        self,
        num_codes,
        embedding_dim=128,
        nhead=2,
        conv_channels=1024,
        conv_kernel_size=3,
        attn_dropout=0.1,
        conv_dropout=0.1,
        num_blocks=1,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.dur_predictor = DurationPredictor(
            num_codes=num_codes,
            embedding_dim=embedding_dim,
            nhead=nhead,
            conv_channels=conv_channels,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            conv_dropout=conv_dropout,
            num_blocks=num_blocks,
        )

    def training_step(self, batch, batch_idx):
        codes, durations, lengths = batch

        log_durations = torch.log2(durations + 1)
        log_durations_ = self.dur_predictor(codes, lengths)

        duration_loss = F.mse_loss(log_durations_, log_durations, reduction="none")
        duration_loss = torch.sum(duration_loss, dim=-1) / lengths
        duration_loss = torch.mean(duration_loss)
        self.log("train_loss", duration_loss, prog_bar=True)
        return duration_loss

    def validation_step(self, batch, batch_idx):
        codes, durations, lengths = batch

        log_durations = torch.log2(durations + 1)
        log_durations_ = self.dur_predictor(codes, lengths)

        duration_loss = F.mse_loss(log_durations_, log_durations, reduction="none")
        duration_loss = torch.sum(duration_loss, dim=-1) / lengths
        duration_loss = torch.mean(duration_loss)
        self.log("val_loss", duration_loss, prog_bar=True)
        return duration_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, src, src_lengths):
        return self.dur_predictor.predict(src, src_lengths)
