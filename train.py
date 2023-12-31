import click
import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from duration_predictor.dataset import DurationsDataset, collate_fn
from duration_predictor.module import LitDurationPredictor


@click.command()
@click.option(
    "--data-dir",
    help="Path to directory containing collapsed codes and durations",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    prompt=True,
)
@click.option(
    "--version-name",
    help="The version name for tensorboard logging",
    type=str,
    prompt=True,
)
@click.option(
    "--num-codes",
    help="The number of unique codes",
    type=int,
    prompt=True,
)
def main(data_dir, version_name, num_codes):

    train_dataset = DurationsDataset(
        root=data_dir,
        subset="train",
    )

    val_dataset = DurationsDataset(
        root=data_dir,
        subset="dev",
    )

    BATCH_SIZE = 256

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    duration_predictor = LitDurationPredictor(
        num_codes=num_codes,
        embedding_dim=128,
        nhead=2,
        conv_channels=1024,
        conv_kernel_size=3,
        attn_dropout=0.1,
        conv_dropout=0.1,
        num_blocks=1,
    )

    tensorboard = pl_loggers.TensorBoardLogger(save_dir="", version=version_name)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, save_last=True, monitor="val_loss"
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        logger=tensorboard,
        max_epochs=-1,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(
        model=duration_predictor,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
