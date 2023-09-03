dependencies = ["torch"]

URLS = {
    "librispeech": {
        50: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/librispeech-kmeans-50.ckpt"
        },
        100: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/librispeech-kmeans-100.ckpt"
        },
        200: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/librispeech-kmeans-200.ckpt"
        },
        500: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/librispeech-kmeans-500.ckpt"
        },
        1000: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/librispeech-kmeans-1000.ckpt"
        },
        2000: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/librispeech-kmeans-2000.ckpt",
        },
    },
    "ljspeech": {
        50: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-50.ckpt"
        },
        100: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-100.ckpt"
        },
        200: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-200.ckpt"
        },
        500: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-500.ckpt",
            4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-500-dp-lambda-4.ckpt",
            8: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-500-dp-lambda-8.ckpt",
            12: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-500-dp-lambda-12.ckpt",
            16: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-500-dp-lambda-16.ckpt",
        },
        1000: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-1000.ckpt",
            4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-1000-dp-lambda-4.ckpt",
        },
        2000: {
            0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/ljspeech-kmeans-2000.ckpt"
        },
    },
}

import torch

from duration_predictor import DurationPredictor
from load_from_checkpoint import _load_model_from_checkpoint


def duration_predictor(
    dataset: str,
    n_clusters: int,
    lmbda: int = 0,
    pretrained: bool = True,
    progress: bool = True,
) -> DurationPredictor:
    # Check that the dataset, n_clusters and dp_smoothing_lambda are available
    allowed_datasets = URLS.keys()
    assert dataset in allowed_datasets, f"dataset must be one of {allowed_datasets}"
    allowed_n_clusters = URLS[dataset].keys()
    assert (
        n_clusters in allowed_n_clusters
    ), f"n_clusters must be one of {allowed_n_clusters} when using {dataset} dataset"

    if lmbda > 0:
        allowed_lmbdas = URLS[dataset][n_clusters].keys()
        assert (
            lmbda in allowed_lmbdas
        ), f"lmbda must be one of {allowed_lmbdas} when using {dataset} dataset and {n_clusters} clusters"

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[dataset][n_clusters][lmbda], progress=progress
        )
        model = _load_model_from_checkpoint(checkpoint)
        model.eval()
    else:
        model = DurationPredictor(
            num_codes=n_clusters,
            embedding_dim=128,
            nhead=2,
            conv_channels=1024,
            conv_kernel_size=3,
            attn_dropout=0.1,
            conv_dropout=0.1,
            num_blocks=1,
        )

    return model
