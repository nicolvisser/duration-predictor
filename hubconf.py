dependencies = ["torch"]

URLS = {
    "hubert-bshall": {
        "librispeech": {
            50: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-librispeech-kmeans-50-15397ff8.ckpt"
            },
            100: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-librispeech-kmeans-100-7a6dbca0.ckpt"
            },
            200: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-librispeech-kmeans-200-9cf35d6d.ckpt"
            },
            500: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-librispeech-kmeans-500-7caf8796.ckpt"
            },
            1000: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-librispeech-kmeans-1000-10c48d81.ckpt"
            },
            2000: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-librispeech-kmeans-2000-ec537612.ckpt",
            },
        },
        "ljspeech": {
            50: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-50-39d1966d.ckpt"
            },
            100: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-100-03d12494.ckpt"
            },
            200: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-200-77231421.ckpt"
            },
            500: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-6e22540b.ckpt",
                4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lambda-4-2bb539be.ckpt",
                8: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lambda-8-4a6ca572.ckpt",
                12: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lambda-12-ee5fe6f0.ckpt",
                16: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lambda-16-8dbea6c3.ckpt",
            },
            1000: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-1000-b6780de5.ckpt",
                4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-1000-dp-lambda-4-92bdf960.ckpt",
            },
            2000: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-2000-ee1cf808.ckpt"
            },
        },
    },
}

import torch

from duration_predictor import DurationPredictor


def duration_predictor(
    features: str = "hubert-bshall",
    dataset: str = "ljspeech",
    n_units: int = 500,
    dp_lmbda: int = 0,
    pretrained: bool = True,
    progress: bool = True,
) -> DurationPredictor:
    # Check that the dataset, n_clusters and dp_smoothing_lambda are available
    allowed_features = URLS.keys()
    assert features in allowed_features, f"features must be one of {allowed_features}"
    allowed_datasets = URLS[features].keys()
    assert (
        dataset in allowed_datasets
    ), f"dataset must be one of {allowed_datasets}, if you choose {features}"
    allowed_units = URLS[features][dataset].keys()
    assert (
        n_units in allowed_units
    ), f"n_units must be one of {allowed_units}, if you choose {features} and {dataset}"
    allowed_lmbdas = URLS[features][dataset][n_units].keys()
    assert (
        dp_lmbda in allowed_lmbdas
    ), f"dp_lmbda must be one of {allowed_lmbdas}, if you choose {features}, {dataset} and {n_units} units"

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            URLS[features][dataset][n_units][dp_lmbda],
            progress=progress,
            check_hash=True,
        )
        model = DurationPredictor.load_model_from_lit_checkpoint(checkpoint)
        model.eval()
    else:
        model = DurationPredictor()

    return model
