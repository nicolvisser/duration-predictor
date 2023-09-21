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
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-100-03d12494.ckpt",
                4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-100-dp-lambda-4-1c5be544.ckpt",
                8: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-100-dp-lambda-8-64b20a14.ckpt",
                12: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-100-dp-lmbda-12-0fb7e92a.ckpt",
                16: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-100-dp-lmbda-16-46dba73c.ckpt",
                20: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-100-dp-lmbda-20-eb5e35d8.ckpt",
            },
            200: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-200-77231421.ckpt",
                4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-200-dp-lambda-4-f10f0ece.ckpt",
                8: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-200-dp-lambda-8-20c83757.ckpt",
                12: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-200-dp-lambda-12-66b6bfc0.ckpt",
                16: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-200-dp-lmbda-16-070539c2.ckpt",
                20: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-200-dp-lmbda-20-6f5becd2.ckpt",
            },
            500: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-6e22540b.ckpt",
                4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lambda-4-2bb539be.ckpt",
                8: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lambda-8-4a6ca572.ckpt",
                12: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lambda-12-ee5fe6f0.ckpt",
                16: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lambda-16-8dbea6c3.ckpt",
                20: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lmbda-20-04cced3f.ckpt",
                24: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lmbda-24-f4de429b.ckpt",
                28: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-500-dp-lmbda-28-7c6414b1.ckpt",
            },
            1000: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-1000-b6780de5.ckpt",
                4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-1000-dp-lambda-4-92bdf960.ckpt",
                8: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-1000-dp-lmbda-8-35eeab02.ckpt",
                12: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-1000-dp-lmbda-12-c3488b6b.ckpt",
                16: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-1000-dp-lmbda-16-2b581e8c.ckpt",
                20: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-1000-dp-lmbda-20-6a19f0c5.ckpt",
            },
            2000: {
                0: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-2000-ee1cf808.ckpt",
                4: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-2000-dp-lmbda-4-fe8e3789.ckpt",
                8: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-2000-dp-lmbda-8-680dd026.ckpt",
                12: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-2000-dp-lmbda-12-098a31b3.ckpt",
                16: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-2000-dp-lmbda-16-ea386768.ckpt",
                20: "https://github.com/nicolvisser/duration-predictor/releases/download/v0.1/duration-hubert-bshall-ljspeech-kmeans-2000-dp-lmbda-20-f6749c04.ckpt",
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
    assert dataset in allowed_datasets, f"dataset must be one of {allowed_datasets}, if you choose {features}"
    allowed_units = URLS[features][dataset].keys()
    assert n_units in allowed_units, f"n_units must be one of {allowed_units}, if you choose {features} and {dataset}"
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
