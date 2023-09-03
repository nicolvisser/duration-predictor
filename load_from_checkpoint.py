import torch

from duration_predictor.model import DurationPredictor


def load_model_from_checkpoint(lit_checkpoint_path):
    checkpoint = torch.load(lit_checkpoint_path)
    model = _load_model_from_checkpoint(checkpoint)
    return model


def _load_model_from_checkpoint(lit_checkpoint):
    hyper_parameters = lit_checkpoint["hyper_parameters"]
    model = DurationPredictor(**hyper_parameters)
    model_weights = lit_checkpoint["state_dict"]
    for key in list(model_weights.keys()):
        model_weights[key.replace("dur_predictor.", "")] = model_weights.pop(key)
    model.load_state_dict(model_weights)
    model.eval()
    return model
