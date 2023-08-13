import click
import torch
from duration_predictor.model import DurationPredictor

def load_model_from_checkpoint(lit_checkpoint_path):

    checkpoint = torch.load(lit_checkpoint_path)

    hyper_parameters = checkpoint["hyper_parameters"]

    model = DurationPredictor(**hyper_parameters)

    model_weights = checkpoint["state_dict"]

    for key in list(model_weights.keys()):
        model_weights[key.replace("dur_predictor.", "")] = model_weights.pop(key)

    model.load_state_dict(model_weights)
    model.eval()

    return model

if __name__ == "__main__":
    lit_checkpoint_path = click.prompt("Path to the checkpoint", type=click.Path(exists=True, dir_okay=False, file_okay=True))
    model = load_model_from_checkpoint(lit_checkpoint_path)
    model.cuda()

    print(model)