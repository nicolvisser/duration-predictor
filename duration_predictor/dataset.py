from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DurationsDataset(Dataset):
    def __init__(self, root: str, subset: str):
        self.paths = sorted(list(Path(root).glob(f"./{subset}*/**/*.npz")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        data = np.load(path)
        codes, durations = data["codes"], data["durations"]

        codes = torch.from_numpy(codes).long()
        durations = torch.from_numpy(durations).long()

        return codes, durations


def collate_fn(batch):
    codes = [x[0] for x in batch]
    durations = [x[1] for x in batch]
    codes = torch.nn.utils.rnn.pad_sequence(codes, batch_first=True)
    durations = torch.nn.utils.rnn.pad_sequence(durations, batch_first=True)
    lengths = torch.sum(durations != 0, dim=-1)
    return codes, durations, lengths
