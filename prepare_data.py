from pathlib import Path

import click
import numpy as np
from tqdm import tqdm


def dedupe(units):
    """
    Removes consecutive entries that are duplicates.

    Parameters:
    - units: An ndarray of duplicated (uncollapsed) units.

    Returns:
    A tuple where the first entry is the deduped codes and the second entry is the duration of each code.
    """

    codes = []
    durations = []

    for i in range(len(units)):
        if i == 0 or units[i] != units[i - 1]:
            codes.append(units[i])
            durations.append(1)
        else:
            durations[-1] += 1

    assert len(codes) == len(durations)  # Sanity check

    return np.array(codes), np.array(durations)


def pad_sequences(sequences, pad_value=0):
    """
    Pad sequences with pad_value so they all have the same length.

    Parameters:
    - sequences: A list of lists or arrays with varying lengths.
    - pad_value: The value used for padding. Default is 0.

    Returns:
    A 2D numpy array with all sequences having the same length.
    """

    max_length = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        padding_length = max_length - len(seq)
        padded_seq = np.pad(seq, (0, padding_length), constant_values=pad_value)
        padded_sequences.append(padded_seq)

    return np.vstack(padded_sequences)


@click.command()
@click.option(
    "--in_dir",
    "-i",
    help="Path to the directory containing the units. Each utterance has one .npy file that contains the duplicated (uncollapsed) units",
    type=click.Path(exists=True, dir_okay=True, file_okay=False),
    prompt=True,
)
@click.option(
    "--out_dir",
    "-o",
    help="Path to the directory to save the deduped codes and durations.",
    type=click.Path(dir_okay=True, file_okay=False),
    prompt=True,
)
def prepare_codes_and_durations(in_dir, out_dir):
    """
    Prepares data for training by collapsing units and calculating durations.
    The output directory will contain a .npz file for each utterance.
    This .npz file has two keys.
    The "codes" entry contains the collapsed units.
    The "durations" entry contains the duration of each code.
    """
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    unit_paths = sorted(list(in_dir.rglob("*.npy")))
    click.echo(f"Found units for {len(unit_paths)} utterances.")

    for unit_path in tqdm(unit_paths):
        units = np.load(unit_path)
        codes, durations = dedupe(units)

        rel_path = unit_path.relative_to(in_dir)
        out_path = out_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, codes=codes, durations=durations)


if __name__ == "__main__":
    prepare_codes_and_durations()
