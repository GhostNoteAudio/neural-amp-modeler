from tap import Tap
import re
import tkinter as tk
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Dict, Optional, Sequence
import os

import yaml

from nam.train.core import Architecture

try:  # 3rd-party and 1st-party imports
    import torch

    from nam import __version__
    from nam.train import core
    from nam.models.metadata import GearType, UserMetadata, ToneType

    # Ok private access here--this is technically allowed access
    from nam.train._names import LATEST_VERSION

    _install_is_valid = True
    _HAVE_ACCELERATOR = torch.cuda.is_available() or torch.backends.mps.is_available()
except ImportError:
    _install_is_valid = False
    _HAVE_ACCELERATOR = False

if _HAVE_ACCELERATOR:
    _DEFAULT_NUM_EPOCHS = 100
    _DEFAULT_BATCH_SIZE = 16
    _DEFAULT_LR_DECAY = 0.007
else:
    _DEFAULT_NUM_EPOCHS = 20
    _DEFAULT_BATCH_SIZE = 1
    _DEFAULT_LR_DECAY = 0.05


def get_input_files(input_dir: str) -> list[str]:
    files = [x for x in os.listdir(input_dir) if x.endswith(".wav")]
    return [os.path.join(input_dir, file) for file in files]


def get_metadata(input_dir: str) -> UserMetadata | None:
    metadata_path = os.path.join(input_dir, "metadata.yaml")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            y = list(yaml.safe_load_all(f))[0]
        metadata = UserMetadata(
            name=y['name'],
            modeled_by=y['modeled_by'],
            gear_type=GearType(y['gear_type']),
            gear_make=y['gear_make'],
            gear_model=y['gear_model'],
            tone_type=ToneType(y['tone_type']))
        return metadata
    return UserMetadata()


def train(input_file: str, input_dir: str, output_dir: str, move_input_file: bool, epochs: int, fit_cab: bool):
    # Advanced options:
    num_epochs: int = epochs
    architecture: str = Architecture.STANDARD.value
    delay = None
    file_list = get_input_files(input_dir)
    if input_file in file_list:
        file_list.remove(input_file) # Remove if training file was in input directory
    metadata = get_metadata(input_dir)

    # Advanced-er options
    # If you're poking around looking for these, then maybe it's time to learn to
    # use the command-line scripts ;)
    lr = 0.004
    lr_decay = _DEFAULT_LR_DECAY
    batch_size = _DEFAULT_BATCH_SIZE
    seed = 0

    # Run it
    for file in file_list:
        print("Now training {}".format(file))
        basename = re.sub(r"\.wav$", "", file.split("/")[-1])

        trained_model = core.train(
            input_file,
            file,
            output_dir,
            epochs=num_epochs,
            delay=delay,
            architecture=architecture,
            batch_size=batch_size,
            lr=lr,
            lr_decay=lr_decay,
            seed=seed,
            silent=True,
            save_plot=True,
            modelname=basename,
            ignore_checks=True,
            local=True,
            fit_cab=fit_cab,
        )
        if trained_model is None:
            print("Model training failed! Skip exporting...")
            continue
        print("Model training complete!")
        print("Exporting...")
        outdir = output_dir
        print(f"Exporting trained model to {outdir}...")
        trained_model.net.export(
            outdir,
            basename=basename,
            user_metadata=metadata,
        )
        print("Done!")


class TrainingArgs(Tap):
    input_file: str
    input_dir: str
    output_dir: str
    move_input_file: bool = True
    epochs: int = 20
    fit_cab: bool = False


if __name__ == '__main__':
    args = TrainingArgs(underscores_to_dashes=True).parse_args()
    train(args.input_file, args.input_dir, args.output_dir, args.move_input_file, args.epochs, args.fit_cab)
