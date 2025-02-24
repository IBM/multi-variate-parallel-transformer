import os

import torch
from pytorch_lightning.cli import LightningCLI

from eeg_datasets import EEGDataset
from models import BrainModel


class BrainCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.init_args.segment_samples",
            "model.init_args.encoder.init_args.size_input",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "trainer.limit_train_batches",
            "data.init_args.limit_train_batches",
        )

    def instantiate_classes(self) -> None:
        super().instantiate_classes()


def cli_main():
    cli = BrainCLI(
        BrainModel,
        EEGDataset,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        save_config_kwargs={
            "code_path": "/mnt/eegmann/generative-neural-inference-engine"
        },
    )


def set_env() -> None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cli_main()
