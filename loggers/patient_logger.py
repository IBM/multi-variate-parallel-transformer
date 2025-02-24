import os
from typing import Optional, Union

import h5py
import numpy as np
import torch
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class PatientLogger:
    def __init__(
        self,
        save_dir: _PATH,
        name: str = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        filename: str = "labels",
        flush_logs_every_n_steps: int = 100,
    ) -> None:
        self._save_dir = os.fspath(save_dir)
        self.name = name
        self.version = version
        self.filename = filename
        self.flush_logs_every_n_steps = flush_logs_every_n_steps
        self.metrics = None
        self.last = None

    @property
    def root_dir(self) -> str:
        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:
        version = (
            self.version if isinstance(self.version, str) else f"version_{self.version}"
        )
        _log_dir = os.path.join(self.root_dir, version)
        try:
            os.mkdir(_log_dir)
        except FileExistsError:
            return _log_dir
        return _log_dir

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @rank_zero_only
    def log(self, metric: torch.Tensor, last: int) -> None:
        if self.metrics is None:
            self.metrics = metric.cpu().float().numpy()
        else:
            self.metrics = np.concatenate([self.metrics, metric.cpu().float().numpy()])
        self.last = last.item()
        if self.metrics.shape[0] >= self.flush_logs_every_n_steps:
            self.save()

    @rank_zero_only
    def save(self):
        if self.metrics is None:
            return
        with h5py.File(
            os.path.join(self.log_dir, f"{self.filename}.h5"), "a"
        ) as output_file:
            predictions = output_file.require_dataset(
                "labels",
                shape=(0, 2),
                dtype=self.metrics.dtype,
                maxshape=(None, 2),
                chunks=(100, 2),
            )
            output_file.attrs["last_id"] = self.last
            predictions.resize(predictions.shape[0] + self.metrics.shape[0], axis=0)
            predictions[-self.metrics.shape[0] :] = self.metrics
        self.metrics = None


class PredictiveLogger(PatientLogger):
    def __init__(
        self,
        save_dir: _PATH,
        name: str = "lightning_logs",
        version: int | str | None = None,
        filename: str = "labels",
        flush_logs_every_n_steps: int = 100,
    ) -> None:
        super().__init__(save_dir, name, version, filename, flush_logs_every_n_steps)

    @rank_zero_only
    def save(self):
        if self.metrics is None:
            return
        with h5py.File(
            os.path.join(self.log_dir, f"{self.filename}.h5"), "a"
        ) as output_file:
            predictions = output_file.require_dataset(
                "cos",
                shape=(0, self.metrics.shape[1]),
                dtype=self.metrics.dtype,
                maxshape=(None, self.metrics.shape[1]),
                chunks=(100, self.metrics.shape[1]),
            )
            output_file.attrs["last_id"] = self.last
            predictions.resize(predictions.shape[0] + self.metrics.shape[0], axis=0)
            predictions[-self.metrics.shape[0] :] = self.metrics
        self.metrics = None
