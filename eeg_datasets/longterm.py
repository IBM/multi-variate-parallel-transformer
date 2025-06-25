import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import h5py
import hdf5plugin
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset, RandomSampler
from . import (
    EEGBatch,
    EEGDataset,
    PatientData,
    consolidate,
    intervals_intersection,
)

from .custom_samplers import (
    ChunkDistributedSamplerWrapper,
    EEGPatientSampler,
    multipatient_collate,
)

_PATIENT_LIST = [
    "ID01",
    "ID02",
    "ID03",
    "ID04",
    "ID05",
    "ID06",
    "ID07",
    "ID08",
    "ID09",
    "ID10",
    "ID11",
    "ID12",
    "ID13",
    "ID14",
    "ID15",
    "ID16",
    "ID17",
    "ID18",
    "ID19",
    "ID20",
    "ID21",
    "ID22",
    "ID23",
    "ID24",
    "ID25",
    "ID26",
    "ID27",
    "ID28",
    "ID29",
    "ID30",
    "ID31",
    "ID32",
    "ID33",
    "ID34",
    "ID35",
    "ID36",
    "ID37",
    "ID38",
    "ID39",
    "ID40",
    "ID41",
    "ID42",
    "ID43",
    "ID44",
    "ID45",
    "ID46",
    "ID47",
    "ID48",
    "ID49",
    "ID50",
    "ID51",
    "ID52",
    "ID53",
    "ID54",
    "ID55",
    "ID56",
    "ID57",
    "ID58",
    "ID59",
    "ID60",
    "ID61",
    "ID62",
    "ID63",
    "ID64",
    "ID65",
    "ID66",
    "ID67",
    "ID68",
]
_CHANNELS = [
    88,
    66,
    64,
    32,
    128,
    32,
    75,
    61,
    48,
    32,
    32,
    56,
    64,
    24,
    98,
    34,
    60,
    42,
    29,
    88,
    66,
    64,
    32,
    32,
    128,
    34,
    32,
    75,
    61,
    48,
    32,
    32,
    104,
    56,
    64,
    24,
    98,
    34,
    60,
    42,
    33,
    63,
    126,
    60,
    47,
    86,
    32,
    57,
    60,
    64,
    89,
    69,
    22,
    54,
    24,
    62,
    40,
    92,
    54,
    74,
    76,
    60,
    64,
    56,
    49,
    39,
    63,
    32,
]


class LongTermEEGData(EEGDataset):
    _SAMPLING_RATE = 512

    def __init__(
        self,
        folder: Union[str, Path],
        batch_size: Optional[int] = 32,
        train_patients: Optional[List[Dict[str, List[int]] | str]] = _PATIENT_LIST,
        val_patients: Optional[List[Dict[str, List[int]] | str]] = [""],
        test_patients: Optional[List[Dict[str, List[int]] | str]] = [""],
        sampling_rate: int = _SAMPLING_RATE,
        segment_n: Optional[int] = 10,
        segment_size: Optional[int] = 10000,
        stride: Optional[int] = None,
        channels: Optional[int | List[str | int]] = None,
        patients_per_batch: Optional[int] = None,
        num_workers: Optional[int] = 0,
        limit_train_batches: Optional[int | float] = None,
        seizures: Optional[List[int]] = [],
        strategy: Optional[str] = None,
        balanced: bool = False,
        slowdown: bool = False,
    ) -> None:
        super().__init__()
        self.folder = folder
        self.batch_size = batch_size
        self.channels = self._parse_channels(channels)
        self.sampling_rate = sampling_rate
        self.num_workers = num_workers
        self.limit_train_batches = limit_train_batches
        self.strategy = strategy
        self.balanced = balanced
        self.train_patients = self._sanitize_patients(train_patients)
        self.val_patients = self._sanitize_patients(val_patients)
        self.test_patients = self._sanitize_patients(test_patients)
        self.seizures = seizures
        self.patients_per_batch = patients_per_batch
        self.segment_n = segment_n
        self.segment_size = segment_size
        self.segment_samples = int(self.segment_size / 1000.0 * self.sampling_rate)
        self.stride = stride if stride is not None else segment_size
        self.window_size = segment_size * segment_n
        self.slowdown = slowdown

    @staticmethod
    def _parse_channels(channels: Optional[int | List[str | int]]):
        if isinstance(channels, int):
            return channels
        if channels is None:
            return channels
        channels_parsed = []
        for desc in channels:
            if ":" in str(desc):
                beg, end = str(desc).split(":")
                channels_parsed.extend(list(range(int(beg), int(end))))
            else:
                channels_parsed.append(int(desc))
        return channels_parsed

    @staticmethod
    def _sanitize_patients(patients: List[str | Dict]) -> List[PatientData]:
        patients_sane = []
        if len(patients) == 0:
            return patients_sane
        if len(patients) == 1 and patients[0] == "":
            return patients_sane
        for pat in patients:
            if isinstance(pat, dict):
                if len(pat) != 1:
                    raise ValueError(
                        f"Patient dict can only contain one key, found {len(pat)}"
                    )
                else:
                    id, seizures = list(pat.items())[0]
                    if not isinstance(seizures, list):
                        raise TypeError(
                            f"Patient seizure description must be a list, found {type(seizures)}."
                        )
            elif isinstance(pat, str):
                if pat == "":
                    continue
                id, seizures = pat, []
            id_num = re.search(r"\d+", id)
            if id_num is None:
                raise ValueError(f"Patient ID not found.")
            id_num = id_num[0]
            id_clean = "ID" + id_num.zfill(2)
            if id_clean not in _PATIENT_LIST:
                raise ValueError(
                    f"Patient ID not found in this dataset, found {id_clean}."
                )
            channels = _CHANNELS[_PATIENT_LIST.index(id_clean)]
            patients_sane.append(
                PatientData(id=id_clean, seizures=seizures, channels=channels)
            )
        return patients_sane

    def setup(self, stage: str) -> None:
        if stage == "fit":
            min_train_channels = min([p.channels for p in self.train_patients])
            if isinstance(self.channels, int) or self.channels is None:
                max_train_channels = self.channels
            else:
                max_train_channels = max(self.channels)
            if (
                self.channels is None
                and len(self.train_patients) > 1
                and self.patients_per_batch is None
            ):
                raise ValueError(
                    "patients_per_batch cannot be None when training on multiple "
                    "patient with all channels"
                )
            if (
                isinstance(self.channels, list)
                and max_train_channels > min_train_channels
                and self.patients_per_batch is None
            ):
                raise ValueError(
                    "max training channels is {} but needs to be less than {}, "
                    "the minimum of all patient channels when patients_per_batch is None".format(
                        max_train_channels, min_train_channels
                    )
                )
            self.dataset_train = LongTermEEGDataset(
                folder=self.folder,
                n_patient=self.train_patients,
                window_n=self.segment_n,
                window=self.window_size,
                stride=self.stride,
                channels=self.channels,
                seizures=self.seizures,
                strategy=self.strategy,
                balanced=self.balanced,
                slowdown=self.slowdown,
                sampling_rate=self.sampling_rate,
            )
            if self.val_patients:
                min_val_channels = min([p.channels for p in self.val_patients])
                if isinstance(self.channels, int):
                    val_channels = list(range(0, min(self.channels, min_val_channels)))
                else:
                    val_channels = self.channels
                self.dataset_val = LongTermEEGDataset(
                    folder=self.folder,
                    n_patient=self.val_patients,
                    window_n=self.segment_n,
                    window=self.window_size,
                    stride=self.stride,
                    slowdown=False,
                    channels=val_channels,
                    seizures=self.seizures,
                )
        if stage == "validate":
            min_val_channels = min([p.channels for p in self.val_patients])
            if isinstance(self.channels, int):
                val_channels = list(range(0, min(self.channels, min_val_channels)))
            else:
                val_channels = self.channels
            self.dataset_val = LongTermEEGDataset(
                folder=self.folder,
                n_patient=self.val_patients,
                window_n=self.segment_n,
                window=self.window_size,
                stride=self.stride,
                slowdown=False,
                channels=val_channels,
                seizures=self.seizures,
                sampling_rate=self.sampling_rate,
            )
        if stage == "test" or stage == "predict":
            min_test_channels = min([p.channels for p in self.test_patients])
            if isinstance(self.channels, int):
                test_channels = list(range(0, min(self.channels, min_test_channels)))
            else:
                test_channels = self.channels
            self.dataset_test = []
            labels_path = None
            if (
                len(self.trainer.loggers) > 0
                and self.trainer.loggers[0].log_dir is not None
            ):
                labels_path = os.path.join(self.trainer.loggers[0].log_dir, "labels.h5")
            start_idx = 0
            if labels_path is not None and os.path.exists(labels_path):
                try:
                    with h5py.File(labels_path) as h5_file:
                        start_idx = h5_file.attrs["last_id"]
                except OSError:
                    os.remove(labels_path)
            for patient in self.test_patients:
                dataset_test = LongTermEEGDataset(
                    folder=self.folder,
                    n_patient=[patient],
                    window_n=self.segment_n,
                    window=self.window_size,
                    stride=self.stride,
                    slowdown=False,
                    channels=test_channels,
                    seizures=self.seizures,
                    start_idx=start_idx,
                    sampling_rate=self.sampling_rate,
                )
                self.dataset_test.append(dataset_test)

    def train_dataloader(self) -> DataLoader:
        num_samples = None
        if self.limit_train_batches is not None:
            if isinstance(self.limit_train_batches, int):
                num_samples = self.batch_size * self.limit_train_batches
            elif isinstance(self.limit_train_batches, float):
                num_samples = int(len(self.dataset_train) * self.limit_train_batches)

        if self.patients_per_batch is not None:
            shuffle = False
            sampler = EEGPatientSampler(
                self.dataset_train,
                batch_size=self.batch_size,
                patients_per_batch=self.patients_per_batch,
                shuffle=True,
                num_samples=num_samples,
                balanced=self.balanced,
            )
            distributed_sampler_kwargs = self.trainer.distributed_sampler_kwargs
            if distributed_sampler_kwargs is not None:
                distributed_sampler_kwargs.setdefault(
                    "seed", int(os.getenv("PL_GLOBAL_SEED", 0))
                )
                distributed_sampler_kwargs.setdefault("shuffle", False)
                sampler = ChunkDistributedSamplerWrapper(
                    sampler=sampler, **distributed_sampler_kwargs
                )
            collate_fn = multipatient_collate(self.patients_per_batch)
        else:
            shuffle = False
            sampler = RandomSampler(
                self.dataset_train, generator=None, num_samples=num_samples
            )
            collate_fn = None
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_patients:
            sampler = EEGPatientSampler(
                self.dataset_val,
                batch_size=self.batch_size,
                patients_per_batch=self.patients_per_batch,
                shuffle=False,
                num_samples=None,
            )
            return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                sampler=sampler,
                shuffle=False,
                collate_fn=multipatient_collate(self.patients_per_batch),
            )
        else:
            return None

    def test_dataloader(self) -> DataLoader:
        test_dataloader = []
        for dataset in self.dataset_test:
            sampler = EEGPatientSampler(
                dataset,
                batch_size=self.batch_size,
                patients_per_batch=self.patients_per_batch,
                shuffle=False,
                num_samples=None,
            )
            test_dataloader.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=False,
                    collate_fn=multipatient_collate(self.patients_per_batch),
                )
            )
        return test_dataloader

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class LongTermEEGDataset(Dataset[EEGBatch]):
    _URL = "https://mb-neuro.medical-blocks.ch/public_access/databases/ieeg/swec_ieeg"
    _SAMPLING_RATE = 512

    def __init__(
        self,
        folder: Union[str, Path],
        n_patient: List[PatientData],
        sampling_rate: int = _SAMPLING_RATE,
        window_n: int = 1,
        window: int = 80000,
        stride: Optional[int] = None,
        slowdown: bool = True,
        seizures: List[int] = [],
        channels: Optional[int | List[int]] = None,
        strategy: Optional[str] = None,
        start_idx: int = 0,
        balanced: bool = False,
    ) -> None:
        self.window_n = window_n
        self.window = window / 1000.0
        self.srate = sampling_rate
        self.slowdown = slowdown
        self.balanced = balanced
        self.start_idx = start_idx
        self.seizures = seizures
        self.channels = channels
        self.stride = stride / 1000.0
        self.slowdown_stride = self.stride
        if self.slowdown:
            self.slowdown_stride = self.stride / self.window_n
        self.n_patient = n_patient
        self.patient_ids = [p.id for p in self.n_patient]
        self.window_samples = int(self.window * self.srate)
        self.stride_samples = int(self.stride * self.srate)
        self.slowdown_stride_samples = int(self.slowdown_stride * self.srate)
        self.slowdown_interval = 0
        if self.slowdown:
            self.slowdown_interval = 2 * self.window
        self.folder = folder
        self.strategy = strategy
        self.trainable_indices: np._ArrayLikeInt = np.empty((0,), dtype=int)
        self.patients_length: List[int] = []
        self.samples_boundaries: List[List[int]] = []
        self.labels_boundaries: List[List[int]] = []
        self.patients: np._ArrayLikeInt
        self.file_paths: List[List[str]] = []
        self.seizure_boundaries: List[List[float]] = []
        self.sampling_rates: List[float] = []
        self.srate_conversion: List[float] = []
        self._patient_files: List[h5py.File] = []
        self.total_length: int
        self.patients_seizures_indices: List[List[int]] = []
        self.patients_slowdown_indices: List[List[int]] = []
        self.delta_powers = []
        self.delta_times = []
        self.load_info()
        self.map_items()
        if self.strategy == "delta":
            self.get_trainable_delta()
        self._patient_files = []

    @property
    def patient_files(self):
        if len(self._patient_files) == 0:
            self._patient_files = [
                h5py.File(
                    os.path.join(
                        os.path.join(self.folder, patient),
                        f"{patient}_total.h5",
                    ),
                    rdcc_nbytes=1000 * 1024 * 1024,
                    rdcc_nslots=500 * 100,
                )
                for patient in self.patient_ids
            ]
        return self._patient_files

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, int, int, float, float, int]:
        if len(self.trainable_indices) > 0:
            n = self.trainable_indices[n]
        patient_id = self.patients[n]
        patient = self.patient_ids.index(patient_id)
        if patient == 0:
            patient_length = 0
        else:
            patient_length = self.patients_length[patient - 1]
        sample_n = n - patient_length
        s_multiplier = self.srate_conversion[patient]
        patient_dataset = self.patient_files[patient]["data/ieeg"]
        patient_length_sec = patient_dataset.shape[1] / (self.srate * s_multiplier)
        begin_sec = self._find_second_from_index(
            sample_n, self.patients_slowdown_indices[patient]
        )
        sample_begin = int(begin_sec * self.srate * s_multiplier)
        sample_length_sec = patient_length_sec - begin_sec
        sample = np.zeros(
            (patient_dataset.shape[0], int(self.window_samples * s_multiplier)),
            dtype=patient_dataset.dtype,
        )
        if sample_length_sec < self.window:
            raise Exception("Sample shorter than needed.")
        else:
            sample_length = int(self.window_samples * s_multiplier)
        patient_dataset.read_direct(
            sample,
            np.s_[
                :,
                sample_begin : sample_begin + sample_length,
            ],
            np.s_[:, :sample_length],
        )
        sample_torch = torch.from_numpy(sample)
        downsampled_seizure = interpolate(
            sample_torch.unsqueeze(0), size=self.window_samples, mode="linear"
        )
        sample = downsampled_seizure.squeeze(0)
        if isinstance(self.channels, int):
            num_channels = min(sample.shape[0], self.channels)
            selected_channels = torch.randperm(sample.shape[0])[:num_channels].sort()[0]
        else:
            selected_channels = self.channels
        if selected_channels is not None:
            sample = sample[selected_channels]
        sample = sample.unfold(
            1,
            self.window_samples // self.window_n,
            self.window_samples // self.window_n,
        )
        seizure_boundaries = self.seizure_boundaries[patient]
        window_pos_begin = begin_sec + self.window - self.window // self.window_n
        window_pos_end = begin_sec + self.window
        before_offsets = window_pos_begin <= seizure_boundaries["offsets"]
        after_onsets = window_pos_end >= seizure_boundaries["onsets"]
        seiz_selected = np.logical_and(before_offsets, after_onsets).flatten()
        label = np.any(seiz_selected)
        label_t = Tensor([label])
        seizure_begins = seizure_boundaries["onsets"].flatten()[
            (np.abs(seizure_boundaries["onsets"].flatten() - window_pos_begin)).argmin()
        ]
        seizure_ends = seizure_boundaries["offsets"].flatten()[
            (
                np.abs(seizure_boundaries["offsets"].flatten() - window_pos_begin)
            ).argmin()
        ]
        done = 0
        if n == self.patients_length[patient] - 1:
            done = 1
        return EEGBatch(
            sample.transpose(0, 1).contiguous(),
            label_t,
            n,
            sample_n,
            patient,
            seizure_begins,
            seizure_ends,
            done,
            self.srate,
        )

    def load_info(self) -> None:
        for patient in self.patient_ids:
            patient_path = os.path.join(self.folder, patient)
            patient_file = h5py.File(os.path.join(patient_path, f"{patient}_total.h5"))
            self._patient_files.append(patient_file)
            seizure_boundaries = patient_file["data/seizures"][:]
            self.seizure_boundaries.append(seizure_boundaries)
            self.sampling_rates.append(float(patient_file.attrs["sampling_rate"]))
            self.srate_conversion.append(
                float(patient_file.attrs["sampling_rate"]) / self.srate
            )
            if self.strategy == "delta":
                self.delta_powers.append(patient_file["data/delta_power"][:])
                self.delta_times.append(patient_file["data/delta_windows"][:])

    def map_items(self) -> None:
        patients_samples_all = []
        patients_seizures_indices = []
        patients_slowdown_indices = []
        for i, patient in enumerate(self.patient_files):
            patient_seizures_idx = []
            patient_slowdown_idx = []
            patient_length = patient["data/ieeg"].shape[1] / (
                self.srate * self.srate_conversion[i]
            )
            seizure_boundaries = self.seizure_boundaries[i]
            sd_end = -1
            for sz in range(0, seizure_boundaries.shape[0]):
                sd_start = max(
                    (
                        self._find_index_from_second(
                            seizure_boundaries["onsets"][sz]
                            - self.slowdown_interval
                            - self.window,
                            patient_slowdown_idx,
                            last=True,
                        )
                        + 1
                    ),
                    0,
                )

                if sd_start > sd_end:
                    patient_slowdown_idx.append(sd_start)
                else:
                    old_end = patient_slowdown_idx.pop()
                    new_end = sd_start = (old_end - sd_start) // 2 + sd_start
                    patient_slowdown_idx.append(new_end)
                    patient_slowdown_idx.append(sd_start)

                sz_start = max(
                    (
                        self._find_index_from_second(
                            seizure_boundaries["onsets"][sz] - self.window,
                            patient_slowdown_idx,
                            last=True,
                        )
                        + 1
                    ),
                    0,
                )
                patient_seizures_idx.append(sz_start)
                sz_end = self._find_index_from_second(
                    seizure_boundaries["offsets"][sz], patient_slowdown_idx, last=True
                )
                sd_end = self._find_index_from_second(
                    seizure_boundaries["offsets"][sz] + self.slowdown_interval,
                    patient_slowdown_idx,
                    last=True,
                )
                last_index = self._find_index_from_second(
                    patient_length - self.window,
                    patient_slowdown_idx,
                    last=True,
                )
                sd_end = min(sd_end, last_index)
                sz_end = min(sz_end, sd_end)
                patient_seizures_idx.append(sz_end)
                patient_slowdown_idx.append(sd_end)
            patients_seizures_indices.append(patient_seizures_idx)
            patients_slowdown_indices.append(patient_slowdown_idx)
            slow_samples = (
                np.array(patient_slowdown_idx[1::2])
                - np.array(patient_slowdown_idx[0::2])
            ).sum()
            slow_length = slow_samples * self.slowdown_stride + self.window
            nonseizure_length = patient_length - slow_length
            patient_samples_nonseizure = (nonseizure_length - self.window) // (
                self.stride
            ) + 1
            patient_samples_seizure = slow_samples
            patients_samples_all.append(
                int(patient_samples_nonseizure + patient_samples_seizure)
            )
        self.patients_length = np.cumsum(patients_samples_all)
        self.patients = np.repeat(self.patient_ids, patients_samples_all)
        self.patients_seizures_indices = patients_seizures_indices
        self.patients_slowdown_indices = patients_slowdown_indices
        patients_usable_boundaries = [[] for _ in range(0, len(self.patient_files))]
        patients_boundaries_labels = [[] for _ in range(0, len(self.patient_files))]
        for i in range(len(patients_slowdown_indices)):
            n_seizures = len(patients_slowdown_indices[i]) // 2
            patient_seizures = self.n_patient[i].seizures
            if i == 0:
                patient_begin = 0
            else:
                patient_begin = self.patients_length[i - 1]
            if len(patient_seizures) == 0:
                beg = max(patient_begin, self.start_idx)
                end = max(self.patients_length[i], self.start_idx)
                if self.balanced:
                    inside_seizure = False
                    for s in self.patients_seizures_indices[i]:
                        s += patient_begin
                        patients_usable_boundaries[i].append([beg, s])
                        beg = s
                        patients_boundaries_labels[i].append(int(inside_seizure))
                        inside_seizure ^= True
                    patients_usable_boundaries[i].append([beg, end])
                    patients_boundaries_labels[i].append(int(inside_seizure))
                else:
                    patients_usable_boundaries[i].append([beg, end])
            for seiz in patient_seizures:
                if seiz == 1:
                    end_previous = float("-inf")
                else:
                    end_previous = self._find_index_from_second(
                        seizure_boundaries["offsets"][seiz - 2],
                        patients_slowdown_indices[i],
                        last=True,
                    )

                if seiz == patient_seizures[-1]:
                    begin_next = float("inf")
                else:
                    begin_next = self._find_index_from_second(
                        seizure_boundaries["onsets"][seiz],
                        patients_slowdown_indices[i],
                        last=True,
                    )

                beg = self._find_index_from_second(
                    seizure_boundaries["onsets"][seiz - 1],
                    patients_slowdown_indices[i],
                    last=True,
                )

                beg_window = self._find_index_from_second(
                    seizure_boundaries["onsets"][seiz - 1] - 30 * 60,
                    patients_slowdown_indices[i],
                    last=True,
                )

                beg = max(0, max(beg_window, beg - (beg - end_previous) // 2))

                end = self._find_index_from_second(
                    seizure_boundaries["offsets"][seiz - 1],
                    patients_slowdown_indices[i],
                    last=True,
                )

                end_window = self._find_index_from_second(
                    seizure_boundaries["offsets"][seiz - 1] + 30 * 60,
                    patients_slowdown_indices[i],
                    last=True,
                )

                end = min(
                    self.patients_length[i],
                    min(end_window, end + (begin_next - end) // 2),
                )

                if self.balanced:
                    inside_seizure = False
                    for s in self.patients_seizures_indices[i]:
                        s += patient_begin
                        if s in range(beg, end):
                            patients_usable_boundaries[i].append([beg, s])
                            beg = s
                            patients_boundaries_labels[i].append(int(inside_seizure))
                            inside_seizure ^= True
                    patients_usable_boundaries[i].append([beg, end])
                    patients_boundaries_labels[i].append(int(inside_seizure))
                else:
                    patients_usable_boundaries[i].append(
                        [
                            beg,
                            end,
                        ]
                    )
        self.samples_boundaries = [
            sorted(consolidate(i)) for i in patients_usable_boundaries
        ]
        self.labels_boundaries = patients_boundaries_labels

    def _find_index_from_second(
        self, sec: float, slowdown_indices: npt.ArrayLike, last: bool = False
    ) -> int:
        after_slowdown_begin = (
            np.flatnonzero(
                sec
                >= np.asarray(
                    [
                        self._find_second_from_index(s, slowdown_indices)
                        for s in slowdown_indices[::2]
                    ]
                )
            ).size
            - 1
        )
        after_slowdown_end = (
            np.flatnonzero(
                sec
                > np.asarray(
                    [
                        self._find_second_from_index(s, slowdown_indices)
                        for s in slowdown_indices[1::2]
                    ]
                )
            ).size
            - 1
        )
        if after_slowdown_begin > after_slowdown_end:
            position_in_slowdown = 2 * after_slowdown_begin
            sec_after = sec - self._find_second_from_index(
                slowdown_indices[position_in_slowdown], slowdown_indices
            )
            if last:
                samples_after = sec_after // self.slowdown_stride
            else:
                samples_after = (
                    max((sec_after - self.window), -self.slowdown_stride)
                    // self.slowdown_stride
                    + 1
                )
        else:
            position_in_slowdown = 2 * after_slowdown_end + 1
            if after_slowdown_begin < 0:
                sec_after = sec
            else:
                sec_after = sec - self._find_second_from_index(
                    slowdown_indices[position_in_slowdown], slowdown_indices
                )
            if last:
                samples_after = sec_after // self.stride
            else:
                samples_after = (
                    max((sec_after - self.window), -self.stride) // self.stride + 1
                )
        if position_in_slowdown < 0:
            slow_samples_before = 0
        else:
            slow_samples_before = slowdown_indices[position_in_slowdown]
        begin_index = slow_samples_before + samples_after
        return int(begin_index)

    def _find_second_from_index(
        self, idx: int, slowdown_indices: npt.ArrayLike
    ) -> float:
        position_in_slowdown = np.searchsorted(slowdown_indices, idx)
        new_slowdown_indices = slowdown_indices[:position_in_slowdown].copy()
        new_slowdown_indices.append(idx)
        slow_samples_before = (
            np.asarray(new_slowdown_indices[1 : position_in_slowdown + 1 : 2])
            - np.asarray(new_slowdown_indices[0:position_in_slowdown:2])
        ).sum()
        begin_sec = (
            slow_samples_before * self.slowdown_stride
            + (idx - slow_samples_before) * self.stride
        )
        return begin_sec

    def get_trainable_delta(self) -> None:
        trainable_indices = []
        for i, _ in enumerate(self.patient_files):
            patient_trainable_indices = []
            for s_bound in zip(
                self.patients_slowdown_indices[i][::2],
                self.patients_slowdown_indices[i][1::2],
            ):
                patient_trainable_indices.append([s_bound[0], s_bound[1]])
            if i - 1 < 0:
                patient_begin = 0
            else:
                patient_begin = self.patients_length[i - 1]

            delta_power = self.delta_powers[i]
            delta_times = self.delta_times[i]

            bin_edges = np.histogram_bin_edges(delta_power, bins=5)
            for bound in zip(bin_edges, bin_edges[1:]):
                bin_power = np.logical_and(
                    delta_power >= bound[0], delta_power < bound[1]
                )
                bin_bound = np.diff(bin_power).nonzero()[0]
                if bin_power[0]:
                    bin_bound = np.insert(bin_bound, 0, 0)
                if bin_power[-1]:
                    bin_bound = np.append(bin_bound, bin_power.size - 1)

                bin_bound_sec = delta_times[bin_bound] // (
                    self.srate * self.srate_conversion[i]
                )
                bin_lengths = bin_bound_sec[1::2] - bin_bound_sec[::2]
                allowed_lengths = np.flatnonzero(bin_lengths >= 20 * 60)
                if len(allowed_lengths) > 0:
                    selected_length = np.random.choice(allowed_lengths)
                    random_start = np.random.randint(
                        bin_lengths[selected_length] - 20 * 60 + 1
                    )
                else:
                    selected_length = np.argmax(bin_lengths)
                    random_start = 0

                beg_idx = patient_begin + self._find_index_from_second(
                    bin_bound_sec[selected_length * 2] + random_start,
                    self.patients_slowdown_indices[i],
                )

                end_idx = patient_begin + self._find_index_from_second(
                    bin_bound_sec[selected_length * 2]
                    + random_start
                    + min(20 * 60, selected_length),
                    self.patients_slowdown_indices[i],
                )

                patient_trainable_indices.append([beg_idx, end_idx])
            union_indices = sorted(consolidate(patient_trainable_indices))
            intersection_indices = intervals_intersection(
                union_indices, self.samples_boundaries[i]
            )
            trainable_indices.append(intersection_indices)

        self.samples_boundaries = trainable_indices

    def __len__(self) -> int:
        total_len = sum(
            [beg[-1] - beg[0] for group in self.samples_boundaries for beg in group]
        )
        if len(self.trainable_indices) > 0:
            total_len = len(self.trainable_indices)
        return total_len
