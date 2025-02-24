import itertools
import math
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Sized, Union
import torch
from lightning_fabric.utilities.distributed import _DatasetSamplerWrapper
from torch.utils.data import Dataset, DistributedSampler, Sampler, SubsetRandomSampler
from torch.utils.data._utils.collate import (
    collate,
    collate_tensor_fn,
    default_collate_fn_map,
)
from torch.utils.data.distributed import T_co


def multipatient_collate(patients_per_batch):
    def multi_collate_tensor_fn(batch, *, collate_fn_map=None):
        batch_len = len(batch) // patients_per_batch
        batches = [None] * patients_per_batch
        for p in range(patients_per_batch):
            batches[p] = collate_tensor_fn(batch[p * batch_len : (p + 1) * batch_len])
        return batches

    multichannel_collate_fn_map = default_collate_fn_map.copy()
    multichannel_collate_fn_map[torch.Tensor] = multi_collate_tensor_fn

    def collate_fn(batch):
        return collate(batch, collate_fn_map=multichannel_collate_fn_map)

    return collate_fn


class ChunkDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        assert len(indices) == self.num_samples
        return iter(indices)


class ChunkDistributedSamplerWrapper(ChunkDistributedSampler):
    def __init__(
        self, sampler: Union[Sampler, Iterable], *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

    def __iter__(self) -> Iterator:
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())


class SubsetSampler(Sampler[int]):
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        yield from self.indices

    def __len__(self) -> int:
        return len(self.indices)


class WeightedMultiSubsetRandomSampler(Sampler[int]):
    indices: Sequence[int]

    def __init__(
        self, indices: Sequence[int], weights: Sequence[float], generator=None
    ) -> None:
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        self.indices = indices
        self.weights = weights_tensor
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.multinomial(
            self.weights,
            sum([len(subset) for subset in self.indices]),
            True,
            generator=self.generator,
        ):
            idx = torch.randint(len(self.indices[i]), (1,), generator=self.generator)
            yield self.indices[i][idx]

    def __len__(self) -> int:
        return sum([len(subset) for subset in self.indices])


class EEGPatientSampler(Sampler[List[int]]):
    def __init__(
        self,
        data_source: Sized,
        replacement: bool = False,
        num_samples: Optional[int] = None,
        batch_size: Optional[int] = None,
        patients_per_batch: Optional[int] = None,
        shuffle: bool = False,
        drop_last: bool = False,
        balanced: bool = False,
        generator=None,
    ) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.shuffle = shuffle
        self.balanced = balanced
        self.drop_last = drop_last
        if batch_size is None:
            batch_size = 0
        if patients_per_batch is None:
            patients_per_batch = 0
        self.batch_size = batch_size
        self.patients_per_batch = patients_per_batch
        if self.batch_size < self.patients_per_batch:
            raise ValueError(
                "batch_size should be greater than patients_per_batch, but "
                "found batch_size: {} and patients_per_batch: {}".format(
                    self.batch_size, self.patients_per_batch
                )
            )
        elif self.batch_size > 0 and self.patients_per_batch > 0:
            if self.batch_size % self.patients_per_batch:
                raise ValueError(
                    "batch_size should be divisible by patients_per_batch, but "
                    "found batch_size: {} and patients_per_batch: {}".format(
                        self.batch_size, self.patients_per_batch
                    )
                )
        if self.patients_per_batch > 0:
            self.items_per_sampler = self.batch_size // self.patients_per_batch
        else:
            self.items_per_sampler = self.batch_size
        patients_length = data_source.samples_boundaries
        patients_labels = data_source.labels_boundaries
        samplers = []
        for i, patient_length in enumerate(patients_length):
            if self.shuffle:
                if self.balanced:
                    patient_labels = torch.tensor(patients_labels[i], dtype=bool)
                    patient_indices = torch.tensor(patient_length, dtype=int)
                    seizure_indices = patient_indices[patient_labels == 1]
                    nonseizure_indices = patient_indices[patient_labels == 0]
                    seizure_indices = itertools.chain.from_iterable(
                        [range(beg, end) for (beg, end) in seizure_indices]
                    )
                    nonseizure_indices = itertools.chain.from_iterable(
                        [range(beg, end) for (beg, end) in nonseizure_indices]
                    )
                    patient_samplers = [
                        WeightedMultiSubsetRandomSampler(
                            [list(seizure_indices), list(nonseizure_indices)],
                            [1.0, 1.0],
                            generator=self.generator,
                        )
                    ]
                else:
                    patient_samplers = [
                        SubsetRandomSampler(
                            list(range(beg, end)), generator=self.generator
                        )
                        for (beg, end) in patient_length
                    ]
                    # all_indices = itertools.chain.from_iterable(
                    #     [range(beg, end) for (beg, end) in patient_length]
                    # )
                    # patient_sampler = SubsetRandomSampler(
                    #     list(all_indices), generator=self.generator
                    # )
            else:
                patient_samplers = [
                    SubsetSampler(list(range(beg, end)))
                    for (beg, end) in patient_length
                ]
                # all_indices = itertools.chain.from_iterable(
                #     [range(beg, end) for (beg, end) in patient_length]
                # )
                # patient_sampler = SubsetSampler(list(all_indices))
            samplers.extend(patient_samplers)
        self.samplers = samplers

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            if self.drop_last:
                return len(self.data_source) // self.batch_size * self.batch_size
            else:
                return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[List[int]]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        samplers_iters = [iter(sampler) for sampler in self.samplers]
        for _ in range(math.ceil(self.num_samples / self.items_per_sampler)):
            sampler_order = torch.randperm(len(self.samplers), generator=generator)
            if self.drop_last:
                for sampler_id in sampler_order:
                    try:
                        batch = [
                            next(samplers_iters[sampler_id])
                            for _ in range(self.items_per_sampler)
                        ]
                        yield from batch
                    except StopIteration:
                        break
            else:
                batch = [0] * self.batch_size
                idx_in_batch = 0
                for sampler_id in sampler_order:
                    idx_in_sampler = 0
                    for idx in samplers_iters[sampler_id]:
                        batch[idx_in_batch] = idx
                        idx_in_batch += 1
                        idx_in_sampler += 1
                        if idx_in_batch == self.batch_size:
                            yield from batch
                            idx_in_batch = 0
                            batch = [0] * self.batch_size
                        if idx_in_sampler == self.items_per_sampler:
                            break
                if idx_in_batch > 0:
                    yield from batch[:idx_in_batch]

    def __len__(self) -> int:
        return self.num_samples
