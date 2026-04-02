from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import numpy as np


PairTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass
class BatchedDataset:
    features: np.ndarray
    targets: np.ndarray
    batch_size: int
    drop_last: bool = False
    shuffle_each_epoch: bool = False
    seed: int | None = None

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = np.arange(len(self.features))
        if self.shuffle_each_epoch:
            generator = np.random.default_rng(self.seed)
            generator.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            stop = start + self.batch_size
            batch_indices = indices[start:stop]
            if self.drop_last and len(batch_indices) < self.batch_size:
                break
            yield self.features[batch_indices], self.targets[batch_indices]

    def __len__(self) -> int:
        batches, remainder = divmod(len(self.features), self.batch_size)
        if remainder and not self.drop_last:
            batches += 1
        return batches


class Dataset:
    def __init__(self, features: Sequence, targets: Sequence) -> None:
        self.features = np.asarray(features, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        if len(self.features) != len(self.targets):
            raise ValueError("Features and targets must have the same length.")

    @classmethod
    def from_arrays(cls, features: Sequence, targets: Sequence) -> "Dataset":
        return cls(features, targets)

    def __len__(self) -> int:
        return len(self.features)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for x_sample, y_sample in zip(self.features, self.targets):
            yield x_sample, y_sample

    def map(self, transform: PairTransform) -> "Dataset":
        mapped_x: list[np.ndarray] = []
        mapped_y: list[np.ndarray] = []
        for x_sample, y_sample in self:
            new_x, new_y = transform(x_sample, y_sample)
            mapped_x.append(np.asarray(new_x))
            mapped_y.append(np.asarray(new_y))
        return Dataset(np.stack(mapped_x), np.stack(mapped_y))

    def shuffle(self, seed: int | None = None) -> "Dataset":
        generator = np.random.default_rng(seed)
        indices = np.arange(len(self))
        generator.shuffle(indices)
        return Dataset(self.features[indices], self.targets[indices])

    def batch(
        self,
        batch_size: int,
        drop_last: bool = False,
        shuffle_each_epoch: bool = False,
        seed: int | None = None,
    ) -> BatchedDataset:
        return BatchedDataset(
            features=self.features,
            targets=self.targets,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle_each_epoch=shuffle_each_epoch,
            seed=seed,
        )

    def split(
        self,
        validation_ratio: float = 0.2,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> tuple["Dataset", "Dataset"]:
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("validation_ratio must be between 0 and 1.")
        dataset = self.shuffle(seed=seed) if shuffle else self
        split_index = int(len(dataset) * (1.0 - validation_ratio))
        train_x = dataset.features[:split_index]
        train_y = dataset.targets[:split_index]
        valid_x = dataset.features[split_index:]
        valid_y = dataset.targets[split_index:]
        return Dataset(train_x, train_y), Dataset(valid_x, valid_y)

    def take(self, amount: int) -> "Dataset":
        return Dataset(self.features[:amount], self.targets[:amount])

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.features.copy(), self.targets.copy()
