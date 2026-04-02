from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miniflow import Dataset, Dense, Dropout, LeakyReLU, Sequential, Sigmoid


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def build_cat_model(
    input_dim: int,
    hidden_dims: tuple[int, int] = (48, 16),
    dropout_rates: tuple[float, float] = (0.3, 0.1),
) -> Sequential:
    first_hidden, second_hidden = hidden_dims
    first_dropout, second_dropout = dropout_rates

    layers = [
        Dense(first_hidden, input_dim=input_dim, initializer="he"),
        LeakyReLU(negative_slope=0.05),
    ]
    if first_dropout > 0.0:
        layers.append(Dropout(first_dropout))

    layers.extend(
        [
            Dense(second_hidden, initializer="he"),
            LeakyReLU(negative_slope=0.05),
        ]
    )
    if second_dropout > 0.0:
        layers.append(Dropout(second_dropout))

    layers.extend([Dense(1), Sigmoid()])
    return Sequential(layers)


def resolve_binary_class_directories(
    *,
    data_dir: str | Path | None = None,
    cats_dir: str | Path | None = None,
    other_dir: str | Path | None = None,
) -> list[tuple[str, Path]]:
    if data_dir is not None and (cats_dir is not None or other_dir is not None):
        raise ValueError("Use either data_dir or the pair cats_dir/other_dir, not both.")

    if data_dir is not None:
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory was not found: {root}")
        class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        if len(class_dirs) != 2:
            raise ValueError("Cat detector expects exactly two class folders inside data_dir.")
        return [(path.name, path) for path in class_dirs]

    if cats_dir is None or other_dir is None:
        raise ValueError("Pass either data_dir or both cats_dir and other_dir.")

    resolved_cats = Path(cats_dir)
    resolved_other = Path(other_dir)
    if not resolved_cats.exists():
        raise FileNotFoundError(f"Cats directory was not found: {resolved_cats}")
    if not resolved_other.exists():
        raise FileNotFoundError(f"Other directory was not found: {resolved_other}")

    return [("cats", resolved_cats), ("other", resolved_other)]


def load_binary_image_paths(
    *,
    data_dir: str | Path | None = None,
    cats_dir: str | Path | None = None,
    other_dir: str | Path | None = None,
) -> tuple[list[Path], np.ndarray, list[str]]:
    class_entries = resolve_binary_class_directories(
        data_dir=data_dir,
        cats_dir=cats_dir,
        other_dir=other_dir,
    )

    image_paths: list[Path] = []
    targets: list[int] = []
    class_names = [class_name for class_name, _ in class_entries]

    for class_index, (_, class_dir) in enumerate(class_entries):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            image_paths.append(image_path)
            targets.append(class_index)

    if not image_paths:
        raise ValueError("No supported images were found in the dataset directories.")

    return image_paths, np.asarray(targets, dtype=np.int64), class_names


def build_cat_datasets(
    *,
    data_dir: str | Path | None = None,
    cats_dir: str | Path | None = None,
    other_dir: str | Path | None = None,
    feature_image_size: int = 64,
    pooled_size: int = 16,
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset, list[str], dict[str, np.ndarray | int]]:
    image_paths, targets, class_names = load_binary_image_paths(
        data_dir=data_dir,
        cats_dir=cats_dir,
        other_dir=other_dir,
    )
    train_indices, validation_indices = _stratified_split_indices(
        targets,
        validation_ratio=validation_ratio,
        seed=seed,
    )

    train_features: list[np.ndarray] = []
    train_targets: list[int] = []
    for sample_index in train_indices:
        with Image.open(image_paths[sample_index]) as image:
            for augmented_image in _iter_augmented_images(image):
                train_features.append(
                    extract_cat_features(
                        augmented_image,
                        feature_image_size=feature_image_size,
                        pooled_size=pooled_size,
                    )
                )
                train_targets.append(int(targets[sample_index]))

    validation_features = [
        extract_cat_features(
            image_paths[sample_index],
            feature_image_size=feature_image_size,
            pooled_size=pooled_size,
        )
        for sample_index in validation_indices
    ]
    validation_targets = targets[validation_indices]

    train_matrix = np.stack(train_features).astype(np.float32)
    validation_matrix = np.stack(validation_features).astype(np.float32)
    mean = train_matrix.mean(axis=0, keepdims=True)
    std = train_matrix.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    train_matrix = (train_matrix - mean) / std
    validation_matrix = (validation_matrix - mean) / std

    metadata = {
        "feature_mean": mean.reshape(-1).astype(np.float32),
        "feature_std": std.reshape(-1).astype(np.float32),
        "feature_image_size": int(feature_image_size),
        "pooled_size": int(pooled_size),
        "base_train_size": int(len(train_indices)),
        "augmented_train_size": int(len(train_features)),
        "validation_size": int(len(validation_indices)),
    }

    return (
        Dataset.from_arrays(train_matrix, np.asarray(train_targets, dtype=np.int64)),
        Dataset.from_arrays(validation_matrix, validation_targets),
        class_names,
        metadata,
    )


def extract_cat_features(
    image_or_path: Image.Image | str | Path,
    *,
    feature_image_size: int = 64,
    pooled_size: int = 16,
) -> np.ndarray:
    if isinstance(image_or_path, (str, Path)):
        with Image.open(image_or_path) as image:
            return extract_cat_features(
                image,
                feature_image_size=feature_image_size,
                pooled_size=pooled_size,
            )

    if feature_image_size <= 0:
        raise ValueError("feature_image_size must be positive.")
    if pooled_size <= 0:
        raise ValueError("pooled_size must be positive.")

    normalized_image = ImageOps.autocontrast(
        image_or_path.convert("L").resize((feature_image_size, feature_image_size))
    )
    grayscale = np.asarray(normalized_image, dtype=np.float32) / 255.0
    intensity_features = _average_pool_2d(grayscale, pooled_size)
    edge_features = _average_pool_2d(_edge_magnitude(grayscale), pooled_size)
    return np.concatenate([intensity_features, edge_features]).astype(np.float32)


def preprocess_image(
    image_path: str | Path,
    *,
    feature_image_size: int,
    pooled_size: int,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    feature_vector = extract_cat_features(
        image_path,
        feature_image_size=feature_image_size,
        pooled_size=pooled_size,
    ).reshape(1, -1)

    if feature_mean is not None and feature_std is not None:
        mean = np.asarray(feature_mean, dtype=np.float32).reshape(1, -1)
        std = np.asarray(feature_std, dtype=np.float32).reshape(1, -1)
        std[std < 1e-6] = 1.0
        feature_vector = (feature_vector - mean) / std

    return feature_vector.astype(np.float32)


def find_best_binary_threshold(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    min_threshold: float = 0.35,
    max_threshold: float = 0.65,
    steps: int = 31,
) -> tuple[float, float]:
    probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    targets = np.asarray(targets).astype(int).reshape(-1)

    best_threshold = 0.5
    best_accuracy = float(np.mean((probabilities >= 0.5).astype(int) == targets))

    for threshold in np.linspace(min_threshold, max_threshold, steps):
        accuracy = float(np.mean((probabilities >= threshold).astype(int) == targets))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    return best_threshold, best_accuracy


def save_cat_metadata(
    path: str | Path,
    *,
    class_names: list[str],
    feature_image_size: int,
    pooled_size: int,
    hidden_dims: tuple[int, int],
    dropout_rates: tuple[float, float],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    decision_threshold: float,
    selected_seed: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "class_names": class_names,
        "image_size": feature_image_size,
        "feature_image_size": feature_image_size,
        "pooled_size": pooled_size,
        "color_mode": "grayscale",
        "hidden_dims": list(hidden_dims),
        "dropout_rates": list(dropout_rates),
        "feature_mean": np.asarray(feature_mean, dtype=np.float32).reshape(-1).tolist(),
        "feature_std": np.asarray(feature_std, dtype=np.float32).reshape(-1).tolist(),
        "decision_threshold": float(decision_threshold),
        "selected_seed": int(selected_seed),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_cat_metadata(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _average_pool_2d(array: np.ndarray, pooled_size: int) -> np.ndarray:
    height, width = array.shape
    pooled_height = max(1, min(pooled_size, height))
    pooled_width = max(1, min(pooled_size, width))
    cropped_height = (height // pooled_height) * pooled_height
    cropped_width = (width // pooled_width) * pooled_width
    cropped = array[:cropped_height, :cropped_width]
    block_height = cropped_height // pooled_height
    block_width = cropped_width // pooled_width
    pooled = cropped.reshape(pooled_height, block_height, pooled_width, block_width).mean(axis=(1, 3))
    return pooled.reshape(-1)


def _edge_magnitude(array: np.ndarray) -> np.ndarray:
    grad_x = np.zeros_like(array)
    grad_y = np.zeros_like(array)
    grad_x[:, 1:-1] = array[:, 2:] - array[:, :-2]
    grad_y[1:-1, :] = array[2:, :] - array[:-2, :]
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def _iter_augmented_images(image: Image.Image) -> Iterable[Image.Image]:
    grayscale = image.convert("L")
    return (
        grayscale,
        ImageOps.mirror(grayscale),
        ImageEnhance.Contrast(grayscale).enhance(1.15),
        ImageEnhance.Brightness(grayscale).enhance(0.9),
    )


def _stratified_split_indices(
    targets: np.ndarray,
    *,
    validation_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []

    for class_id in np.unique(targets):
        class_indices = np.flatnonzero(targets == class_id)
        if len(class_indices) < 2:
            raise ValueError("Each class must contain at least two images for stratified split.")
        rng.shuffle(class_indices)
        validation_count = max(1, int(round(len(class_indices) * validation_ratio)))
        validation_count = min(validation_count, len(class_indices) - 1)
        split_index = len(class_indices) - validation_count
        train_indices.append(class_indices[:split_index])
        validation_indices.append(class_indices[split_index:])

    train_result = np.concatenate(train_indices)
    validation_result = np.concatenate(validation_indices)
    rng.shuffle(train_result)
    rng.shuffle(validation_result)
    return train_result, validation_result
