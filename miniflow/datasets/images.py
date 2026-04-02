from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ..data import Dataset


def load_image_folder_dataset(
    root_dir: str | Path,
    image_size: tuple[int, int] = (32, 32),
    color_mode: str = "rgb",
    normalize: bool = True,
) -> tuple[Dataset, list[str]]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory was not found: {root}")

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Image dataset must contain at least two class folders.")

    features: list[np.ndarray] = []
    targets: list[int] = []
    class_names = [path.name for path in class_dirs]
    mode = "RGB" if color_mode.lower() == "rgb" else "L"

    for class_index, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            with Image.open(image_path) as image:
                array = image.convert(mode).resize(image_size)
            numeric = np.asarray(array, dtype=np.float32)
            if color_mode.lower() == "rgb":
                numeric = numeric.reshape(-1)
            else:
                numeric = numeric.reshape(-1, 1).reshape(-1)
            if normalize:
                numeric /= 255.0
            features.append(numeric)
            targets.append(class_index)

    if not features:
        raise ValueError("No supported images were found in the dataset directory.")

    return Dataset(np.stack(features), np.asarray(targets, dtype=np.int64)), class_names
