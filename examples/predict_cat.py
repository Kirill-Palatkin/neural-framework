from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.cat_model_utils import build_cat_model, load_cat_metadata, preprocess_image


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Проверка одной картинки или папки картинок обученной cat/not-cat моделью.")
    parser.add_argument(
        "--image",
        type=Path,
        help="Путь к одному изображению для проверки.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("examples/cat_or_not"),
        help="Путь к папке с изображениями для пакетной проверки.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("artifacts/cat_detector_weights.npz"),
        help="Путь к сохранённым весам модели.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=Path("artifacts/cat_detector_meta.json"),
        help="Путь к файлу с метаданными модели.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_cat_metadata(args.meta)
    feature_image_size = int(metadata.get("feature_image_size", metadata["image_size"]))
    pooled_size = int(metadata.get("pooled_size", 16))
    class_names = metadata["class_names"]
    hidden_dims = tuple(metadata.get("hidden_dims", [48, 16]))
    dropout_rates = tuple(metadata.get("dropout_rates", [0.3, 0.1]))
    decision_threshold = float(metadata.get("decision_threshold", 0.5))
    feature_mean = np.asarray(metadata.get("feature_mean"), dtype=np.float32)
    feature_std = np.asarray(metadata.get("feature_std"), dtype=np.float32)

    model = build_cat_model(
        input_dim=feature_mean.shape[0],
        hidden_dims=hidden_dims,
        dropout_rates=dropout_rates,
    )
    model.build(input_dim=feature_mean.shape[0])
    model.load_weights(args.weights)

    if args.image is not None:
        _predict_single_image(
            image_path=args.image,
            model=model,
            class_names=class_names,
            feature_image_size=feature_image_size,
            pooled_size=pooled_size,
            feature_mean=feature_mean,
            feature_std=feature_std,
            decision_threshold=decision_threshold,
        )
        return

    _predict_directory(
        images_dir=args.images_dir,
        model=model,
        class_names=class_names,
        feature_image_size=feature_image_size,
        pooled_size=pooled_size,
        feature_mean=feature_mean,
        feature_std=feature_std,
        decision_threshold=decision_threshold,
    )


def _predict_single_image(
    *,
    image_path: Path,
    model,
    class_names: list[str],
    feature_image_size: int,
    pooled_size: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    decision_threshold: float,
) -> None:
    positive_probability, predicted_label = _predict_label(
        image_path=image_path,
        model=model,
        class_names=class_names,
        feature_image_size=feature_image_size,
        pooled_size=pooled_size,
        feature_mean=feature_mean,
        feature_std=feature_std,
        decision_threshold=decision_threshold,
    )
    negative_probability = 1.0 - positive_probability
    confidence = positive_probability if predicted_label == class_names[1] else negative_probability

    print(f"Изображение: {image_path}")
    print(f"Предсказанный класс: {predicted_label}")
    print(f"Уверенность: {confidence:.4f}")
    print(f"Порог решения: {decision_threshold:.2f}")
    print(f"P({class_names[0]}) = {negative_probability:.4f}")
    print(f"P({class_names[1]}) = {positive_probability:.4f}")


def _predict_directory(
    *,
    images_dir: Path,
    model,
    class_names: list[str],
    feature_image_size: int,
    pooled_size: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    decision_threshold: float,
) -> None:
    if not images_dir.exists():
        raise SystemExit(f"Папка с изображениями не найдена: {images_dir}")

    image_paths = sorted(
        path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not image_paths:
        raise SystemExit(f"В папке нет поддерживаемых изображений: {images_dir}")

    print(f"Папка: {images_dir}")
    print(f"Файлов для проверки: {len(image_paths)}")
    print(f"Порог решения: {decision_threshold:.2f}\n")

    predicted_as_cats = 0
    predicted_as_other = 0
    true_cats = 0
    true_other = 0
    correct_cats = 0
    correct_other = 0

    for image_path in image_paths:
        positive_probability, predicted_label = _predict_label(
            image_path=image_path,
            model=model,
            class_names=class_names,
            feature_image_size=feature_image_size,
            pooled_size=pooled_size,
            feature_mean=feature_mean,
            feature_std=feature_std,
            decision_threshold=decision_threshold,
        )
        negative_probability = 1.0 - positive_probability
        confidence = positive_probability if predicted_label == class_names[1] else negative_probability
        expected_label = _expected_label_from_name(image_path.name, class_names)

        if predicted_label == class_names[0]:
            predicted_as_cats += 1
        else:
            predicted_as_other += 1

        if expected_label == class_names[0]:
            true_cats += 1
            if predicted_label == expected_label:
                correct_cats += 1
        elif expected_label == class_names[1]:
            true_other += 1
            if predicted_label == expected_label:
                correct_other += 1

        safe_name = _safe_console_text(image_path.name)
        print(
            f"{safe_name:20} -> {predicted_label:5} | "
            f"confidence={confidence:.4f} | "
            f"P({class_names[0]})={negative_probability:.4f} | "
            f"P({class_names[1]})={positive_probability:.4f}"
        )

    print("\nИтог:")
    print(f"Предсказано как {class_names[0]}: {predicted_as_cats}")
    print(f"Предсказано как {class_names[1]}: {predicted_as_other}")
    if true_cats or true_other:
        total_known = true_cats + true_other
        total_correct = correct_cats + correct_other
        print(f"Правильно определено {class_names[0]}: {correct_cats} из {true_cats}")
        print(f"Правильно определено {class_names[1]}: {correct_other} из {true_other}")
        print(f"Общая точность по размеченным файлам: {total_correct} из {total_known} ({total_correct / total_known:.4f})")


def _predict_label(
    *,
    image_path: Path,
    model,
    class_names: list[str],
    feature_image_size: int,
    pooled_size: int,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    decision_threshold: float,
) -> tuple[float, str]:
    image_vector = preprocess_image(
        image_path,
        feature_image_size=feature_image_size,
        pooled_size=pooled_size,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    positive_probability = float(model.predict(image_vector).reshape(-1)[0])
    predicted_index = 1 if positive_probability >= decision_threshold else 0
    return positive_probability, class_names[predicted_index]


def _expected_label_from_name(filename: str, class_names: list[str]) -> str | None:
    lowered = filename.lower()
    if lowered.startswith("not_cat"):
        return class_names[1]
    if lowered.startswith("cat"):
        return class_names[0]
    return None


def _safe_console_text(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding)


if __name__ == "__main__":
    main()
