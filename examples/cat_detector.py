from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miniflow import BinaryCrossEntropy, GradientClipping
from miniflow.optimizers import Adam
from examples.cat_model_utils import (
    build_cat_datasets,
    build_cat_model,
    find_best_binary_threshold,
    save_cat_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Бинарный детектор котиков на основе полносвязной нейросети.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Путь к папке с двумя подкаталогами классов, например 'cats' и 'other'.",
    )
    parser.add_argument("--cats-dir", type=Path, help="Путь к папке с изображениями котов.")
    parser.add_argument("--other-dir", type=Path, help="Путь к папке с изображениями класса other.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Размер изображения перед извлечением признаков интенсивности и контуров.",
    )
    parser.add_argument(
        "--pooled-size",
        type=int,
        default=16,
        help="Размер pooled-представления. Итоговый вектор имеет длину 2 * pooled_size^2.",
    )
    parser.add_argument("--epochs", type=int, default=70, help="Количество эпох обучения для одного старта.")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер мини-батча.")
    parser.add_argument("--hidden-1", type=int, default=48, help="Размер первого скрытого слоя.")
    parser.add_argument("--hidden-2", type=int, default=16, help="Размер второго скрытого слоя.")
    parser.add_argument("--dropout-1", type=float, default=0.30, help="Dropout после первого скрытого слоя.")
    parser.add_argument("--dropout-2", type=float, default=0.10, help="Dropout после второго скрытого слоя.")
    parser.add_argument(
        "--color-mode",
        choices=["grayscale", "rgb"],
        default="grayscale",
        help="Параметр оставлен для совместимости. Детектор теперь использует grayscale-признаки и контуры.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0006, help="Learning rate оптимизатора Adam.")
    parser.add_argument("--patience", type=int, default=12, help="Сколько эпох ждать улучшения val_accuracy.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Минимальное улучшение метрики для early stopping.")
    parser.add_argument("--validation-ratio", type=float, default=0.2, help="Доля валидационной выборки.")
    parser.add_argument("--seed", type=int, default=42, help="Базовый seed для split и генерации стартов.")
    parser.add_argument(
        "--seed-trials",
        type=int,
        default=10,
        help="Сколько независимых стартов попробовать и выбрать лучший по валидации.",
    )
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=Path("artifacts/cat_detector_weights.npz"),
        help="Куда сохранить обученные веса модели.",
    )
    parser.add_argument(
        "--meta-out",
        type=Path,
        default=Path("artifacts/cat_detector_meta.json"),
        help="Куда сохранить метаданные модели.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)

    train_dataset, validation_dataset, class_names, preprocessing = build_cat_datasets(
        data_dir=args.data_dir,
        cats_dir=args.cats_dir,
        other_dir=args.other_dir,
        feature_image_size=args.image_size,
        pooled_size=args.pooled_size,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )

    input_dim = train_dataset.features.shape[1]
    hidden_dims = (args.hidden_1, args.hidden_2)
    dropout_rates = (args.dropout_1, args.dropout_2)

    best_trial: dict[str, object] | None = None

    for trial_index in range(args.seed_trials):
        trial_seed = args.seed + trial_index
        np.random.seed(trial_seed)
        model = build_cat_model(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout_rates=dropout_rates,
        )

        history = model.fit(
            train_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=Adam(
                learning_rate=args.learning_rate,
                gradient_clipping=GradientClipping(max_norm=1.2),
            ),
            loss=BinaryCrossEntropy(),
            metrics=["accuracy"],
            validation_data=validation_dataset,
            restore_best_weights=True,
            seed=trial_seed,
            verbose=False,
            early_stopping_patience=args.patience,
            min_delta=args.min_delta,
        )

        metrics = model.evaluate(validation_dataset, loss=BinaryCrossEntropy(), metrics=["accuracy"])
        probabilities = model.predict(validation_dataset.features).reshape(-1)
        threshold, tuned_accuracy = find_best_binary_threshold(probabilities, validation_dataset.targets)
        best_epoch = _best_epoch_from_history(history.history.get("val_accuracy", []))

        trial_summary = {
            "seed": trial_seed,
            "model": model,
            "history": history,
            "metrics": metrics,
            "probabilities": probabilities,
            "threshold": threshold,
            "tuned_accuracy": tuned_accuracy,
            "best_epoch": best_epoch,
        }

        print(
            f"Trial {trial_index + 1:02d}/{args.seed_trials:02d} | "
            f"seed={trial_seed} | "
            f"val_loss={metrics['loss']:.4f} | "
            f"val_accuracy@0.5={metrics['accuracy']:.4f} | "
            f"val_accuracy@threshold={tuned_accuracy:.4f} | "
            f"threshold={threshold:.2f}"
        )

        if _is_better_trial(trial_summary, best_trial):
            best_trial = trial_summary

    if best_trial is None:
        raise RuntimeError("Training did not produce any valid trial.")

    best_model = best_trial["model"]
    probabilities = np.asarray(best_trial["probabilities"], dtype=np.float32)
    threshold = float(best_trial["threshold"])
    validation_targets = validation_dataset.targets.astype(int).reshape(-1)
    validation_predictions = (probabilities >= threshold).astype(int)

    confusion = np.zeros((2, 2), dtype=int)
    for true_label, predicted_label in zip(validation_targets, validation_predictions):
        confusion[true_label, predicted_label] += 1

    best_model.save_weights(args.weights_out)
    save_cat_metadata(
        args.meta_out,
        class_names=class_names,
        feature_image_size=int(preprocessing["feature_image_size"]),
        pooled_size=int(preprocessing["pooled_size"]),
        hidden_dims=hidden_dims,
        dropout_rates=dropout_rates,
        feature_mean=np.asarray(preprocessing["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(preprocessing["feature_std"], dtype=np.float32),
        decision_threshold=threshold,
        selected_seed=int(best_trial["seed"]),
    )

    print("\nКлассы:", class_names)
    print(
        "Размеры выборок:",
        {
            "train_base": int(preprocessing["base_train_size"]),
            "train_augmented": int(preprocessing["augmented_train_size"]),
            "validation": int(preprocessing["validation_size"]),
        },
    )
    print(f"Лучшая инициализация: seed={best_trial['seed']}, лучшая эпоха={best_trial['best_epoch']}")
    print(
        "Метрики на валидации:",
        {
            "loss": float(best_trial["metrics"]["loss"]),
            "accuracy@0.5": float(best_trial["metrics"]["accuracy"]),
            "accuracy@threshold": float(best_trial["tuned_accuracy"]),
            "threshold": threshold,
        },
    )
    print("Confusion matrix (строки = истинный класс, столбцы = предсказанный класс):")
    print(confusion)
    print(f"{class_names[0]} -> {class_names[0]}: {confusion[0, 0]}, {class_names[0]} -> {class_names[1]}: {confusion[0, 1]}")
    print(f"{class_names[1]} -> {class_names[0]}: {confusion[1, 0]}, {class_names[1]} -> {class_names[1]}: {confusion[1, 1]}")
    print(f"Веса модели сохранены в: {args.weights_out}")
    print(f"Метаданные модели сохранены в: {args.meta_out}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed_trials < 1:
        raise SystemExit("Параметр --seed-trials должен быть >= 1.")

    if args.data_dir is None:
        if args.cats_dir is None or args.other_dir is None:
            raise SystemExit("Укажите либо --data-dir, либо одновременно --cats-dir и --other-dir.")
    elif args.cats_dir is not None or args.other_dir is not None:
        raise SystemExit("Используйте либо --data-dir, либо пару --cats-dir/--other-dir.")


def _best_epoch_from_history(values: list[float]) -> int | None:
    if not values:
        return None
    best_value = max(values)
    return values.index(best_value) + 1


def _is_better_trial(candidate: dict[str, object], current_best: dict[str, object] | None) -> bool:
    if current_best is None:
        return True

    candidate_accuracy = float(candidate["tuned_accuracy"])
    current_accuracy = float(current_best["tuned_accuracy"])
    if candidate_accuracy != current_accuracy:
        return candidate_accuracy > current_accuracy

    candidate_loss = float(candidate["metrics"]["loss"])
    current_loss = float(current_best["metrics"]["loss"])
    if candidate_loss != current_loss:
        return candidate_loss < current_loss

    return int(candidate["seed"]) < int(current_best["seed"])


if __name__ == "__main__":
    main()
