from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miniflow import Adam, BinaryCrossEntropy, Dataset, Dense, Dropout, GradientClipping, LeakyReLU, MomentumSGD, SGD, Sequential, Sigmoid


@dataclass
class TitanicArtifacts:
    age_median: float
    fare_median: float
    feature_mean: list[float]
    feature_std: list[float]
    feature_names: list[str]
    embarked_values: list[str]
    title_values: list[str]


@dataclass
class TrainingResult:
    seed: int
    optimizer_name: str
    model: Sequential
    history: dict[str, list[float]]
    metrics: dict[str, float]
    predictions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Предсказание выживания пассажиров Титаника с помощью MiniFlow MLP.")
    parser.add_argument("--zip-path", type=Path, default=Path("data/titanic.zip"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seed-trials", type=int, default=5)
    parser.add_argument("--report-out", type=Path, default=Path("artifacts/titanic_report.svg"))
    parser.add_argument("--gif-out", type=Path, default=Path("artifacts/titanic_training.gif"))
    parser.add_argument("--charts-dir", type=Path, default=Path("artifacts/titanic_charts"))
    parser.add_argument("--meta-out", type=Path, default=Path("artifacts/titanic_meta.json"))
    parser.add_argument("--weights-out", type=Path, default=Path("artifacts/titanic_weights.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _load_titanic_rows(args.zip_path)
    train_rows, validation_rows = _train_validation_split(rows, validation_ratio=args.validation_ratio, seed=args.seed)

    processor = TitanicPreprocessor()
    train_x, train_y, artifacts = processor.fit_transform(train_rows)
    validation_x, validation_y = processor.transform(validation_rows)

    train_dataset = Dataset.from_arrays(train_x, train_y)
    validation_dataset = Dataset.from_arrays(validation_x, validation_y)

    best_adam_run: TrainingResult | None = None
    for trial_index in range(args.seed_trials):
        trial_seed = args.seed + trial_index
        result = _train_single_model(
            optimizer_name="Adam",
            optimizer_factory=lambda: Adam(learning_rate=0.01, gradient_clipping=GradientClipping(max_norm=2.0)),
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            validation_features=validation_x,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=trial_seed,
        )
        print(
            f"Запуск {trial_index + 1:02d}/{args.seed_trials:02d} | "
            f"seed={trial_seed} | val_loss={result.metrics['loss']:.4f} | val_accuracy={result.metrics['accuracy']:.4f}"
        )
        if _is_better_run(result, best_adam_run):
            best_adam_run = result

    if best_adam_run is None:
        raise RuntimeError("Не удалось получить корректный запуск обучения Titanic.")

    optimizer_results = _collect_optimizer_comparison(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        validation_features=validation_x,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    confusion = _binary_confusion_matrix(validation_y, best_adam_run.predictions)
    survival_bars = _survival_bar_data(rows)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.gif_out.parent.mkdir(parents=True, exist_ok=True)
    args.charts_dir.mkdir(parents=True, exist_ok=True)
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.weights_out.parent.mkdir(parents=True, exist_ok=True)

    best_adam_run.model.save_weights(args.weights_out)
    args.meta_out.write_text(json.dumps(asdict(artifacts), indent=2), encoding="utf-8")
    _save_titanic_report(
        output_path=args.report_out,
        history=best_adam_run.history,
        metrics=best_adam_run.metrics,
        confusion=confusion,
        train_size=len(train_rows),
        validation_size=len(validation_rows),
        best_seed=best_adam_run.seed,
        optimizer_results=optimizer_results,
        survival_bars=survival_bars,
    )
    chart_paths = _save_matplotlib_charts(
        output_dir=args.charts_dir,
        history=best_adam_run.history,
        metrics=best_adam_run.metrics,
        confusion=confusion,
        optimizer_results=optimizer_results,
        survival_bars=survival_bars,
    )
    _save_training_gif(args.gif_out, best_adam_run.history, best_adam_run.metrics["accuracy"])

    print("\nМетрики Titanic на валидации:", best_adam_run.metrics)
    print("Матрица ошибок (строки = истинный класс, столбцы = предсказанный):")
    print(confusion)
    print(f"Лучший seed: {best_adam_run.seed}")
    print(f"Веса сохранены в: {args.weights_out}")
    print(f"Метаданные сохранены в: {args.meta_out}")
    print(f"SVG-отчёт сохранён в: {args.report_out}")
    print(f"PNG-графики сохранены в: {args.charts_dir}")
    for chart_name, chart_path in chart_paths.items():
        print(f"  - {chart_name}: {chart_path}")
    print(f"GIF-анимация сохранена в: {args.gif_out}")


def build_titanic_model(input_dim: int) -> Sequential:
    return Sequential(
        [
            Dense(32, input_dim=input_dim, initializer="he"),
            LeakyReLU(negative_slope=0.05),
            Dropout(rate=0.15),
            Dense(16, initializer="he"),
            LeakyReLU(negative_slope=0.05),
            Dense(1),
            Sigmoid(),
        ]
    )


class TitanicPreprocessor:
    def __init__(self) -> None:
        self.embarked_values = ["C", "Q", "S"]
        self.title_values = ["Mr", "Mrs", "Miss", "Master", "Rare"]
        self.age_median: float | None = None
        self.fare_median: float | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.feature_names = [
            "sex_female", "age", "age_missing", "sibsp", "parch", "log_fare", "family_size", "is_alone",
            "cabin_known", "child", "pclass_1", "pclass_2", "pclass_3", "embarked_C", "embarked_Q",
            "embarked_S", "title_Mr", "title_Mrs", "title_Miss", "title_Master", "title_Rare",
        ]

    def fit_transform(self, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, TitanicArtifacts]:
        ages = [float(row["Age"]) for row in rows if row["Age"]]
        fares = [float(row["Fare"]) for row in rows if row["Fare"]]
        self.age_median = float(np.median(ages))
        self.fare_median = float(np.median(fares))
        features, targets = self._encode_rows(rows)
        self.feature_mean = features.mean(axis=0, keepdims=True)
        self.feature_std = features.std(axis=0, keepdims=True)
        self.feature_std[self.feature_std < 1e-6] = 1.0
        normalized = (features - self.feature_mean) / self.feature_std
        artifacts = TitanicArtifacts(
            age_median=self.age_median,
            fare_median=self.fare_median,
            feature_mean=self.feature_mean.reshape(-1).astype(float).tolist(),
            feature_std=self.feature_std.reshape(-1).astype(float).tolist(),
            feature_names=self.feature_names,
            embarked_values=self.embarked_values,
            title_values=self.title_values,
        )
        return normalized.astype(np.float32), targets, artifacts

    def transform(self, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        if self.age_median is None or self.fare_median is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Сначала нужно обучить препроцессор, затем вызывать transform.")
        features, targets = self._encode_rows(rows)
        normalized = (features - self.feature_mean) / self.feature_std
        return normalized.astype(np.float32), targets

    def _encode_rows(self, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
        features: list[list[float]] = []
        targets: list[int] = []
        for row in rows:
            age_missing = 0.0 if row["Age"] else 1.0
            age = float(row["Age"]) if row["Age"] else float(self.age_median)
            fare = float(row["Fare"]) if row["Fare"] else float(self.fare_median)
            sibsp = float(row["SibSp"])
            parch = float(row["Parch"])
            pclass = int(row["Pclass"])
            sex_female = 1.0 if row["Sex"] == "female" else 0.0
            embarked = row["Embarked"] if row["Embarked"] else "S"
            family_size = sibsp + parch + 1.0
            is_alone = 1.0 if family_size == 1.0 else 0.0
            cabin_known = 0.0 if not row["Cabin"] else 1.0
            child = 1.0 if age < 16.0 else 0.0
            title = _extract_title(row["Name"])
            encoded = [
                sex_female, age / 80.0, age_missing, sibsp / 8.0, parch / 6.0, np.log1p(fare) / 5.0,
                family_size / 11.0, is_alone, cabin_known, child,
            ]
            encoded.extend(1.0 if pclass == class_id else 0.0 for class_id in (1, 2, 3))
            encoded.extend(1.0 if embarked == value else 0.0 for value in self.embarked_values)
            encoded.extend(1.0 if title == value else 0.0 for value in self.title_values)
            features.append(encoded)
            targets.append(int(row["Survived"]))
        return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def _train_single_model(
    *,
    optimizer_name: str,
    optimizer_factory,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    validation_features: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> TrainingResult:
    np.random.seed(seed)
    model = build_titanic_model(input_dim=train_dataset.features.shape[1])
    history = model.fit(
        train_dataset,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer_factory(),
        loss=BinaryCrossEntropy(),
        metrics=["accuracy"],
        validation_data=validation_dataset,
        restore_best_weights=True,
        early_stopping_patience=30,
        seed=seed,
        verbose=False,
    )
    metrics = model.evaluate(validation_dataset, loss=BinaryCrossEntropy(), metrics=["accuracy"])
    predictions = model.predict_classes(validation_features)
    return TrainingResult(
        seed=seed,
        optimizer_name=optimizer_name,
        model=model,
        history=history.history,
        metrics=metrics,
        predictions=np.asarray(predictions, dtype=np.int64),
    )


def _collect_optimizer_comparison(
    *,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    validation_features: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, float | str]]:
    configs = [
        ("SGD", lambda: SGD(learning_rate=0.05, gradient_clipping=GradientClipping(max_norm=2.0))),
        ("MomentumSGD", lambda: MomentumSGD(learning_rate=0.03, momentum=0.9, gradient_clipping=GradientClipping(max_norm=2.0))),
        ("Adam", lambda: Adam(learning_rate=0.01, gradient_clipping=GradientClipping(max_norm=2.0))),
    ]
    results: list[dict[str, float | str]] = []
    for index, (name, factory) in enumerate(configs):
        result = _train_single_model(
            optimizer_name=name,
            optimizer_factory=factory,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            validation_features=validation_features,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + 100 + index,
        )
        results.append({"name": name, "accuracy": float(result.metrics["accuracy"]), "loss": float(result.metrics["loss"])})
    return results


def _survival_bar_data(rows: list[dict[str, str]]) -> dict[str, list[tuple[str, float, int]]]:
    sex_stats: list[tuple[str, float, int]] = []
    for sex, label in (("female", "Women"), ("male", "Men")):
        matches = [row for row in rows if row["Sex"] == sex]
        sex_stats.append((label, sum(int(row["Survived"]) for row in matches) / len(matches), len(matches)))
    class_stats: list[tuple[str, float, int]] = []
    for pclass in ("1", "2", "3"):
        matches = [row for row in rows if row["Pclass"] == pclass]
        class_stats.append((f"Class {pclass}", sum(int(row["Survived"]) for row in matches) / len(matches), len(matches)))
    return {"sex": sex_stats, "pclass": class_stats}


def _load_titanic_rows(zip_path: Path) -> list[dict[str, str]]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Архив Titanic не найден: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("train.csv") as source:
            return list(csv.DictReader(line.decode("utf-8") for line in source))


def _train_validation_split(rows: list[dict[str, str]], *, validation_ratio: float, seed: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio должен быть между 0 и 1.")
    targets = np.asarray([int(row["Survived"]) for row in rows], dtype=np.int64)
    indices = np.arange(len(rows))
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []
    for class_id in np.unique(targets):
        class_indices = indices[targets == class_id]
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
    return [rows[index] for index in train_result], [rows[index] for index in validation_result]


def _extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", name)
    title = match.group(1).strip() if match else "Unknown"
    aliases = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare", "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"}
    normalized = aliases.get(title, title)
    return normalized if normalized in {"Mr", "Mrs", "Miss", "Master"} else "Rare"


def _binary_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=int)
    for true_value, predicted_value in zip(y_true.astype(int), y_pred.astype(int)):
        matrix[true_value, predicted_value] += 1
    return matrix


def _is_better_run(candidate: TrainingResult, current_best: TrainingResult | None) -> bool:
    if current_best is None:
        return True
    if candidate.metrics["accuracy"] != current_best.metrics["accuracy"]:
        return candidate.metrics["accuracy"] > current_best.metrics["accuracy"]
    if candidate.metrics["loss"] != current_best.metrics["loss"]:
        return candidate.metrics["loss"] < current_best.metrics["loss"]
    return candidate.seed < current_best.seed


def _save_titanic_report(
    *,
    output_path: Path,
    history: dict[str, list[float]],
    metrics: dict[str, float],
    confusion: np.ndarray,
    train_size: int,
    validation_size: int,
    best_seed: int,
    optimizer_results: list[dict[str, float | str]],
    survival_bars: dict[str, list[tuple[str, float, int]]],
) -> None:
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    total_correct = int(confusion[0, 0] + confusion[1, 1])
    total_samples = int(confusion.sum())

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1360" height="1180" viewBox="0 0 1360 1180">
  <rect width="1360" height="1180" fill="#f6f7fb"/>
  <text x="48" y="56" font-family="Segoe UI, Arial, sans-serif" font-size="30" font-weight="700" fill="#111827">Отчёт по задаче Titanic</text>
  <text x="48" y="86" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#4b5563">MiniFlow MLP на табличных признаках с сравнением оптимизаторов и анимацией обучения</text>
  {_summary_card(48, 116, "Точность на валидации", f"{metrics['accuracy']:.2%}")}
  {_summary_card(308, 116, "Потери на валидации", f"{metrics['loss']:.4f}")}
  {_summary_card(568, 116, "Верных ответов", f"{total_correct} / {total_samples}")}
  {_summary_card(828, 116, "Лучший seed", str(best_seed))}
  {_summary_card(1088, 116, "Разбиение", f"{train_size} / {validation_size}")}
  {_line_chart(48, 240, 620, 250, epochs, [("train loss", train_loss, "#2563eb"), ("val loss", val_loss, "#ef4444")], "Потери")}
  {_line_chart(692, 240, 620, 250, epochs, [("train acc", train_acc, "#059669"), ("val acc", val_acc, "#f59e0b")], "Точность")}
  {_confusion_heatmap_block(48, 520, confusion)}
  {_optimizer_bar_chart(692, 520, optimizer_results)}
  {_bar_chart(48, 850, 620, 250, survival_bars["sex"], "Доля выживших по полу", "#2563eb")}
  {_bar_chart(692, 850, 620, 250, survival_bars["pclass"], "Доля выживших по классу билета", "#ef4444")}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")


def _summary_card(x: int, y: int, title: str, value: str) -> str:
    return f"""
  <g>
    <rect x="{x}" y="{y}" rx="18" ry="18" width="220" height="92" fill="#ffffff" stroke="#dbe1ea"/>
    <text x="{x + 20}" y="{y + 34}" font-family="Segoe UI, Arial, sans-serif" font-size="15" fill="#6b7280">{title}</text>
    <text x="{x + 20}" y="{y + 68}" font-family="Segoe UI, Arial, sans-serif" font-size="28" font-weight="700" fill="#111827">{value}</text>
  </g>
"""


def _line_chart(
    x: int,
    y: int,
    width: int,
    height: int,
    x_values: list[int],
    series: list[tuple[str, list[float], str]],
    title: str,
) -> str:
    if not x_values:
        return ""

    padding_left = 56
    padding_right = 18
    padding_top = 34
    padding_bottom = 40
    plot_x = x + padding_left
    plot_y = y + padding_top
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom
    y_points = [value for _, values, _ in series for value in values]
    y_min = float(min(y_points))
    y_max = float(max(y_points))
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    parts = [
        f'<g><rect x="{x}" y="{y}" rx="18" ry="18" width="{width}" height="{height}" fill="#ffffff" stroke="#dbe1ea"/>',
        f'<text x="{x + 22}" y="{y + 28}" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="600" fill="#111827">{title}</text>',
    ]
    for tick in range(5):
        ratio = tick / 4
        line_y = plot_y + plot_height * (1.0 - ratio)
        value = y_min + (y_max - y_min) * ratio
        parts.append(f'<line x1="{plot_x}" y1="{line_y:.1f}" x2="{plot_x + plot_width}" y2="{line_y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x + 8}" y="{line_y + 5:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7280">{value:.2f}</text>')

    parts.append(f'<line x1="{plot_x}" y1="{plot_y + plot_height}" x2="{plot_x + plot_width}" y2="{plot_y + plot_height}" stroke="#9ca3af"/>')
    parts.append(f'<line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_height}" stroke="#9ca3af"/>')
    parts.append(f'<text x="{plot_x}" y="{plot_y + plot_height + 24}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7280">{x_values[0]}</text>')
    parts.append(f'<text x="{plot_x + plot_width - 18}" y="{plot_y + plot_height + 24}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7280">{x_values[-1]}</text>')

    legend_x = x + 22
    legend_y = y + height - 14
    for index, (name, values, color) in enumerate(series):
        points = []
        for pos, value in enumerate(values):
            x_ratio = pos / max(1, len(values) - 1)
            y_ratio = (value - y_min) / (y_max - y_min)
            point_x = plot_x + plot_width * x_ratio
            point_y = plot_y + plot_height * (1.0 - y_ratio)
            points.append(f"{point_x:.1f},{point_y:.1f}")
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{" ".join(points)}"/>')
        offset = index * 160
        parts.append(f'<circle cx="{legend_x + offset}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + offset + 10}" y="{legend_y + 4}" font-family="Segoe UI, Arial, sans-serif" font-size="12" fill="#374151">{name}</text>')

    parts.append("</g>")
    return "\n".join(parts)


def _confusion_heatmap_block(x: int, y: int, confusion: np.ndarray) -> str:
    max_value = max(1, int(confusion.max()))
    labels = [["TN", "FP"], ["FN", "TP"]]
    values = [[int(confusion[0, 0]), int(confusion[0, 1])], [int(confusion[1, 0]), int(confusion[1, 1])]]
    base_colors = [["#2563eb", "#ef4444"], ["#f97316", "#10b981"]]

    parts = [
        f'<g><rect x="{x}" y="{y}" rx="18" ry="18" width="620" height="290" fill="#ffffff" stroke="#dbe1ea"/>',
        f'<text x="{x + 22}" y="{y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="600" fill="#111827">Тепловая карта матрицы ошибок</text>',
        f'<text x="{x + 22}" y="{y + 58}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#6b7280">Строки = истинный класс, столбцы = предсказанный класс</text>',
        f'<text x="{x + 260}" y="{y + 78}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#374151">Предсказано 0</text>',
        f'<text x="{x + 388}" y="{y + 78}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#374151">Предсказано 1</text>',
        f'<text x="{x + 132}" y="{y + 160}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#374151">Истинно 0</text>',
        f'<text x="{x + 132}" y="{y + 288}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#374151">Истинно 1</text>',
    ]

    start_x = x + 240
    start_y = y + 98
    cell = 120
    for row in range(2):
        for col in range(2):
            value = values[row][col]
            alpha = 35 + int(190 * (value / max_value))
            rect_x = start_x + col * cell
            rect_y = start_y + row * cell
            color = f"{base_colors[row][col]}{alpha:02x}"
            parts.append(f'<rect x="{rect_x}" y="{rect_y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
            parts.append(f'<text x="{rect_x + 16}" y="{rect_y + 30}" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="#f9fafb">{labels[row][col]}</text>')
            parts.append(f'<text x="{rect_x + 16}" y="{rect_y + 76}" font-family="Segoe UI, Arial, sans-serif" font-size="34" font-weight="700" fill="#ffffff">{value}</text>')

    accuracy = (confusion[0, 0] + confusion[1, 1]) / max(1, confusion.sum())
    parts.append(f'<text x="{x + 22}" y="{y + 120}" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#374151">Точность на валидации: {accuracy:.2%}</text>')
    parts.append(f'<text x="{x + 22}" y="{y + 150}" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#374151">Истинно положительных: {int(confusion[1, 1])}</text>')
    parts.append(f'<text x="{x + 22}" y="{y + 178}" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#374151">Истинно отрицательных: {int(confusion[0, 0])}</text>')
    parts.append(f'<text x="{x + 22}" y="{y + 206}" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="#374151">Ложно положительных / отрицательных: {int(confusion[0, 1])} / {int(confusion[1, 0])}</text>')
    parts.append("</g>")
    return "\n".join(parts)


def _optimizer_bar_chart(x: int, y: int, optimizer_results: list[dict[str, float | str]]) -> str:
    width = 620
    height = 290
    plot_x = x + 50
    plot_y = y + 56
    plot_width = width - 90
    plot_height = height - 88
    max_value = max(float(result["accuracy"]) for result in optimizer_results)
    min_value = min(float(result["accuracy"]) for result in optimizer_results)
    lower_bound = min(0.5, min_value - 0.05)
    upper_bound = max(0.9, max_value + 0.03)
    bar_width = 110
    gap = 50
    colors = {"SGD": "#2563eb", "MomentumSGD": "#f59e0b", "Adam": "#10b981"}

    parts = [
        f'<g><rect x="{x}" y="{y}" rx="18" ry="18" width="{width}" height="{height}" fill="#ffffff" stroke="#dbe1ea"/>',
        f'<text x="{x + 22}" y="{y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="600" fill="#111827">Сравнение оптимизаторов</text>',
        f'<text x="{x + 22}" y="{y + 58}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#6b7280">Точность на одной и той же выборке и одном наборе признаков</text>',
    ]

    for tick in range(5):
        ratio = tick / 4
        value = lower_bound + (upper_bound - lower_bound) * ratio
        line_y = plot_y + plot_height * (1.0 - ratio)
        parts.append(f'<line x1="{plot_x}" y1="{line_y:.1f}" x2="{plot_x + plot_width}" y2="{line_y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x + 8}" y="{line_y + 5:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7280">{value:.2%}</text>')

    for index, result in enumerate(optimizer_results):
        accuracy = float(result["accuracy"])
        normalized = (accuracy - lower_bound) / max(1e-9, upper_bound - lower_bound)
        current_x = plot_x + 36 + index * (bar_width + gap)
        current_y = plot_y + plot_height * (1.0 - normalized)
        current_height = plot_y + plot_height - current_y
        color = colors[str(result["name"])]
        parts.append(f'<rect x="{current_x}" y="{current_y:.1f}" width="{bar_width}" height="{current_height:.1f}" rx="12" fill="{color}"/>')
        parts.append(f'<text x="{current_x + 18}" y="{current_y - 8:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="600" fill="#111827">{accuracy:.2%}</text>')
        parts.append(f'<text x="{current_x + 6}" y="{plot_y + plot_height + 24}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#374151">{result["name"]}</text>')

    parts.append("</g>")
    return "\n".join(parts)


def _bar_chart(
    x: int,
    y: int,
    width: int,
    height: int,
    rows: list[tuple[str, float, int]],
    title: str,
    color: str,
) -> str:
    plot_x = x + 50
    plot_y = y + 56
    plot_width = width - 90
    plot_height = height - 88
    bar_width = 120
    gap = 60

    parts = [
        f'<g><rect x="{x}" y="{y}" rx="18" ry="18" width="{width}" height="{height}" fill="#ffffff" stroke="#dbe1ea"/>',
        f'<text x="{x + 22}" y="{y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="20" font-weight="600" fill="#111827">{title}</text>',
    ]
    for tick in range(5):
        ratio = tick / 4
        line_y = plot_y + plot_height * (1.0 - ratio)
        parts.append(f'<line x1="{plot_x}" y1="{line_y:.1f}" x2="{plot_x + plot_width}" y2="{line_y:.1f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x + 8}" y="{line_y + 5:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7280">{ratio:.0%}</text>')

    for index, (label, rate, count) in enumerate(rows):
        current_x = plot_x + 56 + index * (bar_width + gap)
        current_y = plot_y + plot_height * (1.0 - rate)
        current_height = plot_y + plot_height - current_y
        parts.append(f'<rect x="{current_x}" y="{current_y:.1f}" width="{bar_width}" height="{current_height:.1f}" rx="12" fill="{color}"/>')
        parts.append(f'<text x="{current_x + 18}" y="{current_y - 8:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="13" font-weight="600" fill="#111827">{rate:.1%}</text>')
        parts.append(f'<text x="{current_x + 18}" y="{plot_y + plot_height + 24}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="#374151">{label}</text>')
        parts.append(f'<text x="{current_x + 34}" y="{plot_y + plot_height + 42}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="#6b7280">n={count}</text>')

    parts.append("</g>")
    return "\n".join(parts)


def _save_matplotlib_charts(
    *,
    output_dir: Path,
    history: dict[str, list[float]],
    metrics: dict[str, float],
    confusion: np.ndarray,
    optimizer_results: list[dict[str, float | str]],
    survival_bars: dict[str, list[tuple[str, float, int]]],
) -> dict[str, Path]:
    _configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = {
        "loss": output_dir / "training_loss.png",
        "accuracy": output_dir / "training_accuracy.png",
        "confusion": output_dir / "confusion_matrix.png",
        "optimizers": output_dir / "optimizer_comparison.png",
        "sex": output_dir / "survival_by_sex.png",
        "pclass": output_dir / "survival_by_class.png",
    }

    _save_training_curve_chart(
        output_path=chart_paths["loss"],
        epochs=list(range(1, len(history.get("loss", [])) + 1)),
        train_values=history.get("loss", []),
        validation_values=history.get("val_loss", []),
        title="Потери на обучении Titanic",
        ylabel="Значение функции потерь",
        train_label="Обучение",
        validation_label="Валидация",
        colors=("#2563eb", "#ef4444"),
    )
    _save_training_curve_chart(
        output_path=chart_paths["accuracy"],
        epochs=list(range(1, len(history.get("accuracy", [])) + 1)),
        train_values=history.get("accuracy", []),
        validation_values=history.get("val_accuracy", []),
        title=f"Точность на обучении Titanic (итог: {metrics['accuracy']:.2%})",
        ylabel="Точность",
        train_label="Обучение",
        validation_label="Валидация",
        colors=("#059669", "#f59e0b"),
    )
    _save_confusion_matrix_chart(chart_paths["confusion"], confusion)
    _save_optimizer_comparison_chart(chart_paths["optimizers"], optimizer_results)
    _save_bar_chart(
        output_path=chart_paths["sex"],
        rows=survival_bars["sex"],
        title="Доля выживших по полу",
        ylabel="Доля выживших",
        color="#2563eb",
    )
    _save_bar_chart(
        output_path=chart_paths["pclass"],
        rows=survival_bars["pclass"],
        title="Доля выживших по классу билета",
        ylabel="Доля выживших",
        color="#ef4444",
    )
    return chart_paths


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )


def _save_training_curve_chart(
    *,
    output_path: Path,
    epochs: list[int],
    train_values: list[float],
    validation_values: list[float],
    title: str,
    ylabel: str,
    train_label: str,
    validation_label: str,
    colors: tuple[str, str],
) -> None:
    if not epochs:
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot(epochs, train_values, color=colors[0], linewidth=2.6, label=train_label)
    ax.plot(epochs, validation_values, color=colors[1], linewidth=2.6, label=validation_label)
    ax.set_title(title, pad=14)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.legend(frameon=False)
    ax.set_facecolor("#fcfcfd")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix_chart(output_path: Path, confusion: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=160)
    image = ax.imshow(confusion, cmap="Blues")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    labels = [["Истинно отриц.", "Ложно полож."], ["Ложно отриц.", "Истинно полож."]]
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            color = "white" if confusion[row, col] > confusion.max() * 0.55 else "#111827"
            ax.text(
                col,
                row,
                f"{int(confusion[row, col])}\n{labels[row][col]}",
                ha="center",
                va="center",
                fontsize=11,
                color=color,
                fontweight="bold",
            )

    accuracy = (confusion[0, 0] + confusion[1, 1]) / max(1, int(confusion.sum()))
    ax.set_title(f"Матрица ошибок Titanic\nТочность на валидации: {accuracy:.2%}", pad=16)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_xticks([0, 1], ["Не выжил", "Выжил"])
    ax.set_yticks([0, 1], ["Не выжил", "Выжил"])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_optimizer_comparison_chart(output_path: Path, optimizer_results: list[dict[str, float | str]]) -> None:
    names = [str(result["name"]) for result in optimizer_results]
    accuracies = [float(result["accuracy"]) for result in optimizer_results]
    colors = ["#2563eb", "#f59e0b", "#10b981"]

    fig, ax = plt.subplots(figsize=(8.8, 5.8), dpi=160)
    bars = ax.bar(names, accuracies, color=colors[: len(names)], width=0.58)
    ax.set_title("Сравнение оптимизаторов на Titanic", pad=14)
    ax.set_ylabel("Точность на валидации")
    ax.set_ylim(0.0, max(1.0, max(accuracies) + 0.08))
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{accuracy:.2%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_bar_chart(
    *,
    output_path: Path,
    rows: list[tuple[str, float, int]],
    title: str,
    ylabel: str,
    color: str,
) -> None:
    labels = [label for label, _, _ in rows]
    values = [rate for _, rate, _ in rows]
    counts = [count for _, _, count in rows]

    fig, ax = plt.subplots(figsize=(8.6, 5.8), dpi=160)
    bars = ax.bar(labels, values, color=color, width=0.6)
    ax.set_title(title, pad=14)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)

    for bar, value, count in zip(bars, values, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.1%}\nn={count}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_training_gif(output_path: Path, history: dict[str, list[float]], final_accuracy: float) -> None:
    train_loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    train_acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    total_epochs = len(train_loss)
    if total_epochs == 0:
        return

    frame_indices = sorted(set([0] + [min(total_epochs - 1, index) for index in range(4, total_epochs, 4)] + [total_epochs - 1]))
    frames = [
        _gif_frame(
            epoch_index=epoch_index,
            train_loss=train_loss[: epoch_index + 1],
            val_loss=val_loss[: epoch_index + 1],
            train_acc=train_acc[: epoch_index + 1],
            val_acc=val_acc[: epoch_index + 1],
            final_accuracy=final_accuracy,
        )
        for epoch_index in frame_indices
    ]
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=180, loop=0)


def _gif_frame(
    *,
    epoch_index: int,
    train_loss: list[float],
    val_loss: list[float],
    train_acc: list[float],
    val_acc: list[float],
    final_accuracy: float,
) -> Image.Image:
    image = Image.new("RGB", (960, 560), "#f6f7fb")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 936, 536), radius=22, fill="#ffffff", outline="#dbe1ea")
    draw.text((48, 44), "Анимация обучения Titanic", fill="#111827")
    draw.text((48, 70), f"Эпоха {epoch_index + 1} | лучшая точность на валидации {final_accuracy:.2%}", fill="#4b5563")
    _draw_gif_chart(draw=draw, box=(48, 108, 450, 460), title="Потери", series=[("train", train_loss, "#2563eb"), ("val", val_loss, "#ef4444")])
    _draw_gif_chart(draw=draw, box=(500, 108, 902, 460), title="Точность", series=[("train", train_acc, "#059669"), ("val", val_acc, "#f59e0b")])
    return image


def _draw_gif_chart(
    *,
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, list[float], str]],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill="#ffffff", outline="#dbe1ea")
    draw.text((x0 + 18, y0 + 16), title, fill="#111827")
    plot_left = x0 + 42
    plot_top = y0 + 42
    plot_right = x1 - 18
    plot_bottom = y1 - 34
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    values = [value for _, seq, _ in series for value in seq]
    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    for tick in range(5):
        ratio = tick / 4
        line_y = plot_top + plot_height * (1.0 - ratio)
        draw.line((plot_left, line_y, plot_right, line_y), fill="#eef2f7", width=1)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#9ca3af", width=1)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#9ca3af", width=1)

    for index, (name, seq, color) in enumerate(series):
        points = []
        for pos, value in enumerate(seq):
            x_ratio = pos / max(1, len(seq) - 1)
            y_ratio = (value - y_min) / (y_max - y_min)
            px = plot_left + plot_width * x_ratio
            py = plot_top + plot_height * (1.0 - y_ratio)
            points.append((px, py))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        elif points:
            px, py = points[0]
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
        legend_x = x0 + 20 + index * 110
        legend_y = y1 - 20
        draw.ellipse((legend_x, legend_y, legend_x + 8, legend_y + 8), fill=color)
        translated = {"train": "обучение", "val": "валидация"}.get(name, name)
        draw.text((legend_x + 14, legend_y - 3), translated, fill="#374151")


if __name__ == "__main__":
    main()
