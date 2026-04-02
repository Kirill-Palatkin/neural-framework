from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miniflow import Adam, CategoricalCrossEntropy, Dataset, Dense, GradientClipping, ReLU, Sequential, Softmax


def main() -> None:
    try:
        from sklearn.datasets import load_iris
    except ImportError as error:
        raise SystemExit("Для примера Iris установите scikit-learn: pip install scikit-learn") from error

    iris = load_iris()
    features = iris.data.astype(np.float32)
    targets = iris.target.astype(np.int64)

    dataset = Dataset.from_arrays(features, targets).shuffle(seed=7)
    train_dataset, validation_dataset = dataset.split(validation_ratio=0.2, shuffle=False)

    model = Sequential(
        [
            Dense(16, input_dim=4, initializer="he"),
            ReLU(),
            Dense(16, initializer="he"),
            ReLU(),
            Dense(3),
            Softmax(),
        ]
    )

    history = model.fit(
        train_dataset,
        epochs=120,
        batch_size=16,
        optimizer=Adam(learning_rate=0.01, gradient_clipping=GradientClipping(max_norm=2.0)),
        loss=CategoricalCrossEntropy(),
        metrics=["accuracy"],
        validation_data=validation_dataset,
    )

    metrics = model.evaluate(
        validation_dataset,
        loss=CategoricalCrossEntropy(),
        metrics=["accuracy"],
    )
    print("\nИтоговые метрики на валидации:", metrics)
    print("Предсказанные классы для первых 5 валидационных объектов:", model.predict_classes(validation_dataset.features[:5]))
    print("Ключи истории обучения:", list(history.history.keys()))


if __name__ == "__main__":
    main()
