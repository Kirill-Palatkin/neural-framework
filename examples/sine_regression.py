from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from miniflow import Dataset, Dense, GradientClipping, MSELoss, Sequential, Tanh
from miniflow.optimizers import MomentumSGD


def main() -> None:
    generator = np.random.default_rng(42)
    features = np.linspace(-3.0, 3.0, 320, dtype=np.float32).reshape(-1, 1)
    targets = np.sin(features) + 0.08 * generator.normal(size=features.shape).astype(np.float32)

    dataset = Dataset.from_arrays(features, targets).shuffle(seed=42)
    train_dataset, validation_dataset = dataset.split(validation_ratio=0.2, shuffle=False)

    model = Sequential(
        [
            Dense(32, input_dim=1, initializer="he"),
            Tanh(),
            Dense(32, initializer="he"),
            Tanh(),
            Dense(1),
        ]
    )

    model.fit(
        train_dataset,
        epochs=250,
        batch_size=32,
        optimizer=MomentumSGD(
            learning_rate=0.03,
            momentum=0.9,
            gradient_clipping=GradientClipping(max_norm=1.5),
        ),
        loss=MSELoss(),
        metrics=["mse", "mae"],
        validation_data=validation_dataset,
    )

    metrics = model.evaluate(validation_dataset, loss=MSELoss(), metrics=["mse", "mae"])
    sample_predictions = model.predict(np.array([[-2.5], [0.0], [2.5]], dtype=np.float32)).reshape(-1)

    print("\nМетрики на валидации:", metrics)
    print("Предсказания для x = -2.5, 0.0, 2.5:", np.round(sample_predictions, 4))


if __name__ == "__main__":
    main()
