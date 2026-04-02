import numpy as np
from miniflow import Sequential, Dense, Tanh, MSELoss
from miniflow.optimizers import Adam

X = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)
y = np.array([[-1.0], [0.0], [1.0], [2.0]], dtype=np.float32)

model = Sequential([
    Dense(8, input_dim=1, initializer="he"),
    Tanh(),
    Dense(1),
])

history = model.fit(
    X, y,
    epochs=300,
    batch_size=4,
    optimizer=Adam(learning_rate=0.01),
    loss=MSELoss(),
    metrics=["mse"],
    verbose=False,
)

print("final loss:", history.history["loss"][-1])
print("predictions:", model.predict(X).reshape(-1))
