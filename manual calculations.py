import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# MODULE 1: Load data
# -----------------------------
def load_data():
    X = np.array([1, 2, 3], dtype=float)
    Y = np.array([2, 3, 4], dtype=float)
    return X, Y

# -----------------------------
# MODULE 2: Prediction function
# -----------------------------
def predict(X, m, b):
    return m * X + b

# -----------------------------
# MODULE 3: Compute MSE
# -----------------------------
def compute_error(Y, Y_pred):
    return np.mean((Y - Y_pred) ** 2)

# -----------------------------
# MODULE 4: Compute gradients
# -----------------------------
def compute_gradients(X, Y, m, b):
    n = len(Y)
    Y_pred = predict(X, m, b)
    dm = (-2 / n) * np.sum(X * (Y - Y_pred))
    db = (-2 / n) * np.sum(Y - Y_pred)
    return dm, db

# -----------------------------
# MODULE 5: Gradient descent for 4 iterations
# -----------------------------
def gradient_descent_4_iterations(X, Y, m_init=0.0, b_init=0.0, lr=0.01):
    m, b = m_init, b_init
    history = {"m": [], "b": [], "error": [], "predictions": []}

    for i in range(1, 5):  # Only iterations 1 to 4
        dm, db = compute_gradients(X, Y, m, b)

        # Update parameters
        m -= lr * dm
        b -= lr * db

        # Compute predictions and error
        Y_pred = predict(X, m, b)
        error = compute_error(Y, Y_pred)

        # Store history
        history["m"].append(m)
        history["b"].append(b)
        history["error"].append(error)
        history["predictions"].append(Y_pred)

        # Print detailed info for each iteration
        print(f"Iteration {i}:")
        print(f"  Gradient dm = {dm:.4f}, db = {db:.4f}")
        print(f"  Updated m = {m:.4f}, b = {b:.4f}")
        print(f"  Predictions = {Y_pred}")
        print(f"  Error = {error:.4f}\n")

    return m, b, history
