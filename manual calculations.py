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
