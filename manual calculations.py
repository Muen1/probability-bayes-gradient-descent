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

# -----------------------------
# MODULE 6: Visualization
# -----------------------------
def plot_history(history):
    iterations = range(1, 5)

    # Plot m and b
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, history["m"], marker='o', label='m')
    plt.plot(iterations, history["b"], marker='s', label='b')
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.title("Parameter Updates (m and b) Over Iterations")
    plt.legend()
    plt.grid(True)

    # Plot Error
    plt.subplot(1, 2, 2)
    plt.plot(iterations, history["error"], color='red', marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.title("Error Over Iterations")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# -----------------------------
# MODULE 7: Main runner
# -----------------------------
def main():
    X, Y = load_data()
    m_final, b_final, history = gradient_descent_4_iterations(X, Y, lr=0.01)
    
    print(f"Final values after 4 iterations: m = {m_final:.4f}, b = {b_final:.4f}")
    print(f"Final predictions: {history['predictions'][-1]}")

    plot_history(history)

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

# -----------------------------
# MODULE 6: Visualization
# -----------------------------
def plot_history(history):
    iterations = range(1, 5)

    # Plot m and b
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(iterations, history["m"], marker='o', label='m')
    plt.plot(iterations, history["b"], marker='s', label='b')
    plt.xlabel("Iteration")
    plt.ylabel("Parameter Value")
    plt.title("Parameter Updates (m and b) Over Iterations")
    plt.legend()
    plt.grid(True)

    # Plot Error
    plt.subplot(1, 2, 2)
    plt.plot(iterations, history["error"], color='red', marker='x')
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error")
    plt.title("Error Over Iterations")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# -----------------------------
# MODULE 7: Main runner
# -----------------------------
def main():
    X, Y = load_data()
    m_final, b_final, history = gradient_descent_4_iterations(X, Y, lr=0.01)
    
    print(f"Final values after 4 iterations: m = {m_final:.4f}, b = {b_final:.4f}")
    print(f"Final predictions: {history['predictions'][-1]}")

    plot_history(history)

# -----------------------------
# Execute
# -----------------------------
if __name__ == "__main__":
    main()
