"""
A small, dependency-free (NumPy only) multi-layer perceptron.

This exists so model_training.py can expose genuine, functional
deep-learning-style hyperparameters (learning rate, batch size, epochs,
optimizer choice, dropout rate) without requiring a heavyweight
TensorFlow / PyTorch install -- convenient on a laptop with no GPU, and
this whole model trains on the tiny image dataset in well under a second
per epoch.
"""
import copy

import numpy as np


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


class NumpyMLP:
    def __init__(self, input_dim: int, hidden_layers: list, num_classes: int,
                 dropout_rate: float, optimizer: str, learning_rate: float,
                 random_seed: int = 42):
        self.layer_sizes = [input_dim] + list(hidden_layers) + [num_classes]
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.lr = learning_rate
        self.rng = np.random.default_rng(random_seed)

        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)  # He init, good fit for ReLU
            self.weights.append(self.rng.normal(0, scale, size=(fan_in, fan_out)).astype(np.float32))
            self.biases.append(np.zeros((1, fan_out), dtype=np.float32))

        # Adam moment buffers
        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # Adam timestep

    def _forward(self, X: np.ndarray, training: bool):
        activations = [X]
        dropout_masks = []
        a = X
        n_layers = len(self.weights)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ W + b
            is_output_layer = (i == n_layers - 1)
            if is_output_layer:
                a = softmax(z)
            else:
                a = np.maximum(0, z)  # ReLU
                if training and self.dropout_rate > 0:
                    mask = (self.rng.random(a.shape) > self.dropout_rate).astype(np.float32)
                    mask /= (1.0 - self.dropout_rate)  # inverted dropout scaling
                    a = a * mask
                    dropout_masks.append(mask)
                else:
                    dropout_masks.append(None)
            activations.append(a)
        return activations, dropout_masks

    def _backward(self, activations, dropout_masks, y_onehot):
        n = len(y_onehot)
        n_layers = len(self.weights)
        grads_w = [None] * n_layers
        grads_b = [None] * n_layers

        # cross-entropy + softmax gradient
        delta = (activations[-1] - y_onehot) / n

        for i in reversed(range(n_layers)):
            a_prev = activations[i]
            grads_w[i] = a_prev.T @ delta
            grads_b[i] = delta.sum(axis=0, keepdims=True)
            if i > 0:
                delta = delta @ self.weights[i].T
                relu_mask = (activations[i] > 0).astype(np.float32)
                delta = delta * relu_mask
                if dropout_masks[i - 1] is not None:
                    delta = delta * dropout_masks[i - 1]
        return grads_w, grads_b

    def _apply_gradients(self, grads_w, grads_b):
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * grads_w[i]
                self.biases[i] -= self.lr * grads_b[i]
        elif self.optimizer == "adam":
            self.t += 1
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            for i in range(len(self.weights)):
                self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * grads_w[i]
                self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (grads_w[i] ** 2)
                m_hat = self.m_w[i] / (1 - beta1 ** self.t)
                v_hat = self.v_w[i] / (1 - beta2 ** self.t)
                self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

                self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * grads_b[i]
                self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (grads_b[i] ** 2)
                m_hat_b = self.m_b[i] / (1 - beta1 ** self.t)
                v_hat_b = self.v_b[i] / (1 - beta2 ** self.t)
                self.biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}', expected 'sgd' or 'adam'")

    @staticmethod
    def _cross_entropy(probs: np.ndarray, y_onehot: np.ndarray) -> float:
        eps = 1e-9
        return float(-np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1)))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        activations, _ = self._forward(X, training=False)
        return activations[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def fit(self, X_train, y_train, X_val, y_val, num_classes: int,
            epochs: int, batch_size: int, early_stopping_patience: int):
        y_train_oh = one_hot(y_train, num_classes)
        y_val_oh = one_hot(y_val, num_classes)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = np.inf
        best_state = None
        patience_counter = 0

        n = len(X_train)
        for epoch in range(1, epochs + 1):
            perm = self.rng.permutation(n)
            X_shuf, y_shuf = X_train[perm], y_train_oh[perm]

            for start in range(0, n, batch_size):
                end = start + batch_size
                Xb, yb = X_shuf[start:end], y_shuf[start:end]
                activations, dropout_masks = self._forward(Xb, training=True)
                grads_w, grads_b = self._backward(activations, dropout_masks, yb)
                self._apply_gradients(grads_w, grads_b)

            train_probs = self.predict_proba(X_train)
            train_loss = self._cross_entropy(train_probs, y_train_oh)
            train_acc = float(np.mean(np.argmax(train_probs, axis=1) == y_train))

            val_probs = self.predict_proba(X_val)
            val_loss = self._cross_entropy(val_probs, y_val_oh)
            val_acc = float(np.mean(np.argmax(val_probs, axis=1) == y_val))

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"[model_training] epoch {epoch}/{epochs} "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy((self.weights, self.biases))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"[model_training] early stopping at epoch {epoch} "
                          f"(no val_loss improvement for {early_stopping_patience} epochs)")
                    break

        if best_state is not None:
            self.weights, self.biases = best_state

        return history
