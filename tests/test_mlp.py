import numpy as np
import pytest

from mlp import NumpyMLP, one_hot, softmax


def test_one_hot_encodes_labels():
    y = np.array([0, 2, 1, 2], dtype=np.int64)
    oh = one_hot(y, num_classes=3)
    assert oh.shape == (4, 3)
    np.testing.assert_array_equal(oh[0], np.array([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(oh[1], np.array([0.0, 0.0, 1.0], dtype=np.float32))


def test_softmax_rows_sum_to_one():
    z = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]], dtype=np.float32)
    probs = softmax(z)
    row_sums = probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(2), rtol=1e-6, atol=1e-6)


def test_predict_proba_and_predict_shapes(tiny_classification_data):
    X_train, y_train, X_val, y_val = tiny_classification_data
    model = NumpyMLP(
        input_dim=2,
        hidden_layers=[8],
        num_classes=2,
        dropout_rate=0.0,
        optimizer="adam",
        learning_rate=0.01,
        random_seed=42,
    )
    model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=2,
        epochs=3,
        batch_size=4,
        early_stopping_patience=3,
    )

    probs = model.predict_proba(X_val)
    preds = model.predict(X_val)

    assert probs.shape == (len(X_val), 2)
    assert preds.shape == (len(X_val),)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(len(X_val)), rtol=1e-6, atol=1e-6)


def test_fit_returns_history_with_expected_keys(tiny_classification_data):
    X_train, y_train, X_val, y_val = tiny_classification_data
    model = NumpyMLP(
        input_dim=2,
        hidden_layers=[6],
        num_classes=2,
        dropout_rate=0.1,
        optimizer="sgd",
        learning_rate=0.05,
        random_seed=7,
    )

    history = model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=2,
        epochs=5,
        batch_size=4,
        early_stopping_patience=2,
    )

    for key in ["train_loss", "train_acc", "val_loss", "val_acc"]:
        assert key in history
        assert 1 <= len(history[key]) <= 5


def test_unknown_optimizer_raises_value_error():
    model = NumpyMLP(
        input_dim=2,
        hidden_layers=[4],
        num_classes=2,
        dropout_rate=0.0,
        optimizer="not-valid",
        learning_rate=0.01,
        random_seed=0,
    )
    grads_w = [np.zeros_like(w) for w in model.weights]
    grads_b = [np.zeros_like(b) for b in model.biases]

    with pytest.raises(ValueError, match="Unknown optimizer"):
        model._apply_gradients(grads_w, grads_b)