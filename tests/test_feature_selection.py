import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold

from feature_selection import build_selector, flatten


def test_flatten_converts_image_batch_to_feature_matrix():
    X = np.zeros((4, 8, 8), dtype=np.float32)
    out = flatten(X)
    assert out.shape == (4, 64)


@pytest.mark.parametrize(
    "method,expected_type",
    [
        ("select_k_best", SelectKBest),
        ("pca", PCA),
        ("variance_threshold", VarianceThreshold),
    ],
)
def test_build_selector_supported_methods(method, expected_type):
    selector = build_selector(method, num_features=10, variance_threshold=0.01)
    assert isinstance(selector, expected_type)


def test_build_selector_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown feature_selection.method"):
        build_selector("not-a-method", num_features=10, variance_threshold=0.01)