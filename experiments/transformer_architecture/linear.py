import numpy as np

class Linear:
    """
    y = x*W^T + b

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.

    Attributes
    ----------
    weights : np.ndarray
        The weight matrix of the layer (out_features, in_features).
    bias : np.ndarray
        The bias vector of the layer (out_features,).

    """

    def __init__(self, in_features: int, out_features: int):
        self.out_features = out_features
        self.in_features = in_features
        self.weights = np.random.randn(self.out_features, self.in_features)
        self.bias = np.zeros(self.out_features)

    def forward(self, input_tensor) -> np.ndarray:
        return np.matmul(input_tensor, self.weights.T) + self.bias
