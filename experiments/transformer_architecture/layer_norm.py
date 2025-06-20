class LayerNorm:
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Parameters
    ----------
    normalized_shape : int
        Size of the layer's output.
    beta : np.ndarray
        The shift parameter (bias) to be added.
    gamma : np.ndarray
        The scale parameter to be multiplied.
    eps : float
        A value added to the denominator for numerical stability (default: 1e-12).

    """

    def __init__(self, normalized_shape: int, beta: np.ndarray,
                 gamma: np.ndarray, eps: float = 1e-12):
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
        self.eps = eps
        self.beta = beta
        self.gamma = gamma

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
         Apply the layer normalization on the input tensor.

         Parameters
         ----------
         x : np.ndarray
             The input data for normalization.

         Returns
         -------
         np.ndarray
             The normalized data.
         """

        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        return (((x - mean) / np.sqrt(var + self.eps)) * self.gamma) + self.beta