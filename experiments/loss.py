import torch.nn as nn
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO)

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin


    def forward(self, anchor, positive, negative, size_average=True):
        a_dim = anchor.ndim
        p_dim = positive.ndim
        n_dim = negative.ndim
        if not (a_dim == p_dim and p_dim == n_dim):
            raise RuntimeError(
                f"The anchor, positive, and negative tensors are expected to have "
                f"the same number of dimensions, but got: anchor {a_dim}D, "
                f"positive {p_dim}D, and negative {n_dim}D inputs")

        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        logging.info(f'Calculated loss...{losses.mean()}')

        return losses.mean() if size_average else losses.sum()
