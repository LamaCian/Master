import torch
import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, in_features: int, out_features: int, p: float = 0.5):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb1 = self.linear1(x)
        emb1_activated = torch.nn.functional.gelu(emb1)
        emb2 = self.drop(self.linear2(emb1_activated))
        return self.layer_norm(emb1 + emb2)