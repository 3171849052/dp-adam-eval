from dataclasses import dataclass
import torch

Tensor = torch.Tensor
CovarianceDict = dict[str, Tensor]


@dataclass
class CovariancePair:
    A: CovarianceDict
    G: CovarianceDict
