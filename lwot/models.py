import torch
from torch import nn as nn


def topk_mask(x, k):
    shape = x.shape
    x = x.view(-1)
    smallest = torch.topk(x, k)[0][-1]
    x = x * (x >= smallest).float()
    return x.view(shape)

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
      return x * mask

    @staticmethod
    def backward(ctx, grad_output):
      return grad_output, None

class GEMLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        topk=None,
        threshold=0.5,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.scores = nn.Parameter(torch.rand_like(self.weight), requires_grad=True)
        self.topk = topk
        self.threshold = threshold

    def get_mask(self):
        if self.topk is None:
            return self.scores > self.threshold
        return topk_mask(self.scores, self.topk)

    def masked_weight(self):
        return StraightThroughEstimator.apply(self.weight, self.get_mask())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.masked_weight(), self.bias)