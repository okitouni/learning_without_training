import torch
from torch import nn

class Scale(nn.Module):
    def __init__(self, scale=1.0, train=False, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.prefactor = scale
            self.scale_ = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=train)
        else:
            self.scale_ = nn.Parameter(torch.tensor(scale), requires_grad=train)
    
    @property
    def scale(self):
        if self.use_sigmoid:
            return self.prefactor * torch.sigmoid(self.scale_)
        return self.scale_

    def forward(self, x):
        return x * self.scale

def topk_mask(x, k):
    shape = x.shape
    x = x.view(-1)
    smallest = torch.topk(x, k)[0][-1]
    x = x * (x >= smallest).float()
    return x.view(shape)

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
      return mask
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
        train_weights=False,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.weight.requires_grad_(train_weights)
        self.scores = nn.Parameter(torch.rand_like(self.weight), requires_grad=True)
        self.topk = topk
        self.threshold = threshold
  
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.masked_weight, self.bias)

    def get_mask(self):
        if self.topk is None:
            mask = (self.scores > self.threshold).float()
        else: 
            mask = topk_mask(self.scores, self.topk)
        return StraightThroughEstimator.apply(self.scores, mask)

    def train_weights(self, train=True):
        self.weight.requires_grad_ = train

    def sparsity(self):
        with torch.no_grad():
            return 1 - self.get_mask().sum() / self.get_mask().numel()

    @property
    def masked_weight(self):
        return self.weight * self.get_mask()