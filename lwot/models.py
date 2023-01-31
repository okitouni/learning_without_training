import torch
from torch import nn
from torchvision.models import resnet18
from .utils import topk_mask


class Scale(nn.Module):
    def __init__(self, scale=1.0, train=False, use_sigmoid=False):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.prefactor = scale
            self.scale_ = nn.Parameter(
                torch.zeros(1, dtype=torch.float), requires_grad=train
            )
        else:
            self.scale_ = nn.Parameter(torch.tensor(scale).float(), requires_grad=train)

    @property
    def scale(self):
        if self.use_sigmoid:
            return self.prefactor * torch.sigmoid(self.scale_)
        return self.scale_

    def forward(self, x):
        return x * self.scale


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class GEMBase:
    def __init__(self, threshold, topk, train_weights, train_scores=True, bias=True):
        self.weight.requires_grad_(train_weights)
        self.weight_scores = nn.Parameter(
            torch.rand_like(self.weight), requires_grad=train_scores
        )
        if bias:
          self.bias.requires_grad_(train_weights)
          self.bias_scores = nn.Parameter(torch.rand_like(self.bias), requires_grad=train_scores)
        self.topk = topk
        self.threshold = threshold

    @property
    def masked_weight(self):
        return self.weight * self.get_mask(which="weight")

    @property
    def masked_bias(self):
        return self.bias * self.get_mask(which="bias")

    def get_mask(self, which="weight"):
        """Get the mask for the layer. Intended to be used for the backward evaluation to propagate
            gradients through the mask into the scores.


        Args:
            which (str, optional): Which mask to return, either "weight", "bias". Defaults to "weight".

        Raises:
            ValueError: If `which` is not one of "weight" or "bias".

        Returns:
            Tensor: The mask for the layer.
        """
        scores = getattr(self, which + "_scores")
        if self.topk is not None:
            mask = topk_mask(scores, self.topk)
        else:
            mask = (scores >= self.threshold).float()
        return StraightThroughEstimator.apply(scores, mask)

    def train_weights(self, train=True):
        self.weight.requires_grad_(train)
        if self.bias is not None:
            self.bias.requires_grad_(train)

    def train_scores(self, train=True):
        self.weight_scores.requires_grad_(train)
        if self.bias is not None:
            self.bias_scores.requires_grad_(train)

    def sparsity(self, which="weight"):
        """Get the sparsity of the layer.

        Args:
            which (str, optional): Which sparsity to return, either "weight", "bias" or "both". Defaults to "weight".

        Raises:
            ValueError: If `which` is not one of "weight", "bias" or "both".

        Returns:
            float: The sparsity of the layer as a fraction of total weights, biases or both.
        """
        with torch.no_grad():
            if which == "weight":
                return 1 - self.get_mask().sum() / self.get_mask().numel()
            elif which == "bias":
                return 1 - self.get_mask("bias").sum() / self.get_mask("bias").numel()
            elif which == "both":
                return 1 - (
                    self.get_mask_weight().sum() + self.get_mask("bias").sum()
                ) / (self.get_mask_weight().numel() + self.get_mask("bias").numel())
            else:
                raise ValueError(f"Unknown sparsity type {which}")


class GEMLinear(nn.Linear, GEMBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        topk=None,
        threshold=0.5,
        train_weights=False,
    ) -> None:
        nn.Linear.__init__(
            self, in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        GEMBase.__init__(self, threshold, topk, train_weights, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(input, self.masked_weight, self.masked_bias)


class GEMConv2d(nn.Conv2d, GEMBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        topk=None,
        threshold=0.5,
        train_weights=False,
    ) -> None:
        nn.Conv2d.__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        GEMBase.__init__(self, threshold, topk, train_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.masked_weight, self.masked_bias)


def get_model(
    kind="mlp",
    width=512,
    depth=3,
    scale=None,
    threshold=0.5,
    topk=None,
    train_weights=False,
    flatten=True,
    tau=None,
    dropout=None, #
    batchnorm="None", # could be 'all' or 'first'
):
    if kind == "mlp":
        model = nn.Sequential()
        if flatten:
            model.add_module("flatten", nn.Flatten())
        if scale is not None:
            model.add_module("scale", Scale(scale, train=False))
        i = 0
        model.add_module(
            f"gem{i}",
            GEMLinear(
                784, width, threshold=threshold, topk=topk, train_weights=train_weights
            ),
        )
        model.add_module(f"relu{i}", nn.ReLU())
        if batchnorm != "None":
            model.add_module(f"bn{i}", nn.BatchNorm1d(width, affine=False))
        if dropout is not None:
            model.add_module(f"droupout{i}", nn.Dropout(dropout))

        for _ in range(depth - 2):
            i += 1
            model.add_module(
                f"gem{i}",
                GEMLinear(
                    width,
                    width,
                    threshold=threshold,
                    topk=topk,
                    train_weights=train_weights,
                ),
            )
            model.add_module(f"relu{i}", nn.ReLU())
            if batchnorm == "all":
                model.add_module(f"bn{i}", nn.BatchNorm1d(width, affine=False))
            if dropout is not None:
                model.add_module(f"droupout{i}", nn.Dropout(dropout))
        model.add_module(
            f"gem{depth-1}",
            GEMLinear(
                width, 10, threshold=threshold, topk=topk, train_weights=train_weights
            ),
        )
        if tau is not None:
            model.add_module("tau", Scale(tau, train=False))
        return model
    elif kind == "resnet":
        return resnet18()
