import torch
from torch import nn, Tensor
from torch.nn import functional as F

eps=1e-10

class Binarize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor):
        """We simply take the sign of input tensor"""
        input_tensor = torch.masked_fill(input_tensor, input_tensor == 0, eps)
        ctx.save_for_backward(input_tensor)
        return torch.sign(input_tensor)

    @staticmethod
    def backward(ctx, grad_output):
        """
        We use the formula in the paper
        q = sign(r) => dq/dr = 1 if |r|<= 1 else 0
        (https://arxiv.org/pdf/1602.02830 Page 2 EQ(4))
        """
        input_tensor, = ctx.saved_tensors
        return torch.masked_fill(grad_output, torch.abs(input_tensor) > 1, 0)


class BinaryLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.sign(torch.randn((out_features, in_features))))
        self.bias = nn.Parameter(torch.sign(torch.randn(out_features)))
        self.binarize = Binarize.apply

    def forward(self, x: Tensor) -> Tensor:
        weight = self.binarize(self.weight)
        bias = self.binarize(self.bias)
        return F.linear(x, weight, bias)
    

class BinaryBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int):

        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.linear = BinaryLinear(in_features, out_features)
        #self.batch_norm = nn.BatchNorm1d(out_features)
        self.binarize = Binarize.apply

    def forward(self, input_tensor: Tensor) -> Tensor:

        input_tensor = self.linear(input_tensor)
        #input_tensor = self.batch_norm(input_tensor)
        return self.binarize(input_tensor)

