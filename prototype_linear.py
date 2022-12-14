# %%
import torch
import torch.nn as nn

# %%
# topK functional implementation with torch.topk and backward pass
class TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        # return masked input
        smallest = torch.topk(input, k)[0][-1]
        return input * (input >= smallest).float()
    @staticmethod
    def backward(ctx, grad_output):
        # return straight through gradient
        grad_input = grad_output.clone()
        return grad_input, None

# take all parameters of an nn.Module topK mask them and replace them with the masked values
def topK(module, k, combined=False):
    if combined:
        # need to combine all parameters into a single tensor
        # and then apply topk and update the parameters of the module
        params = torch.cat([p.view(-1) for p in module.parameters()])
        params.data = TopK.apply(params, k)
        # update the parameters of the module
        for param in module.parameters():
            param.data = params[:param.numel()].view(param.shape)
            params = params[param.numel():] 
    else:
        for name, param in module.named_parameters():
            if 'weight' in name:
                param.data = TopK.apply(param, k)


# %%
model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10),)

topK(model, 3, combined=True)

for name, param in model.named_parameters():
    if 'weight' in name:
        print(name, param.data)

# %%
# subclassing nn.Linear
class MLinear(nn.Module):
    def __init__(self, in_features, out_features, train_weights=False, top_k=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=train_weights)
        self.bias = nn.Parameter(torch.randn(out_features), requires_grad=train_weights)
        self.mask = nn.Parameter(torch.randn(out_features, in_features))
        self.top_k = top_k

    def forward(self, x):
        if self.top_k > 0:
            _, indices = torch.topk(self.mask, self.top_k, dim=1)
            self.weight.data = torch.zeros_like(self.weight.data)
            self.weight.data.scatter_(1, indices, self.mask)
        return x @ self.weight.t()  + self.bias
    
    def train_weights(self, train_weights):
        self.weight.requires_grad = train_weights
        self.bias.requires_grad = train_weights

# %%
