# %%
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from lwot.models import get_model
from lwot.utils import Loader, accuracy
from copy import deepcopy
from matching import weight_matching

# CONFIG
DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# MODEL PARAMS
SCALE = 1
WIDTH = 512
# Loss PARAMS
LR = 1e-4
WD = 1e-5
TAU = 1
ALPHA = 1
BATCHSIZE = -1

torch.manual_seed(0)
train_dataset = MNIST(root="/data/ml_data", train=True, download=True)
val_dataset = MNIST(root="/data/ml_data", train=False, download=True)
train_dataset.data = train_dataset.data.float() / 255
val_dataset.data = val_dataset.data.float() / 255

trainloader = Loader(train_dataset, batch_size=BATCHSIZE, device=DEV)
valloader = Loader(val_dataset, batch_size=-1, device=DEV)

# BTW THIS IS MEANT TO BE RUN IN A JUPYTER NOTEBOOK
# SO THE LOGICAL FLOW IS A BIT WONKY lol
# lets see
# where are the models loaded
model1 = get_model(
    width=WIDTH,
    depth=3,
    scale=SCALE,
    tau=TAU
)

model2 = get_model(
    width=WIDTH,
    depth=3,
    scale=SCALE,
    tau=TAU
)


# %%
def similarity(model1, model2, loader):
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEV)
            y = y.to(DEV)
            y1 = model1(x)
            y2 = model2(x)
            # cosine similarity
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos(y1, y2)
            return sim


def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEV)
            y = y.to(DEV)
            y1 = model(x)
            loss = F.mse_loss(y1, F.one_hot(y, 10))
            acc = accuracy(y1, y)
            return loss, acc
# %%
cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
def weight_similarity(model1, model2, all=False):
    res = {}
    if all:
        for (name, p1), (_, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            res[name] = cos(p1.reshape(-1), p2.reshape(-1)).mean().item()
    else:
        for (name, c1), (_, c2) in zip(model1.named_children(), model2.named_children()):
            if hasattr(c1, "masked_weight"):
                res[name+".masked_weight"] = cos(c1.masked_weight.reshape(-1), c2.masked_weight.reshape(-1)).mean().item()
                res[name+".masked_bias"] = cos(c1.masked_bias.reshape(-1), c2.masked_bias.reshape(-1)).mean().item()
    string = ["{}: {:.3f}".format(k, v) for k, v in res.items()]
    return ", ".join(string)

# %%
torch.set_float32_matmul_precision('high')
model1 = torch.compile(model1)
model2 = torch.compile(model2)

# LOAD MODEL WEIGHTS
# name = f"512_tau10_scale1_ms1_seed1_wd1e-07_TWFalse"
name = f"512_tau10_scale1_ms0_seed0_wd5e-07_TWFalse_bnfirst_drp0.01"
model1 = torch.load(f"/data/kitouni/LWOT/MNIST/MLP/{name}/checkpoints/model_init.pt")
model1.load_state_dict(
    torch.load(f"/data/kitouni/LWOT/MNIST/MLP/{name}/checkpoints/MLP_9999.pt")
)
# name = f"512_tau10_scale1_ms0_seed1_wd1e-07_TWFalse"
name = "512_tau10_scale1_ms1_seed0_wd5e-07_TWFalse_bnfirst_drp0.01"
model2 = torch.load(f"/data/kitouni/LWOT/MNIST/MLP/{name}/checkpoints/model_init.pt")
model2.load_state_dict(
    torch.load(f"/data/kitouni/LWOT/MNIST/MLP/{name}/checkpoints/MLP_9999.pt")
)
model1 = model1.to(DEV)
model2 = model2.to(DEV)
model1.eval()
model2.eval()

# sim = similarity(model1, model2, valloader)
# print("mean similarity", sim.mean().item())
# print("max similarity", sim.max().item())
# print("min similarity", sim.min().item())
# print("std similarity", sim.std().item())
# %%
def get_loss_barrier(model1, model2, lambd=0.5):
  model3 = deepcopy(model2)
  for p1, p2, p3 in zip(model1.parameters(), model2.parameters(), model3.parameters()):
      p3.data = lambd * p1.data + (1 - lambd) * p2.data

  model3 = model3.to(DEV)
  print(f"Performance with convex combination lambda {lambd}")
#   sim = similarity(model3, model1, valloader)
#   print("mean similarity", sim.mean().item())
#   print("max similarity", sim.max().item())
#   print("min similarity", sim.min().item())
#   print("std similarity", sim.std().item())

  print("Val Performance")
  print("model1", "loss: {}, acc: {}".format(*evaluate(model1, valloader)))
  print("model2", "loss: {}, acc: {}".format(*evaluate(model2, valloader)))
  print("combo model", "loss: {}, acc: {}".format(*evaluate(model3, valloader)), "<-----")

  print("Train Performance")
  print("model1", "loss: {}, acc: {}".format(*evaluate(model1, trainloader)))
  print("model2", "loss: {}, acc: {}".format(*evaluate(model2, trainloader)))
  print("combo model", "loss: {}, acc: {}".format(*evaluate(model3, trainloader)),"<-----")
# %%
print("pre-weight matching")
get_loss_barrier(model1, model2)
print("weight similarity before matching\n", weight_similarity(model1, model2),)
print("post-weight matching")
model3 = weight_matching(model1, model2, inplace=False, on_masks=True)
get_loss_barrier(model1, model3)
print("weight similarity after matching\n", weight_similarity(model1, model3))



# %%
