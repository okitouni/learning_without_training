# %%
import torch
from torch import nn 
from torchvision.datasets import MNIST
from lwot.models import GEMLinear, Scale
from lwot.utils import Loader, accuracy
from copy import deepcopy

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# MODEL PARAMS
SCALE = 2
WIDTH = 512
# Loss PARAMS
LR = 1e-3
WD = 1e-5
TAU = 100
ALPHA = 1
BATCHSIZE = -1
WANDB = True

torch.manual_seed(0)
train_dataset = MNIST(root='/data/ml_data', train=True, download=True)
val_dataset = MNIST(root='/data/ml_data', train=False, download=True)
train_dataset.data = train_dataset.data.float() / 255
val_dataset.data = val_dataset.data.float() / 255

trainloader = Loader(train_dataset, batch_size=BATCHSIZE, device=DEV)
valloader = Loader(val_dataset, batch_size=-1, device=DEV)



model1 = nn.Sequential(
    nn.Flatten(),
    Scale(SCALE, train=True, use_sigmoid=True),
    GEMLinear(28*28, WIDTH, threshold=0.5, train_weights=False),
    nn.ReLU(),
    GEMLinear(WIDTH, WIDTH, threshold=0.5, train_weights=False),
    nn.ReLU(),
    GEMLinear(WIDTH, 10, threshold=0.5, train_weights=False))

model2 = nn.Sequential(
    nn.Flatten(),
    Scale(SCALE, train=True, use_sigmoid=True),
    GEMLinear(28*28, WIDTH, threshold=0.5, train_weights=False),
    nn.ReLU(),
    GEMLinear(WIDTH, WIDTH, threshold=0.5, train_weights=False),
    nn.ReLU(),
    GEMLinear(WIDTH, 10, threshold=0.5, train_weights=False))

# %%
model1.load_state_dict(torch.load("/data/kitouni/LWOT/MNIST/MLP/checkpoints/MLP_9500.pt"))
model2.load_state_dict(torch.load("/data/kitouni/LWOT/MNIST/MLP/checkpoints/MLP_9000.pt"))

model1 = model1.to(DEV)
model2 = model2.to(DEV)

model1.eval()
model2.eval()

for x, y in val_dataset:
    x = x.to(DEV)
    y = y.to(DEV)
    y1 = model1(x)
    y2 = model2(x)
    # cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    print(cos(y1, y2))



# %%
