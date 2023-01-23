import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from lwot.models import GEMLinear, Scale
from lwot.utils import Loader, accuracy
import tqdm
import wandb
import os
from scipy.optimize import linear_sum_assignment
import numpy as np

def activation_matching(a1, a2):
  # permute a2 to match a1 as close as possible in terms of frobenius norm
  # a1 and a2 are of shape (n_training_points, features)
  # returns a2 permuted
  a1 = a1.detach().cpu().numpy()
  a2 = a2.detach().cpu().numpy()
  cost = np.linalg.norm(a1[:, :, None] - a2[:, None, :], axis=1)
  row_ind, col_ind = linear_sum_assignment(cost)
  return a2[:, col_ind]

EPOCHS = 10000
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

name = f"{WIDTH}_tau{TAU}_scale{SCALE}"
if WANDB:
    wandb.init(project=f"LWOT", entity="iaifi", name=name)
    wandb.config = {
        "learning_rate": LR,
        "weight_decay": WD,
        "width": WIDTH,
        "depth": 3,
        "epochs": EPOCHS,
        "batch_size": BATCHSIZE,
        "model": "MLP",
        "optimizer": "Adam",
        "loss": "CrossEntropy",
        "dataset": "MNIST",
        "activation": "ReLU",
    }
    root = "/data/kitouni/LWOT/MNIST/MLP/{name}/"
    os.makedirs(root + "checkpoints", exist_ok=True)
    wandb.save(__file__)


torch.manual_seed(0)
train_dataset = MNIST(root='/data/ml_data', train=True, download=True)
val_dataset = MNIST(root='/data/ml_data', train=False, download=True)
train_dataset.data = train_dataset.data.float() / 255
val_dataset.data = val_dataset.data.float() / 255

trainloader = Loader(train_dataset, batch_size=BATCHSIZE, device=DEV)
valloader = Loader(val_dataset, batch_size=-1, device=DEV)



model = nn.Sequential(
    nn.Flatten(),
    Scale(SCALE, train=True, use_sigmoid=True),
    GEMLinear(28*28, WIDTH, threshold=0.5, train_weights=False),
    nn.ReLU(),
    GEMLinear(WIDTH, WIDTH, threshold=0.5, train_weights=False),
    nn.ReLU(),
    GEMLinear(WIDTH, 10, threshold=0.5, train_weights=False))

model.to(DEV)

criterion_ = nn.CrossEntropyLoss()
criterion = lambda output, target: criterion_(output * TAU, target) 
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

pbar = tqdm.tqdm(range(EPOCHS))

valloss, valacc = -1, -1
for epoch in pbar:
    model.train()
    for data, target in trainloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        acc = accuracy(output, target) 
        optimizer.step()
        msg = f'Loss: {loss.item():.2f}|{valloss:.2f} - Acc: {acc:.1f}|{valacc:.1f}'
        sparsities = [f"{module.sparsity()*100:.1f}" for module in model if hasattr(module, 'sparsity')]
        msg += f' Sparsities: {sparsities}'
        pbar.set_description(msg)

    model.eval()
    with torch.no_grad():
        for data, target in valloader:
            output = model(data)
            valloss = criterion(output, target).item()
            valacc = accuracy(output, target)


    if WANDB: wandb.log({"loss": loss, "acc": acc, "valloss": valloss, "valacc": valacc})
    if WANDB:
        if epoch % (EPOCHS//20) == 0 or epoch == EPOCHS-1:
            torch.save(model.state_dict(),
                       root + f"checkpoints/MLP_{epoch}.pt")
            wandb.save(root+ f"checkpoints/MLP_{epoch}.pt", base_path=root)