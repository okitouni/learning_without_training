import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from lwot.models import get_model, GEMBase
from lwot.utils import Loader, accuracy
import tqdm
import wandb
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--EPOCHS", type=int, default=10000)
parser.add_argument("--DEV", type=str, default="cuda:0")
# MODEL PARAMS
parser.add_argument("--SCALE", type=int, default=1)
parser.add_argument("--WIDTH", type=int, default=512)
parser.add_argument("--MASK_SEED", type=int, default=0)
# parser.add_argument("--TW", type=bool, default=False)
parser.add_argument("--SEED", type=int, default=0)
parser.add_argument("--LR", type=float, default=1e-4)
parser.add_argument("--WD", type=float, default=5e-7)
parser.add_argument("--TAU", type=int, default=1)
parser.add_argument("--ALPHA", type=float, default=1)
parser.add_argument("--BATCHSIZE", type=int, default=-1)
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--DROPOUT", type=float, default=None)
parser.add_argument("--BN", type=str, default=None)
args = parser.parse_args()

torch.set_float32_matmul_precision('high')
EPOCHS = args.EPOCHS
DEV = torch.device(args.DEV) if torch.cuda.is_available() else torch.device("cpu")
# MODEL PARAMS
SCALE = args.SCALE
WIDTH = args.WIDTH
MASK_SEED = args.MASK_SEED
TW = MASK_SEED is None
# Loss PARAMS
SEED = args.SEED
LR = args.LR
WD = args.WD
TAU = args.TAU
ALPHA = args.ALPHA
BATCHSIZE = args.BATCHSIZE
WANDB = args.wandb
DROPOUT = args.DROPOUT  # applied after every layer except last. None for no droupout
BN = args.BN   # None for no batch norm "first" or "all" layers expect last

name = f"{WIDTH}_tau{TAU}_scale{SCALE}_ms{MASK_SEED}_seed{SEED}_wd{WD:.0e}_TW{TW}_bn{BN}_drp{DROPOUT}"
print("training run for", name)

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
    root = f"/data/kitouni/LWOT/MNIST/MLP/{name}/"
    os.makedirs(root + "checkpoints", exist_ok=True)
    wandb.save(__file__)

# LOADING DATA
torch.manual_seed(SEED)
train_dataset = MNIST(root='/data/ml_data', train=True, download=True)
val_dataset = MNIST(root='/data/ml_data', train=False, download=True)
train_dataset.data = train_dataset.data.float() / 255
val_dataset.data = val_dataset.data.float() / 255

trainloader = Loader(train_dataset, batch_size=BATCHSIZE, device=DEV)
valloader = Loader(val_dataset, batch_size=-1, device=DEV)


# SETTING UP MODEL
model_orig = get_model(width=WIDTH, depth=3, scale=SCALE, train_weights=TW, tau=TAU, dropout=DROPOUT, batchnorm=BN)

# Setting up seeded random mask
if MASK_SEED is not None: torch.manual_seed(MASK_SEED)
for module in model_orig.modules():
    if isinstance(module, GEMBase):
        if MASK_SEED is not None:
            module.weight_scores.data = torch.rand_like(module.weight_scores.data)
            if module.bias is not None:
                module.bias_scores.data = torch.rand_like(module.bias_scores.data)
        else:
            module.weight_scores.data = torch.ones_like(module.weight_scores.data)
            if module.bias is not None:
                module.bias_scores.data = torch.ones_like(module.bias_scores.data)
            module.train_scores(False) 
            module.train_weights(True)

# Setting up loss and optimizer
model_orig.to(DEV)
model = torch.compile(model_orig)
criterion_ = nn.MSELoss()
criterion = lambda output, target: criterion_(output, torch.nn.functional.one_hot(target, 10).float())
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

pbar = tqdm.tqdm(range(EPOCHS))

valloss, valacc = -1, -1

if WANDB:
    torch.save(model_orig, root + f"checkpoints/model_init.pt")
    wandb.save(root+ f"checkpoints/model_init.pt", base_path=root)

# TRAINING
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
        sparsities = [f"{module.sparsity()*100:.1f}" for module in model_orig if hasattr(module, 'sparsity')]
        msg += f' Sparsities: {sparsities}'
        pbar.set_description(msg)

    model.eval()
    with torch.no_grad():
        for data, target in valloader:
            output = model(data)
            valloss = criterion(output, target).item()
            valacc = accuracy(output, target)

    # WANDB LOGGING
    if WANDB:
        wandb.log({"loss": loss, "acc": acc, "valloss": valloss, "valacc": valacc})
        for name, module in model_orig.named_modules():
            if hasattr(module, 'sparsity'):
                sparsity = module.sparsity()
                wandb.log({f"sparsity_{name}": sparsity.item()})
        if epoch % (EPOCHS//20) == 0 or epoch == EPOCHS-1:
            torch.save(model_orig.state_dict(),
                       root + f"checkpoints/MLP_{epoch}.pt")
            wandb.save(root+ f"checkpoints/MLP_{epoch}.pt", base_path=root)
