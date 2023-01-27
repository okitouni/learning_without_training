import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from lwot.models import get_model, GEMBase
from lwot.utils import Loader, accuracy
import tqdm
import wandb
import os

# CONFIG # i made a new file (match_mnist_models), how (from where) do i load two saved models? It's in rebasin.py
EPOCHS = 10000
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# MODEL PARAMS
SCALE = 1
WIDTH = 512
TW = True # train weights # OK I added Masked seed = None should lead to regular training # cool beans
MASK_SEED = None # none regular training
# Loss PARAMS
LR = 1e-3
WD = 1e-4
TAU = 1
ALPHA = 1
BATCHSIZE = -1
WANDB = True
SEED = 1
DROPOUT=None # applied after every layer except last. None for no droupout
BN=None # None for no batch norm "first" or "all" layers expect last

name = f"{WIDTH}_tau{TAU}_scale{SCALE}_ms{MASK_SEED}_seed{SEED}"
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
model = get_model(width=WIDTH, depth=3, scale=SCALE, train_weights=TW, tau=TAU, dropout=DROPOUT, batchnorm=BN)

# Setting up seeded random mask
if MASK_SEED is not None: torch.manual_seed(MASK_SEED)
for module in model.modules():
    if isinstance(module, GEMBase):
        if MASK_SEED is not None:
            module.weight_scores.data = torch.rand_like(module.weight_scores.data)
            if module.bias is not None:
                module.bias_scores.data = torch.rand_like(module.bias_scores.data)
        else:
            module.weight_scores.data = torch.ones_like(module.weight_scores.data)
            if module.bias is not None:
                module.bias_scores.data = torch.ones_like(module.bias_scores.data)
            module.train_scores(False) # so now we can just run weight training? I think so. Try it. will it save to a new dir by itself? The whole ass pipeline great lol
            module.train_weights(True)

# Setting up loss and optimizer
model.to(DEV)
criterion_ = nn.MSELoss()
criterion = lambda output, target: criterion_(output, torch.nn.functional.one_hot(target, 10).float())
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

pbar = tqdm.tqdm(range(EPOCHS))

valloss, valacc = -1, -1

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
        sparsities = [f"{module.sparsity()*100:.1f}" for module in model if hasattr(module, 'sparsity')]
        # We can check weights # the terminal got stuck for me # new terminal just dropped
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
        for name, module in model.named_modules():
            if hasattr(module, 'sparsity'):
                sparsity = module.sparsity()
                wandb.log({f"sparsity_{name}": sparsity.item()})
        if epoch % (EPOCHS//20) == 0 or epoch == EPOCHS-1:
            torch.save(model.state_dict(),
                       root + f"checkpoints/MLP_{epoch}.pt")
            wandb.save(root+ f"checkpoints/MLP_{epoch}.pt", base_path=root)
