import torch
import torch.nn as nn
from torchvision.datasets import MNIST, CIFAR10
from lwot.models import get_model, GEMBase
from lwot.utils import Loader, accuracy
import tqdm
import wandb
import os
from config import root, format_name, get_parser

args = get_parser("MNIST").parse_args()

torch.set_float32_matmul_precision('high')
DEV = torch.device(args.DEV) if torch.cuda.is_available() else torch.device("cpu")
# MODEL PARAMS
TW = args.MASK_SEED is None
# Loss PARAMS

name = format_name(args)
basedir = os.path.join(root, name, "checkpoints")

print("training run for", name)

if args.WANDB:
    wandb.init(project=f"LWOT", entity="iaifi", name=name)
    wandb.config = {
        "learning_rate": args.LR,
        "weight_decay": args.WD,
        "width": args.WIDTH,
        "depth": 3,
        "epochs": args.EPOCHS,
        "batch_size": args.BATCHSIZE,
        "model": "MLP",
        "optimizer": "Adam",
        "loss": "CrossEntropy",
        "dataset": "MNIST",
        "activation": "ReLU",
    }
    os.makedirs(basedir, exist_ok=True)
    wandb.save(__file__)

# LOADING DATA
torch.manual_seed(args.SEED)

DATA = CIFAR10 if args.DATASET == "CIFAR10" else "MNIST"

train_dataset = DATA(root='/data/ml_data', train=True, download=True)
val_dataset = DATA(root='/data/ml_data', train=False, download=True)
norm = train_dataset.data.max()
train_dataset.data = train_dataset.data.float() / norm
val_dataset.data = val_dataset.data.float() / norm

trainloader = Loader(train_dataset, batch_size=args.BATCHSIZE, device=DEV)
valloader = Loader(val_dataset, batch_size=-1, device=DEV)


# SETTING UP MODEL
model_orig = get_model(width=args.WIDTH, depth=3, scale=args.SCALE, train_weights=TW, tau=args.TAU, dropout=args.DROPOUT, batchnorm=args.BN)


# Setting up seeded random mask
if args.MASK_SEED is not None: torch.manual_seed(args.MASK_SEED)
for module in model_orig.modules():
    if isinstance(module, GEMBase):
        if args.MASK_SEED is not None:
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
optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.WD)

pbar = tqdm.tqdm(range(args.EPOCHS))

valloss, valacc = -1, -1

if args.WANDB:
    torch.save(model_orig, os.path.join(basedir, "model_init.pt"))
    wandb.save(os.path.join(basedir, "model_init.pt"), base_path=basedir)

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
    if args.WANDB:
        wandb.log({"loss": loss, "acc": acc, "valloss": valloss, "valacc": valacc})
        for name, module in model_orig.named_modules():
            if hasattr(module, 'sparsity'):
                sparsity = module.sparsity()
                wandb.log({f"sparsity_{name}": sparsity.item()})
        if epoch % (args.EPOCHS//20) == 0 or epoch == args.EPOCHS-1:
            torch.save(model_orig.state_dict(),
                       os.path.join(basedir, f"{epoch}.pt"))
            wandb.save(os.path.join(basedir, f"{epoch}.pt"), base_path=basedir)
