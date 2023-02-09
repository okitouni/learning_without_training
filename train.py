import torch
import os
import wandb
from config import parse_arguments, get_qualified_name, get_root
from train_cifar import train_cifar
from train_mnist import train_mnist
from train_modular_addition import train_modular_addition

# load DS from env
DS = os.environ.get("DS", "MNIST")

args = parse_arguments(DS)

torch.set_float32_matmul_precision('high')
DEV = torch.device(args.DEV) if torch.cuda.is_available() else torch.device("cpu")


name = get_qualified_name(DS, args)
root = get_root(DS)
basedir = os.path.join(root, name, "checkpoints")

if args.WANDB:
    wandb.init(project=f"LWOT", entity="iaifi", name=name)
    wandb.config = vars(args)
    os.makedirs(basedir, exist_ok=True)
    wandb.save(__file__)

torch.manual_seed(args.SEED)
device = torch.device(args.DEV) if torch.cuda.is_available() else torch.device("cpu")

print(f"training run for {name}")

if DS == "MNIST":
  train_mnist(args, device, basedir)
elif DS == "CIFAR":
  train_cifar(args, device, basedir)
elif DS == "MODULAR_ADDITION":
  train_modular_addition(args, device, basedir)
