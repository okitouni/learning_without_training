import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from lwot.models import mlp, GEMBase
from lwot.utils import Loader, accuracy
import tqdm
import wandb
from utils import save_model


def train_mnist(args, device, basedir):

  TW = args.MASK_SEED is None
  # Loss PARAMS

  train_dataset = MNIST(root='/data/ml_data', train=True, download=True)
  val_dataset = MNIST(root='/data/ml_data', train=False, download=True)
  norm = train_dataset.data.max()
  train_dataset.data = train_dataset.data.float() / norm
  val_dataset.data = val_dataset.data.float() / norm

  trainloader = Loader(train_dataset, batch_size=args.BATCHSIZE, device=device)
  valloader = Loader(val_dataset, batch_size=-1, device=device)


  # SETTING UP MODEL
  model_orig = mlp(width=args.WIDTH, depth=3, scale=args.SCALE, train_weights=TW, tau=args.TAU, dropout=args.DROPOUT, batchnorm=args.BN)


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
  model_orig.to(device)
  model = torch.compile(model_orig)
  criterion_ = nn.MSELoss()
  criterion = lambda output, target: criterion_(output, torch.nn.functional.one_hot(target, 10).float())
  optimizer = torch.optim.Adam(model.parameters(), lr=args.LR, weight_decay=args.WD)

  pbar = tqdm.tqdm(range(args.EPOCHS))

  valloss, valacc = -1, -1

  if args.WANDB:
      save_model(basedir, model_orig, 0)

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
              save_model(basedir, model_orig, epoch)
