import torch
import wandb
import os

def save_model(basedir, model, epoch):
    if epoch == 0:
        wandb.watch(model)
        torch.save(model, os.path.join(basedir, "model_init.pt"))
        wandb.save(os.path.join(basedir, "model_init.pt"), base_path=basedir)
    else:
        torch.save(model.state_dict(),
                  os.path.join(basedir, f"{epoch}.pt"))
        wandb.save(os.path.join(basedir, f"{epoch}.pt"), base_path=basedir)
