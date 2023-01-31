import argparse

Hyperparameters = dict(
    EPOCHS=[10000],
    SCALE=[1],
    WIDTH=[512],
    MASK_SEED=[0, 1],
    SEED=[0, 1],
    LR=[1e-4],
    WD=[0., 5e-7],
    TAU=[1.0],
    ALPHA=[1.0],
    BATCHSIZE=[-1],
    DROPOUT=[0],
    BN=["None"],
)


def get_parser():
    parser = argparse.ArgumentParser()
    for k, v in Hyperparameters.items():
        parser.add_argument(f"--{k}", type=type(v[0]), default=v[0]) # TODO review float

    # operations params
    parser.add_argument("--DEV", type=str, default="cuda:0")
    parser.add_argument("--wandb", action="store_true", default=False)
    return parser


name = "".join(
    [
        "MLP",
        "_width{WIDTH}",
        "_maskseed{MASK_SEED}",
        "_seed{SEED}",
        "_lr{LR}",
        "_wd{WD}",
        "_tau{TAU}",
        "_alpha{ALPHA}",
        "_batchsize{BATCHSIZE}",
        "_dropout{DROPOUT}",
        "_bn{BN}",
    ]
)
root = "/data/kitouni/LWOT/MNIST/MLP/"

def format_name(args):
    return name.format(**vars(args))

def train_cmd(
    hyperparams,
):
    return " ".join(
        [
            f"python train.py",
            f"--wandb", # or not
        ]
        + [f"--{k} {v}" for k, v in hyperparams.items()]
    )

