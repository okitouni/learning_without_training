import argparse

Hyperparameters_MNIST = dict(
    EPOCHS=[10000],
    SCALE=[1],
    WIDTH=[512],
    MASK_SEED=[0, 1],
    SEED=[0, 1],
    LR=[1e-4],
    WD=[0.0, 5e-7],
    TAU=[1.0],
    ALPHA=[1.0],
    BATCHSIZE=[-1],
    DROPOUT=[0],
    BN=["None"],
)

Hyperparameters_CIFAR = dict(
    SEED=[0],
    EPOCHS=[100],
    BATCH_SIZE=[512],
    MOMENTUM=[0.9],
    WEIGHT_DECAY=[0.256],
    WEIGHT_DECAY_BIAS=[0.004],
    EMA_UPDATE_FREQ=[5],
    EMA_RHO=[0.99],
    SCALE_OUT=[0.125],
    ALPHA_SMOOTHING=[0.2],
    MASKING=[True],
    LR_MULTIPLIER=[1],
    LR_BIAS_MULTIPLIER=[64],
)


def get_parser(ds="MNIST"):
    parser = argparse.ArgumentParser()
    if ds == "MNIST":
        hyperparams = Hyperparameters_MNIST
    elif ds == "CIFAR":
        hyperparams = Hyperparameters_CIFAR
    for k, v in hyperparams.items():
        parser.add_argument(
            f"--{k}", type=type(v[0]), default=v[0]
        )  # TODO review float

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


def train_cmd(hyperparams,):
    return " ".join(
        [f"python train.py", f"--wandb",]  # or not
        + [f"--{k} {v}" for k, v in hyperparams.items()]
    )

