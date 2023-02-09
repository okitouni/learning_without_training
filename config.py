import argparse

WANDB=False

# WARNING use only float/int/str here, bools will fail!

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
    EPOCHS=[30],
    BATCH_SIZE=[512],
    MOMENTUM=[0.9],
    WEIGHT_DECAY=[0.256],
    WEIGHT_DECAY_BIAS=[0.004],
    EMA_UPDATE_FREQ=[5],
    EMA_RHO=[0.99],
    SCALE_OUT=[0.125],
    ALPHA_SMOOTHING=[0.2],
    MASKING=[0],
    LR_MULTIPLIER=[1],
    LR_BIAS_MULTIPLIER=[64],
)

Hyperparameters_MODULAR_ADDITION = dict(
    SEED=[0],
    EPOCHS=[10000],
    D_MODEL=[64],
    WEIGHT_DECAY=[1e-3],
    MASKING=[1], # boolean, but snakemake hates that
    LR=[1e-2],
    MODULO=[53],
)

def get_hyperparams(ds):
  assert ds in ["MNIST", "CIFAR", "MODULAR_ADDITION"]
  return eval(f"Hyperparameters_{ds}")

def parse_arguments(ds="MNIST"):
    parser = argparse.ArgumentParser()
    hyperparams = get_hyperparams(ds)
    for k, v in hyperparams.items():
        parser.add_argument(
            f"--{k}", type=type(v[0]), default=v[0]
        )  # TODO review float

    # operations params
    parser.add_argument("--DEV", type=str, default="cuda:0")
    parser.add_argument("--WANDB", action="store_true", default=False)
    return parser.parse_args()


def make_suffix_for(arg: str):
    """
    define the name suffixes when adding an arg, eg mask_seed -> _maskseed{MASKSEED}
    this suffix has unfilled format braces, to be filled by the get_qualifed_name function
    """
    return f"_{arg.lower().replace('_', '')}{{{arg}}}"


def get_name(task="MNIST"):
    hyperparams = get_hyperparams(task)
    name = "".join(
        [
            "MLP",
            *[make_suffix_for(hp) for hp in hyperparams.keys()],
        ]
    )
    return name


def get_root(task="MNIST"):
    return f"/data/kitouni/LWOT/{task}/MLP"


def get_qualified_name(task, args):
    """
    task in ["MNIST", "CIFAR", "MODULAR_ADDITION"]
    """
    name = get_name(task)
    return name.format(**vars(args))


def train_cmd(
    task = "MNIST",
    hyperparams = {}, # filled by snakemake wildcards
):
    return " ".join(
        [
            f"DS={task}",
            f"python train.py",
            f"--wandb" if WANDB else "",
        ]  # or not
        + [f"--{k} {v}" for k, v in hyperparams.items()]
    )
