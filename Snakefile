from pytools.persistent_dict import PersistentDict
from time import sleep
import os

import config

onstart:
  gpu_jobs = PersistentDict("jobs")
  gpu_jobs.store("gpu0", 0)
  gpu_jobs.store("gpu1", 0)
  print("gpu_jobs:", (gpu_jobs.fetch("gpu0"), gpu_jobs.fetch("gpu1")))
  sleep(.5)

def run_on_free_gpu(cmd, max_jobs_per_gpu=5):
  gpu_jobs = PersistentDict("jobs")
  while True:
    gc0, gc1 = (gpu_jobs.fetch("gpu0"), gpu_jobs.fetch("gpu1"))
    if not (gc0 >= max_jobs_per_gpu and gc1 >= max_jobs_per_gpu):
      cuda_id = 0 if gc0 <= gc1 else 1
      gpu_jobs.store(f"gpu{cuda_id}", gpu_jobs.fetch(f"gpu{cuda_id}") + 1)
      print(f"running on GPU {cuda_id}")
      shell(cmd + f" --DEV cuda:{cuda_id}")
      gpu_jobs.store(f"gpu{cuda_id}", gpu_jobs.fetch(f"gpu{cuda_id}") - 1)
      break
    sleep(15)
    print("waiting...")


class Locations:
  checkpoints_mnist = os.path.join(config.get_root("MNIST"), config.get_name("MNIST"), "checkpoints")
  checkpoints_cifar = os.path.join(config.get_root("CIFAR"), config.get_name("CIFAR"), "checkpoints")
  checkpoints_modular_addition = os.path.join(config.get_root("MODULAR_ADDITION"), config.get_name("MODULAR_ADDITION"), "checkpoints")

rule all:
  input:
    # expand(Locations.checkpoints_mnist,
    #         **config.Hyperparameters_MNIST),
    # expand(Locations.checkpoints_cifar,
    #         **config.Hyperparameters_CIFAR)
    expand(Locations.checkpoints_modular_addition,
            **config.Hyperparameters_MODULAR_ADDITION)


rule train_mnist:
  output:
    directory(Locations.checkpoints_mnist)
  run:
    cmd = config.train_cmd("MNIST", wildcards)
    run_on_free_gpu(cmd)

rule train_cifar:
  output:
    directory(Locations.checkpoints_cifar)
  run:
    cmd = config.train_cmd("CIFAR", wildcards)
    run_on_free_gpu(cmd)

rule train_modular_addition:
  output:
    directory(Locations.checkpoints_modular_addition)
  run:
    cmd = config.train_cmd("MODULAR_ADDITION", wildcards)
    print(cmd)
    run_on_free_gpu(cmd)
