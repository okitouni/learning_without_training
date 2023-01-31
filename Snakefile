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
  checkpoints = os.path.join(config.root, config.name, "checkpoints")

rule many_seeds:
  input:
    expand(Locations.checkpoints,
            **config.Hyperparameters),


rule train_many_seeds:
  output:
    directory(Locations.checkpoints)
  run:
    cmd = config.train_cmd(wildcards)
    run_on_free_gpu(cmd)
