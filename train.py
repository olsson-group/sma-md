import torch
from sma_md import SMA_MD
from parameters import params

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

params = Struct(**params)

if torch.cuda.is_available():
    gpu_device = torch.cuda.get_device_name(0)
    print(f"* Running on a {gpu_device}.", flush=True)

model = SMA_MD(params)
model.build_model()

model.train(save_every_epoch = False)