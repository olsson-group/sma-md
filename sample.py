import os
import sys
import time
import torch
import numpy as np
from sma_md import SMA_MD
from parameters import params

if torch.cuda.is_available():
    gpu_device = torch.cuda.get_device_name(0)
    print(f"* Running on a {gpu_device}.", flush=True)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

params = Struct(**params)

model = SMA_MD(params)
model.build_model()

model_name = sys.argv[2] 
state_dict = torch.load("models/"+model_name, map_location=torch.device('cpu'))
model.rb_model.load_state_dict(state_dict)
model.params.model_name = model_name

#sampling parameters
mol_indices = np.load(sys.argv[1])
diffusion_steps = 20
n_confs = 10000
compute_log_prob = True
log_prob_components = True
ode = True
max_gen_batch_size = 2500
time_profile = True
save_total_time = True

directories = ["sampled_data", "sampled_data/conformations", "sampled_data/log_prob", "sampled_data/log_prob/"+ model_name,
               "sampled_data/conformations/gen_model", "sampled_data/conformations/gen_model/"+ model_name]
for dir in directories:
    if not os.path.isdir(dir):
        os.mkdir(dir)
    
if save_total_time:
    directories = ["sampled_data/performance", "sampled_data/performance/time_profile", "sampled_data/performance/time_profile/gen_model",
                   "sampled_data/performance/time_profile/gen_model/"+model_name]
    for dir in directories:
        if not os.path.isdir(dir):
            os.mkdir(dir)
        
#sampling

for i_mol in mol_indices:
    dataset_idx = i_mol
    print(f"Sampling molecule {i_mol}...",end=" ", flush=True)
    gen_batch_size = max_gen_batch_size
    finished = False

    while gen_batch_size > 0 and not finished:
        conformers, log_prob = model.generate_conformations(dataset_idx, n_confs, gen_batch_size, device = params.device, compute_log_prob = compute_log_prob,
                                                            ode = ode, time_profile = time_profile, diffusion_steps = diffusion_steps)
        try:
            if compute_log_prob:
                t0 = time.time()
                conformers, log_prob = model.generate_conformations(dataset_idx, n_confs, gen_batch_size, device = params.device, compute_log_prob = compute_log_prob,
                                                                    ode = ode, time_profile = time_profile, diffusion_steps = diffusion_steps)
                t1 = time.time()
                print(t1-t0, " s.", flush=True)            
            else:
                conformers = model.generate_conformations(dataset_idx, n_confs, gen_batch_size, device = params.device, compute_log_prob = compute_log_prob,
                                                          ode = ode, time_profile = time_profile, diffusion_steps = diffusion_steps)
            conf_filename = "sampled_data/conformations/gen_model/"+ model_name + "/{:0=5}".format(dataset_idx)+".npy"
            
            np.save(conf_filename, conformers)
            if compute_log_prob:
                log_prob_filename = "sampled_data/log_prob/"+ model_name +"/{:0=5}".format(dataset_idx)+".npy"
                np.save(log_prob_filename, log_prob)
                
            if save_total_time:
                np.save("sampled_data/performance/time_profile/gen_model/"+model_name+"/{:0=5}_".format(dataset_idx)+str(int(n_confs/1000))+"k.npy", t1-t0)
            finished = True
        except:
            gen_batch_size/= 2
            gen_batch_size = int(gen_batch_size)