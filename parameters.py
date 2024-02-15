import os

# define transBG-specific parameters
params = {

    "model_name" : "test",
    "device" : "cuda:0",

    "random_seed" : 0,

    # Dataset parameters
    "sdf_path" : "datasets/mdqm9-nc/mdqm9-nc.sdf",
    "hdf5_path" : "datasets/mdqm9-nc/mdqm9-nc.hdf5",
    "aux_hdf5_path" : "datasets/auxiliary_datasets/aux-mdqm9-nc.hdf5",
    "splits_path" : "datasets/mdqm9-nc/splits/",

    # Training parameters
    "val_set_size" : 0.2,
    "test_set_size" : 0.1,

    #rotable bonds model
    "conf_batch_size" : 5, #5, number of conformations per molecule in a batch
    "mol_batch_size" : 256, #256, number of molecules in a batch
    "rb_epochs" : 750, #750
    "rb_learning_rate": 1e-3, #1e-3
    "rb_lr_factor" : 0.7, # 0.7
    "rb_scheduler_patience": 20, #20
    "rb_min_rel_lr" : 1./100, # 1./100
    "rb_weight_decay" : 0, # 0 
    "rb_num_workers" : 1, #1
    "sigma_min": 0.01*3.14, #0.01*3.14
    "sigma_max": 3.14, #3.14
    "limit_train_mols": 0,
    "boltzmann_weight": False, #False
    "rb_max_conformations": int(16e3),

    #Torsional diffusion parameters

    #model arguments
    "num_conv_layers" : 4, #4
    "max_radius": 5.0, #5.0 nm
    "scale_by_sigma": True,
    "ns": 32, #32
    "nv": 8, #8
    "no_residual": False,
    "no_batch_norm": False,
    "use_second_order_repr": False,
    "inference_steps": 20,

    #feature arguments
    "in_node_features" : 44,
    "in_edge_features" : 4,
    "sigma_embed_dim" : 32,
    "radius_embed_dim" : 50,
    "max_conformations" : int(16e3), #what is this?
}