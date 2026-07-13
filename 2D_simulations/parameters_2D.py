# Inside parameters_2D.py

Ny = 100
Nx = 100
dx = 1.0
steps = 5000               # Matched to YAML
dt = 0.01
save_every = 200
stopping_threshold = 1e-4
min_steps = 1000           # Matched to YAML

spike_value = 1.0          # Matched to YAML
noise_amplitude = 0.0
nucleation_rate = 0.0

n_points = 20

init_mode = "random_uniform_over0"
activator_type = "juxtacrine"

params = {
    "act_half_sat": 1.0,      
    "inh_half_sat": 1.0,      
    "act_decay_rate": 1.0,    
    "basal_prod": 0.0,        
    
    # --- CRITICAL CHANGES FROM YOUR YAML ---
    "act_diffusion": 1.0,      # TURNED ON (Allows stripes to stretch)
    "inh_diffusion": 10.0,     # Set to 10.0
    
    "act_prod_rate": 5.0,    
    "inh_prod_rate": 1.0,      # Set to 1.0 (Nicole's βi = 1 stripe regime)
    "inh_decay_rate": 0.5,    

    "act_hill_coeff": 3,       # Softened boundaries to allow merging
    "inh_hill_coeff": 3,       # Softened boundaries to allow merging
    # ---------------------------------------
}