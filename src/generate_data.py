import solver_1d
import numpy as np

D_true = 0.2
noise_level = 0.07
time_stamps = [50, 100, 200, 300, 400, 500]
x, history = solver_1d.solve_diffusion_1d(D_true)

clean_profile = [history[i] for i in time_stamps]
noisy_profile = [history[i] + noise_level * np.random.randn(len(x)) for i in time_stamps]

clean_profiles = np.array(clean_profile)
noisy_profiles = np.array(noisy_profile)
time_stamps = np.array(time_stamps)

np.savez(
    "data/synthetic_observations.npz",
    x=x,
    times=time_stamps,
    observations=noisy_profiles,
    clean=clean_profiles
)

data = np.load("data/synthetic_observations.npz")

print(data["x"].shape)            # (Nx,)
print(data["times"])              # [50 100 200 ...]
print(data["observations"].shape) # (num_times, Nx)
