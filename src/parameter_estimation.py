import numpy as np
import matplotlib.pyplot as plt
import solver_1d

# load synthetic data once
data = np.load("data/synthetic_observations.npz")
observations = data["observations"]
times = data["times"]

def compute_error(D_guess):
    """
    Computes mean squared error between simulated profiles
    and noisy observed profiles for a given D_guess.
    """
    # run forward model
    x_guess, history_guess = solver_1d.solve_diffusion_1d(D_guess)

    diff = 0.0

    # loop over observed time indices
    for k in range(len(times)):
        t = times[k]

        u_obs = observations[k]
        u_guess = history_guess[t]

        diff += np.mean((u_obs - u_guess) ** 2)

    # average over all observed times
    return diff / len(times)


# -------------------------------
# Parameter search
# -------------------------------

# choose candidate D values BELOW stability limit
D_candidates = np.linspace(0.05, 0.25, 40)

errors = []

for D in D_candidates:
    try:
        err = compute_error(D)
        errors.append(err)
    except ValueError:
        # skip unstable D values
        errors.append(np.nan)

errors = np.array(errors)

# find best D (ignoring NaNs)
best_index = np.nanargmin(errors)
best_D = D_candidates[best_index]
best_error = errors[best_index]

print(f"Estimated D: {best_D}")
print(f"Minimum error: {best_error}")

# -------------------------------
# Plot error vs D
# -------------------------------

plt.plot(D_candidates, errors, marker="o")
plt.xlabel("D (diffusion coefficient)")
plt.ylabel("Mean Squared Error")
plt.title("Error vs Diffusion Coefficient")
plt.show()
