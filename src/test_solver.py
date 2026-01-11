import matplotlib.pyplot as plt
import solver_1d

x, history = solver_1d.solve_diffusion_1d(D=0.2)
time_indices = [0, 10, 50, 100, 200, 300, 400, 500]
for t in time_indices:
    plt.plot(x, history[t], label=f'Time step {t}')
plt.legend()
plt.show()