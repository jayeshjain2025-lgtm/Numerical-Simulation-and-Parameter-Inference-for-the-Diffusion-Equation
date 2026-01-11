import numpy as np

def solve_diffusion_1d(D, L = 1.0, T = 0.1, Nx = 100, Nt = 500):
    """
    D_true = 0.2

    Simulates 1D diffusion over a fixed spatial domain and time period.

    :param D: diffusion coefficient controlling how fast values spread
    :param L: total length of the 1D spatial domain
    :param T: total simulation time
    :param Nx: number of spatial grid points
    :param Nt: number of time steps
    """

    # spatial and time step sizes
    dx = L / (Nx - 1)
    dt = T / Nt

    # spatial grid
    x = np.linspace(0, L, Nx)

    # current state values at each grid point
    u = np.zeros(Nx)
    u[Nx // 2] = 1.0  # single spike to observe spreading clearly

    # stability-related parameter
    r = D * dt / (dx ** 2)
    if r > 0.5:
        raise ValueError("Stability condition not satisfied.")


    # store solution at each time step
    history = [u.copy()]

    for _ in range(Nt):
        # compute next state without overwriting current values
        u_new = u.copy()

        # update interior points only
        for j in range(1, Nx - 1):
            u_new[j] = u[j] + r * (u[j+1] - 2*u[j] + u[j-1])

        # no-flux boundary conditions (mirror the nearest interior values)
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]

        # advance in time and store snapshot
        u = u_new
        history.append(u.copy())

    # return grid and full time evolution for analysis and plotting
    return x, history