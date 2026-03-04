import numpy as np
from numba import njit
import argparse
import matplotlib.pyplot as plt
# import csv
from matplotlib.colors import ListedColormap
# from scipy.optimize import curve_fit

custom_cmap = ListedColormap(['mintcream', 'tomato', 'darkgreen'])

plt.rcParams.update({ # Use LaTeX for text rendering
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})

running = True # This is to handle exiting the code when the figure is closed
def on_close(event):
    global running
    running = False


def init_sirs(N=50, i=0.333, r=0.333):
    lattice = np.random.rand(N, N)
    
    i_mask = (lattice < i)
    r_mask = (lattice > (1-r))
    s_mask = ~(i_mask + r_mask)

    lattice[s_mask] = 0
    lattice[i_mask] = 1
    lattice[r_mask] = 2

    return lattice
    
def sirs_update_full_lattice(lattice, psi, pir, prs): # Incorrect implementation, do not use!!!

    N = lattice.shape[0]

    s_mask = (lattice == 0)
    i_mask = (lattice == 1)
    r_mask = (lattice == 2)

    temp_lattice = np.random.rand(N, N)

    temp_lattice[i_mask] = np.where(temp_lattice[i_mask] <= pir, 2, 1)
    temp_lattice[r_mask] = np.where(temp_lattice[r_mask] <= prs, 0, 2)

    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    i_neighbours = np.zeros(lattice.shape)

    for dir in directions:
        i_neighbours += np.roll(i_mask, dir, axis=(0, 1))

    at_risk = ((i_neighbours >= 1) & s_mask)
    safe = (~at_risk & s_mask)

    temp_lattice[at_risk] = np.where(temp_lattice[at_risk] <= psi, 1, 0)
    temp_lattice[safe] = 0

    return temp_lattice.copy()

# def sirs_update(lattice, psi, pir, prs):

#     N = lattice.shape[0]
#     new_lattice = lattice.copy()

#     for _ in range(N**2):

#         i, j = np.random.randint(0, N, size=2)
#         p = np.random.rand()
#         state = new_lattice[i, j]

#         if state == 0: # Chosen site is susceptible

#             neighbours = sum([
#                 lattice[(i-1) % N, j],
#                 lattice[(i+1) % N, j],
#                 lattice[i, (j-1) % N],
#                 lattice[i, (j+1) % N]
#             ])

#             if neighbours >= 1 and p <= psi:
#                 new_lattice[i, j] = 1

#         elif state == 1 and p <= pir: # Chosen site is infected
#             new_lattice[i, j] = 2

#         elif state == 2 and p <= prs: # Chosen site is recovered
#             new_lattice[i, j] = 0
        
#     return new_lattice

@njit
def sirs_update(lattice, psi, pir, prs):

    N = lattice.shape[0]

    for _ in range(N * N): # Runs a whole sweep (no need for control over single site updates)

        i = np.random.randint(0, N)
        j = np.random.randint(0, N)

        state = lattice[i, j]

        if state == 0:

            i_neighbour = (
                lattice[(i-1) % N, j] == 1 or
                lattice[(i+1) % N, j] == 1 or
                lattice[i, (j-1) % N] == 1 or
                lattice[i, (j+1) % N] == 1
            )

            if i_neighbour:
                if np.random.rand() < psi:
                    lattice[i, j] = 1

        elif state == 1:
            if np.random.rand() < pir:
                lattice[i, j] = 2

        elif state == 2:
            if np.random.rand() < prs:
                lattice[i, j] = 0

    return lattice



def sirs(N=50, pi=0.1, pr=0.3, ps=0.3):

    lattice = init_sirs(N)
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", on_close) # Handles closing figure
    img = ax.imshow(lattice, cmap=custom_cmap, vmin=0, vmax=2)
    ax.set_title("SIRS")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    while running:
        lattice = sirs_update(lattice, psi=pi, pir=pr, prs=ps)
        img.set_data(lattice)
        plt.pause(0.05)
    plt.ioff()

def phase(N):
    t_eq = 100
    n_meas = 1000
    grid_spacing = 0.05

    n_sites = N**2

    pir = 0.5
    prs_vals = np.arange(0, 1, grid_spacing)
    psi_vals = np.arange(0, 1, grid_spacing)

    means = np.empty([prs_vals.shape[0], psi_vals.shape[0]], dtype=np.float64)
    variances = np.empty([prs_vals.shape[0], psi_vals.shape[0]], dtype=np.float64)

    for x, prs in enumerate(prs_vals):
        print(f'{prs*100}%')
        for y, psi in enumerate(psi_vals):
            fracs = np.empty([n_meas], dtype=np.float64)

            lattice = init_sirs(N)

            for _ in range(t_eq):
                lattice = sirs_update(lattice, psi, pir, prs)


            for i in range(n_meas):
                lattice = sirs_update(lattice, psi, pir, prs)
                fracs[i] = np.sum((lattice == 1)) / n_sites
            means[x, y] = np.mean(fracs)
            variances[x, y] = np.var(fracs)

    np.savetxt("sirs means.csv", means, delimiter=",")
    np.savetxt("sirs variances.csv", variances, delimiter=",")

def graph_sirs():
    try:
        means = np.loadtxt("sirs means.csv", delimiter=",")
        variances = np.loadtxt("sirs variances.csv", delimiter=",")
    except:
        print("Non-existent files, run phase() first")

    plt.imshow(means, cmap='plasma', extent=(0, 1, 0, 1), origin='lower', aspect='equal')
    plt.ylabel(r"P(r$\rightarrow$s)")
    plt.xlabel(r"P(s$\rightarrow$i)")
    cbar = plt.colorbar()
    cbar.set_label(r"$\langle I \rangle / N$", rotation='horizontal')
    plt.savefig("sirs means.png", dpi=300, bbox_inches='tight')
    plt.close()

    plt.imshow(variances, cmap='plasma', extent=(0, 1, 0, 1), origin='lower', aspect='equal')
    plt.ylabel(r"P(r$\rightarrow$s)")
    plt.xlabel(r"P(s$\rightarrow$i)")
    cbar = plt.colorbar()
    cbar.set_label(r"$\sigma^2(\langle I \rangle / N)$", rotation='horizontal')
    plt.savefig("sirs variances.png", dpi=300, bbox_inches='tight')
    plt.close()

def var_cut(N):
    n_sites = N**2
    t_eq = 500
    n_meas = 10000
    pir = 0.5
    prs = 0.5
    psi_vals = np.linspace(0.2, 0.5, 50)
    var_list = np.empty_like(psi_vals)
    err_list = np.empty_like(psi_vals)

    for i, psi in enumerate(psi_vals):
        print(f'{2*i}%')

        lattice = init_sirs(N)
        for _ in range(t_eq):
            lattice = sirs_update(lattice, psi, pir, prs)

        fracs = np.empty([n_meas], dtype=np.float64)
        for j in range(n_meas):
            lattice = sirs_update(lattice, psi, pir, prs)
            fracs[j] = np.sum((lattice == 1)) / n_sites

        var_list[i] = np.var(fracs)
        err_list[i] = np.std(fracs)

    plt.plot(psi_vals, var_list)
    # plt.errorbar(psi_vals, var_list, err_list)
    plt.show()


if __name__ == "__main__":

    # Handle command line arguments
    parser = argparse.ArgumentParser(description="SIRS Model")

    parser.add_argument("-a", "--action", help="What to do with model: 'animate' (animation) or 'phase' (calculate phases) or 'graph' (create equilibration graph, requires .csv files from 'phase'). Default='animate'", type=str, default='animate')
    parser.add_argument("-L", "--length", help="LxL size of lattice. Default=50", type=int, default=50)
    parser.add_argument("-pi", help="Probability of infection p(s->i)", type=float, default=0.3)
    parser.add_argument("-pr", help="Probability of recovery p(i->r)", type=float, default=0.3)
    parser.add_argument("-ps", help="Probability of becoming susceptible p(r->s)", type=float, default=0.3)

    
    args = parser.parse_args()
    action = args.action
    L = args.length
    pi = args.pi
    pr = args.pr
    ps = args.ps

    if action == 'animate':
        sirs(L, pi, pr, ps)
    elif action == 'phase':
        phase(L)
    elif action == 'graph':
        graph_sirs()
    elif action == 'variance_plot':
        var_cut(L)
    
