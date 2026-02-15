import numpy as np
import pandas as pd
from numba import njit, prange
import time
import argparse
from random import random
import matplotlib.pyplot as plt
color = plt.get_cmap("summer")(0.)


running = True # This is to handle exiting the code when the figure is closed
def on_close(event):
    global running
    running = False

def init_lattice(N=50, p=0.5):
    return np.where(np.random.rand(N, N) < p, 1, -1).astype(np.int8)

@njit
def total_energy(lattice):
    N = lattice.shape[0]
    energy = 0.
    for i in range(N):
        for j in range(N):
            S = lattice[i, j]

            # sum all values of neighbours, note %N ensures rollover for edge cases (open boundaries)
            neighbor_total = lattice[(i+1)%N, j] + lattice[i, (j+1)%N] + lattice[(i-1)%N, j] + lattice[i, (j-1)%N]

            energy += -S * neighbor_total

    E = energy / 2  # Correction because each pair is counted twice

    return E

@njit
def single_site_dE(lattice, i, j):
        # Energy for spin flip
        N = lattice.shape[0]
        S = lattice[i, j]
        neighbor_total = lattice[(i+1)%N, j] + lattice[i, (j+1)%N] + lattice[(i-1)%N, j] + lattice[i, (j-1)%N]

        return 2 * S * neighbor_total # Always true, dE is just a function of initial energy at site, draw all flip cases for intuition

@njit
def nearest_neighbours_dE(lattice, i1, j1, i2, j2):
    N = lattice.shape[0]

    S1 = lattice[i1, j1]
    S2 = lattice[i2, j2]

    #                                                                       The correction for NN case is here vvvvvvvvvvvvvvvvv
    neighbours1 = lattice[(i1+1)%N, j1] + lattice[i1, (j1+1)%N] + lattice[(i1-1)%N, j1] +lattice[i1, (j1-1)%N] - lattice[i2, j2]
    neighbours2 = lattice[(i2+1)%N, j2] + lattice[i2, (j2+1)%N] + lattice[(i2-1)%N, j2] +lattice[i2, (j2-1)%N] - lattice[i1, j1]

    return 2 * S1 * neighbours1 + 2 * S2 * neighbours2


@njit
def glauber_update(lattice, T, n_update=1):
        # Runs n_step Glauber-Metropolis updates

        N = lattice.shape[0]

        for _ in range(n_update):

            # Choose lattice point and find the spin-flip energy difference
            i, j = np.random.randint(0, N, size=2)
            dE = single_site_dE(lattice, i, j)

            if dE <= 0:
                # Always flip if energetically favourable
                lattice[i, j] *= -1

            else:
                p = np.exp(-dE / T) # Probability of flip

                # Only flips sometimes
                if np.random.random() < p:
                    lattice[i, j] *= -1
                else:
                    pass

@njit
def kawasaki_update(lattice, T, n_update=1):
    # Runs n_step Kawasaki updates

    N = lattice.shape[0]

    for _ in range(n_update):
    
        i1, j1, i2, j2 = np.random.randint(0, N, size=4)
    
        # Avoids having identical lattice sites (allows neighbours though)
        while i1 == i2 and j1 == j2:
            i1, j1, i2, j2 = np.random.randint(0, N, size=4)

        S1 = lattice[i1, j1]
        S2 = lattice[i2, j2]

        if S1 != S2: # Don't bother swapping flip states if they're the same

            # Nearest neighbours detection
            if ( (np.abs(i1-i2)==1 or np.abs(i1-i2)==N-1) and j1==j2 ) or\
                  ( (np.abs(j1-j2)==1 or np.abs(j1-j2)==N-1) and i1==i2 ):
                
                dE = nearest_neighbours_dE(lattice, i1, j1, i2, j2)

            else:
                dE = single_site_dE(lattice, i1, j1) + single_site_dE(lattice, i2, j2)


            if dE <= 0: # Flip if energetically favourable
                lattice[i1, j1] *= -1
                lattice[i2, j2] *= -1

            else:
                p = np.exp(-dE / T) # Probability of flip

                # Only flips sometimes
                if np.random.random() < p:
                    lattice[i1, j1] *= -1
                    lattice[i2, j2] *= -1

@njit
def calc_chi(M, N, T):
    return (np.mean(M**2) - np.mean(M)**2) / (N**2 * T)

@njit
def calc_c(E, N, T): # Note this returns in units of k_b
    return (np.mean(E**2) - np.mean(E)**2) / (N**2 * T**2)


def bootstrap(sample, function, k, *args):

    n = len(sample)

    vals = np.empty(k, dtype=np.float64)

    for j in range(k):

        indices = np.random.random_integers(0, n-1, size=n)

        vals[j] = function(sample[indices], *args)

    return np.sqrt(np.mean(vals**2) - np.mean(vals)**2)


def plot(N, T, method: str):
    # Plots model updates

    lattice = init_lattice(N)
    sweep_increment = 1
    iteration_step = (N**2) * sweep_increment # Sweep size

    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", on_close) # Handles closing figure
    img = ax.imshow(lattice, cmap="summer", vmin=-1, vmax=1)

    plt.suptitle(f'{method.capitalize()} dynamics')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    n_sweeps = 0
    while running:
        n_sweeps += sweep_increment

        if method == 'glauber':
            glauber_update(lattice=lattice, T=T, n_update=iteration_step)
        elif method == 'kawasaki':
            kawasaki_update(lattice=lattice, T=T, n_update=iteration_step)

        img.set_data(lattice)
        ax.set_title(f"Number of sweeps = {n_sweeps}")
        plt.pause(0.02)

    plt.ioff()

@njit
def run_and_record(method, N, T, n_measure, n_equi_sweeps, n_sweeps_per_measure):
    # Records magnetisation and energy for a given T

    lattice = np.where(np.random.rand(N, N) < 0.5, 1, -1).astype(np.int8)

    sweep_size = N ** 2
    
    # Find total number of individual updates before equilibration, then equilibrate
    n_equilibration = sweep_size * n_equi_sweeps
    if method == 0:
        glauber_update(lattice, T, n_update=n_equilibration)
    else:
        kawasaki_update(lattice, T, n_update=n_equilibration)


    M_list = np.empty(n_measure, dtype=np.float64)
    E_list = np.empty(n_measure, dtype=np.float64)
    list_pos = 0

    # Find the required number of individual updates for each step
    n_per_step = sweep_size * n_sweeps_per_measure
    for _ in range(n_measure):
        
        if method == 0:
            glauber_update(lattice, T, n_update=n_per_step)
        else:
            kawasaki_update(lattice, T, n_update=n_per_step)

        M_list[list_pos] = np.sum(lattice)        # Calculation for magnetisation
        E_list[list_pos] = total_energy(lattice)
        list_pos += 1

    return M_list, E_list

@njit(parallel=True)
def parallel_calc(method, N, T_list, n_measurements, t_equilibration, t_step):

    n_T = len(T_list)

    mag = np.empty(n_T, dtype=np.float64)
    chi = np.empty(n_T, dtype=np.float64)
    e = np.empty(n_T, dtype=np.float64)
    c = np.empty(n_T, dtype=np.float64)
    
    all_m = np.empty((n_T, n_measurements), dtype=np.float64)
    all_e = np.empty((n_T, n_measurements), dtype=np.float64)

    for i in prange(n_T):
        T = T_list[i]

        M_list, E_list = run_and_record(method, N, T, n_measurements, t_equilibration, t_step)

        all_m[i, :] = M_list
        all_e[i, :] = E_list

        mag[i] = np.mean(np.abs(M_list))
        chi[i] = calc_chi(M_list, N, T)
        e[i] = np.mean(E_list)
        c[i] = calc_c(E_list, N, T)


    return mag, chi, e, c, all_m, all_e

def errors(all_m, all_e, T_list, N, k=1000):

    n_T = len(T_list)

    sigma_mag = np.empty(n_T)
    sigma_chi = np.empty(n_T)
    sigma_e = np.empty(n_T)
    sigma_c = np.empty(n_T)

    for i, T in enumerate(T_list):
        sigma_mag[i] = bootstrap(all_m[i], np.mean, k)
        sigma_chi[i] = bootstrap(all_m[i], calc_chi, k, N, T)
        sigma_e[i] = bootstrap(all_e[i], np.mean, k)
        sigma_c[i] = bootstrap(all_e[i], calc_c, k, N, T)

    return sigma_mag, sigma_chi, sigma_e, sigma_c

def calc(method, N, n_measurements, t_equilibration, t_step):

    T_list = np.arange(1, 3.1, 0.1)

    method_int = 0 if method == "glauber" else 1

    m, chi, e, c, all_m, all_e = parallel_calc(method_int, N, T_list, n_measurements, t_equilibration, t_step)
    sigma_m, sigma_chi, sigma_e, sigma_c = errors(all_m, all_e, T_list, N)

    df = pd.DataFrame({'T': T_list,
                       'Magnetisation': m,
                       'Magnetisation Error': sigma_m,
                       'Susceptibility': chi,
                       'Susceptibility Error': sigma_chi,
                       'Energy': e,
                       'Energy Error': sigma_e,
                       'Heat Capacity': c,
                       'Heat Capacity Error': sigma_c})
    df.to_csv(f'{method.capitalize()} values.csv', index=False)

def figures(method):
    df = pd.read_csv(f'{method.capitalize()} values.csv')
    T_list = df['T']
    m = df['Magnetisation']
    chi = df['Susceptibility']
    e = df['Energy']
    c = df['Heat Capacity']
    sigma_m = df['Magnetisation Error']
    sigma_chi = df['Susceptibility Error']
    sigma_e = df['Energy Error']
    sigma_c = df['Heat Capacity Error']

    index_max_chi = np.argmax(chi)
    max_chi = max(chi)

    index_max_c = np.argmax(c)
    max_c = max(c)

    plt.rcParams.update({ # Use LaTeX for text rendering
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
    })

    if method.capitalize() == 'Glauber':
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 8), gridspec_kw={'wspace':0., 'hspace':0.1})
    
        ax[0].plot(T_list, m, color=color)
        ax[0].errorbar(T_list, m, sigma_m, color=color, capsize=2)
        ax[0].text(
            x=0.98, y=0.90,
            s='Magnetisation',
            transform=ax[0].transAxes,
            va='top',
            ha='right',
            fontsize=10,
            color='black'
        )
        ax[0].tick_params(axis='x', top=True, bottom=True, direction='in')
        ax[0].set_ylabel(r'$\mathrm{\overline{M}}$')

        ax[1].plot(T_list, chi, color=color)
        ax[1].errorbar(T_list, chi, sigma_chi, color=color, capsize=2)
        ax[1].text(
            x=0.98, y=0.90,
            s='Susceptibility',
            transform=ax[1].transAxes,
            va='top',
            ha='right',
            fontsize=10,
            color='black'
        )
        ax[1].text(x=T_list[index_max_chi]-0.05, y=max_chi-10, s=rf'$\mathrm{{k_BT}}$={T_list[index_max_chi]:.1f}', horizontalalignment='right')
        ax[1].tick_params(axis='x', top=True, bottom=True, direction='in')
        ax[1].set_ylabel(r'$\mathrm{\chi}$')

        ax[2].plot(T_list, e, color=color)
        ax[2].errorbar(T_list, e, +sigma_e, color=color, capsize=2)
        ax[2].text(
            x=0.98, y=0.90,
            s='Energy',
            transform=ax[2].transAxes,
            va='top',
            ha='right',
            fontsize=10,
            color='black'
        )
        ax[2].tick_params(axis='x', top=True, bottom=True, direction='in')
        ax[2].set_ylabel(r'$\mathrm{\overline{E}}$')

        ax[3].plot(T_list, c, color=color)
        ax[3].errorbar(T_list, c, sigma_c, color=color, capsize=2)
        ax[3].text(
            x=0.98, y=0.90,
            s='Heat Capacity',
            transform=ax[3].transAxes,
            va='top',
            ha='right',
            fontsize=10,
            color='black'
        )
        ax[3].text(x=T_list[index_max_c]-0.05, y=max_c-0.1, s=rf'$\mathrm{{k_BT}}$={T_list[index_max_c]:.1f}', horizontalalignment='right')
        ax[3].tick_params(axis='x', top=True, bottom=True, direction='in')
        ax[3].set_ylabel(r'$\mathrm{C}$')

        ax[3].set_xlabel(r'$\mathrm{k_BT}$')
   
    else:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 4), gridspec_kw={'wspace':0., 'hspace':0.1})


        ax[0].plot(T_list, e, color=color)
        ax[0].errorbar(T_list, e, +sigma_e, color=color, capsize=2)
        ax[0].text(
            x=0.98, y=0.90,
            s='Energy',
            transform=ax[0].transAxes,
            va='top',
            ha='right',
            fontsize=10,
            color='black'
        )
        ax[0].tick_params(axis='x', top=True, bottom=True, direction='in')
        ax[0].set_ylabel(r'$\mathrm{\overline{E}}$')

        ax[1].plot(T_list, c, color=color)
        ax[1].errorbar(T_list, c, sigma_c, color=color, capsize=2)
        ax[1].text(
            x=0.98, y=0.90,
            s='Heat Capacity',
            transform=ax[1].transAxes,
            va='top',
            ha='right',
            fontsize=10,
            color='black'
        )
        ax[1].text(x=T_list[index_max_c], y=max_c-0.3, s=rf'$\mathrm{{k_BT}}$={T_list[index_max_c]:.1f}', horizontalalignment='center')
        ax[1].tick_params(axis='x', top=True, bottom=True, direction='in')
        ax[1].set_ylabel(r'$\mathrm{C}$')

        ax[1].set_xlabel(r'$\mathrm{k_BT}$')
    
    plt.savefig(f'{method.capitalize()} plots.png', dpi=300, bbox_inches='tight')




if __name__ == "__main__":

    # Handle command line arguments
    parser = argparse.ArgumentParser(description="Runs an ising model using glauber or kawasaki dynamics")

    parser.add_argument("-a", "--action", help="What to do with model: 'plot' (animation) or 'measure' (calculate thermodynamic observables) or 'draw' (plot observables, requires .csv files from 'measure'). Default='plot'", type=str, default='plot')
    parser.add_argument("-t", "--temperature", help="Thermal energy of system (k_B * T). Default=1.5", type=float, default=1.5)
    parser.add_argument("-m", "--method", help="Monte-Carlo update method: 'glauber' or 'kawasaki'. Default='glauber'", type=str, default='glauber')
    parser.add_argument("-s", "--size", help="NxN size of lattice. Default=50", type=int, default=50)
    
    args = parser.parse_args()
    action = args.action
    temperature = args.temperature
    method = args.method
    N = args.size


    if action == 'plot':
        plot(N, temperature, method=method)
    elif action == 'measure':
        t1 = time.time()
        calc(method, N, n_measurements=1000, t_equilibration=50000, t_step=100)
        t2 = time.time()
        print(f'Time: {t2-t1:2f}')
    elif action == 'draw':
        figures(method)
    
    