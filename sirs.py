import numpy as np
from numba import njit, prange
import time
import argparse
import matplotlib.pyplot as plt
import csv
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit

custom_cmap = ListedColormap(['mintcream', 'tomato', 'darkgreen'])

plt.rcParams.update({ # Use LaTeX for text rendering
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times"],
})

def line(x, a, b):
    return a * x + b

running = True # This is to handle exiting the code when the figure is closed
def on_close(event):
    global running
    running = False

def init_random(N=50, p=0.5):
    return (np.random.rand(N, N) < p)

def init_state(N, state):

    lattice = np.zeros((N, N), dtype=np.bool_)
    mid = N // 2

    if state == 'glider':
        pattern = {
            (1, 0),
            (2, 1),
            (0, 2),
            (1, 2),
            (2, 2),
        }
    elif state == 'toad':
        pattern = {
            (2, 0),
            (0, 1),
            (3, 1),
            (0, 2),
            (3, 2),
            (1, 3)
        }
    elif state == 'pulsar':
        pattern = {
            (2, 0), (3, 0), (4, 0),
            (8, 0), (9, 0), (10, 0),
            (0, 2), (5, 2), (7, 2), (12, 2),
            (0, 3), (5, 3), (7, 3), (12, 3),
            (0, 4), (5, 4), (7, 4), (12, 4),
            (2, 5), (3, 5), (4, 5),
            (8, 5), (9, 5), (10, 5),
            (2, 7), (3, 7), (4, 7),
            (8, 7), (9, 7), (10, 7),
            (0, 8), (5, 8), (7, 8), (12, 8),
            (0, 9), (5, 9), (7, 9), (12, 9),
            (0,10), (5,10), (7,10), (12,10),
            (2,12), (3,12), (4,12),
            (8,12), (9,12), (10,12),
        }
    elif state == 'gosper_glider_gun':
        pattern = {
            (24,0),
            (22,1), (24,1),
            (12,2), (13,2), (20,2), (21,2), (34,2), (35,2),
            (11,3), (15,3), (20,3), (21,3), (34,3), (35,3),
            (0,4), (1,4), (10,4), (16,4), (20,4), (21,4),
            (0,5), (1,5), (10,5), (14,5), (16,5), (17,5), (22,5), (24,5),
            (10,6), (16,6), (24,6),
            (11,7), (15,7),
            (12,8), (13,8),
        }
    centre = max(value for pair in pattern for value in pair) // 2
    for dx, dy in pattern:
        lattice[mid - centre + dy][mid - centre + dx] = 1
    
    return lattice

def life_update(lattice):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    neighbours = np.zeros(lattice.shape)

    for dir in directions:
        neighbours += np.roll(lattice, dir, axis=(0, 1))

    return ((neighbours == 3) | (lattice & (neighbours == 2)))

@njit
def numba_life_update(lattice):
    rows, cols = lattice.shape
    neighbours = np.zeros_like(lattice)

    for i in range(rows):
        for j in range(cols):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni = (i + di) % rows
                    nj = (j + dj) % cols
                    neighbours[i, j] += lattice[ni, nj]

    return ((neighbours == 3) | (lattice & (neighbours == 2))).astype(np.int32)


def plot(state=None, N=50):

    if state == None:
        lattice = init_random(N)
    else:
        lattice = init_state(N, state)
    print(type(lattice))
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect("close_event", on_close) # Handles closing figure
    img = ax.imshow(lattice, cmap="Greens", vmin=0, vmax=1.8)
    ax.set_title("Conway's Game of Life")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    while running:
        lattice = life_update(lattice)
        img.set_data(lattice)
        plt.pause(0.05)
    plt.ioff()

def equilibrate(state=None, N=50, runs=100, max_run_length=15000, engine='roll'):
    times = []

    for _ in range(runs):
        if state == None:
            lattice = init_random(N)
        else:
            lattice = init_state(N, state)

        N = 0
        alive_previous = np.sum(lattice)

        for i in range(1, max_run_length):
            if i == max_run_length:
                times.append(i)
                break

            lattice = life_update(lattice)
            alive = np.sum(lattice)

            if alive == alive_previous:
                N += 1
                if N >= 50:
                    times.append(i)
                    break
            else:
                N = 0
            alive_previous = alive

    with open('equilibration times.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(times)
    
    plt.hist(times)
    plt.show()
    

def graph():
    
    try:
        with open('equilibration times.csv', newline='') as f:
            reader = csv.reader(f)
            times = next(reader)
            times = [int(time) for time in times]
    except:
        print("File doesn't exist. Try running 'equilibrate' first.")

    print(times)
    plt.hist(times, bins=100, color='forestgreen')
    plt.xlabel('Equilibration time')
    plt.ylabel('N')
    plt.savefig('equilibration time.png', dpi=300, bbox_inches='tight')



def compute_com(lattice):
    L = lattice.shape[0]
    total = np.sum(lattice)

    x_indices = np.arange(L)
    y_indices = np.arange(L)

    x_com = np.sum(np.sum(lattice, axis=1) * x_indices) / total
    y_com = np.sum(np.sum(lattice, axis=0) * y_indices) / total

    return x_com, y_com

def glider_speed():
    L = 50
    t = np.arange(1000)
    lattice = init_state(L, 'glider')

    x_positions = []
    y_positions = []

    x_offset = 0
    y_offset = 0

    prev_x, prev_y = compute_com(lattice)

    for _ in t:
        x, y = compute_com(lattice)

        dx = x - prev_x
        if dx > L/2:
            x_offset -= L
        elif dx < -L/2:
            x_offset += L

        dy = y - prev_y
        if dy > L/2:
            y_offset -= L
        elif dy < -L/2:
            y_offset += L

        x_positions.append(x + x_offset)
        y_positions.append(y + y_offset)

        prev_x, prev_y = x, y
        lattice = life_update(lattice)

    x_positions, y_positions, t = np.array(x_positions), np.array(y_positions), np.array(t, dtype=float)
    indices = np.arange(len(x_positions))
    mask = (indices - 90) % 200 <= 15
    x_positions[mask] = np.nan
    y_positions[mask] = np.nan
    t[mask] = np.nan

    mask = indices >= 100
    x_positions[mask] += L * ((indices[mask] - 100) // 200 + 1)
    y_positions[mask] += L * ((indices[mask] - 100) // 200 + 1)

    x_positions = x_positions[~np.isnan(x_positions)]
    y_positions = y_positions[~np.isnan(y_positions)]
    t = t[~np.isnan(t)]

    x_popt, _ = curve_fit(line, t, x_positions)
    y_popt, _ = curve_fit(line, t, y_positions)

    vx = x_popt[0]
    vy = y_popt[0]
    vtot = np.sqrt(vx**2 + vy**2)

    # print(f'Vx = {vx:.5f}')
    # print(f'Vy = {vy:.5f}')
    print(f'Glider velocity: {vtot:.5f} cells/timestep')


    # plt.plot(t, x_positions, label='x')


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



def speedtest():
    N = 500000

    t1 = time.time()

    lattice = init_random()
    for _ in range(N):
        lattice = life_update(lattice)
    
    t2 = time.time()

    lattice = init_random()
    for _ in range(N):
        lattice = numba_life_update(lattice)

    t3 = time.time()

    print(f"{N} iterations")
    print(f"Roll: {t2-t1:.2f}s")
    print(f"Numba: {t3-t2:.2f}s")


if __name__ == "__main__":

    # Handle command line arguments
    parser = argparse.ArgumentParser(description="Conway's Game of Life")

    parser.add_argument("-a", "--action", help="What to do with model: 'animate' (animation) or 'measure' (calculate equilibration time) or 'plot' (plot equilibration graph, requires .csv files from 'measure'). Default='plot'", type=str, default='plot')
    parser.add_argument("-n", "--number", help="NxN size of lattice. Default=50", type=int, default=50)
    parser.add_argument("-s", "--state", help="Initialise the system with a specific state (eg. 'Glider')", default=None)
    parser.add_argument("-e", "--engine", help="Which update method to use ('roll' or 'numba')", default="roll")
    parser.add_argument("-pi", help="Probability of infection p(s->i)", type=float, default=0.1)
    parser.add_argument("-pr", help="Probability of recovery p(i->r)", type=float, default=0.3)
    parser.add_argument("-ps", help="Probability of becoming susceptible p(r->s)", type=float, default=0.3)

    
    args = parser.parse_args()
    action = args.action
    N = args.number
    state = args.state
    engine = args.engine
    pi = args.pi
    pr = args.pr
    ps = args.ps


    if action == 'plot':
        plot(state, N)
    elif action == 'equilibrate':
        equilibrate(state, N, runs=10000)
    elif action == 'graph':
        graph()
    elif action == 'glider_speed':
        glider_speed()
    elif action == 'sirs':
        sirs(N, pi, pr, ps)
    elif action == 'phase':
        phase(N)
    elif action == 'graph_sirs':
        graph_sirs()
    elif action == 'variance_plot':
        var_cut(N)
    
