import numpy as np
from numba import njit
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

custom_cmap = ListedColormap(['mintcream', 'tomato', 'darkgreen'])

def bootstrap(sample, function, k, *args):

    n = len(sample)

    vals = np.empty(k, dtype=np.float64)

    for j in range(k):

        indices = np.random.random_integers(0, n-1, size=n)

        vals[j] = function(sample[indices], *args)

    return np.sqrt(np.mean(vals**2) - np.mean(vals)**2)

running = True # This is to handle exiting the code when the figure is closed
def on_close(event):
    global running
    running = False


def init_sirs(L=50, i=0.333, r=0.333, immune=0.):
    lattice = np.random.rand(L, L)
    
    i_mask = (lattice < i)
    r_mask = (lattice > (1-r))
    s_mask = ~(i_mask + r_mask)

    lattice[s_mask] = 0
    lattice[i_mask] = 1
    lattice[r_mask] = 2

    immune_mask = (np.random.rand(L, L) < immune)
    lattice[immune_mask] = 3

    return lattice
    

@njit
def sirs_update(lattice, psi, pir, prs):

    L = lattice.shape[0]

    for _ in range(L**2): # Runs a whole sweep (no need for control over single site updates)

        i = np.random.randint(0, L)
        j = np.random.randint(0, L)

        state = lattice[i, j]

        if state == 0:

            i_neighbour = (
                lattice[(i-1) % L, j] == 1 or
                lattice[(i+1) % L, j] == 1 or
                lattice[i, (j-1) % L] == 1 or
                lattice[i, (j+1) % L] == 1
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



def sirs(L=50, pi=0.1, pr=0.3, ps=0.3, immune=0.):

    lattice = init_sirs(L, immune=immune)
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

def phase(L, s):
    t_eq = 100

    if s:
        n_sweep= s
    else:
        n_sweep = 1000
    grid_spacing = 0.05

    n_sites = L**2

    pir = 0.5
    prs_vals = np.arange(0, 1, grid_spacing)
    psi_vals = np.arange(0, 1, grid_spacing)

    means = np.empty([prs_vals.shape[0], psi_vals.shape[0]], dtype=np.float64)
    # variances = np.empty([prs_vals.shape[0], psi_vals.shape[0]], dtype=np.float64)

    for x, prs in enumerate(prs_vals):
        print(f'{prs*100}%')
        for y, psi in enumerate(psi_vals):
            fracs = np.empty([n_sweep], dtype=np.float64)

            lattice = init_sirs(L)

            for _ in range(t_eq):
                lattice = sirs_update(lattice, psi, pir, prs)


            for i in range(n_sweep):
                lattice = sirs_update(lattice, psi, pir, prs)
                fracs[i] = np.sum((lattice == 1)) / n_sites
            means[x, y] = np.mean(fracs)
            # variances[x, y] = np.var(fracs)

    np.savetxt("means.csv", means, delimiter=",")
    # np.savetxt("variances.csv", variances, delimiter=",")

    graph_sirs()

def graph_sirs():
    try:
        means = np.loadtxt("means.csv", delimiter=",")
        # variances = np.loadtxt("ariances.csv", delimiter=",")
    except:
        print("Non-existent files, run phase() first")

    plt.imshow(means, cmap='plasma', extent=(0, 1, 0, 1), origin='lower', aspect='equal')
    plt.ylabel(r"P(r$\rightarrow$s)")
    plt.xlabel(r"P(s$\rightarrow$i)")
    cbar = plt.colorbar()
    cbar.set_label(r"$\langle I \rangle / N$", rotation='horizontal')
    plt.savefig("means.png", dpi=300, bbox_inches='tight')
    plt.close()

    # plt.imshow(variances, cmap='plasma', extent=(0, 1, 0, 1), origin='lower', aspect='equal')
    # plt.ylabel(r"P(r$\rightarrow$s)")
    # plt.xlabel(r"P(s$\rightarrow$i)")
    # cbar = plt.colorbar()
    # cbar.set_label(r"$\sigma^2(\langle I \rangle / N)$", rotation='horizontal')
    # plt.savefig("variances.png", dpi=300, bbox_inches='tight')
    # plt.close()


def var_cut(L, s):
    n_sites = L**2
    t_eq = 500
    if s:
        n_sweep = s
    else:
        n_sweep = 10000
    pir = 0.5
    prs = 0.5
    psi_vals = np.linspace(0.2, 0.5, 50)
    var_list = np.empty_like(psi_vals)
    err_list = np.empty_like(psi_vals)

    for i, psi in enumerate(psi_vals):
        print(f'{2*i:.1f}%')

        lattice = init_sirs(L)
        for _ in range(t_eq):
            lattice = sirs_update(lattice, psi, pir, prs)

        fracs = np.empty([n_sweep], dtype=np.float64)
        for j in range(n_sweep):
            lattice = sirs_update(lattice, psi, pir, prs)
            fracs[j] = np.sum((lattice == 1))

        var_list[i] = np.var(fracs)
        err_list[i] = bootstrap(fracs, np.var, k=1000)

    df = pd.DataFrame({
        "P(s->i)": psi_vals,
        "Variance": var_list,
        "Error": err_list
    })
    df.to_csv('variances.csv', sep=',')
    graph_variance()


def graph_variance():
    df = pd.read_csv('variances.csv')
    psi_vals = df['P(s->i)']
    var_list = df['Variance']
    err_list = df['Error']

    # plt.plot(psi_vals, var_list)
    plt.figure(figsize=(16, 6))
    plt.errorbar(psi_vals, var_list, err_list)
    plt.xlabel('P(s->i)')
    plt.ylabel(r'$\mathrm{\sigma}^{2}(\langle I \rangle)$')
    plt.savefig('variances.png', dpi=300, bbox_inches='tight')

def immunity(L, s):
    L2 = L**2

    if s:
        n_sweeps = s
    else:
        n_sweeps = 1000

    immune = np.linspace(0, 1, 50)
    infected_means = np.empty_like(immune)

    for j, i in enumerate(immune):
        lattice = init_sirs(L, immune=i)
        frac_list = np.empty(n_sweeps)

        for _ in range(100):
            lattice = sirs_update(lattice, 0.5, 0.5, 0.5)

        for k in range(n_sweeps):
            lattice = sirs_update(lattice, 0.5, 0.5, 0.5)
            frac_list[k] = np.sum(lattice == 1) / L2
        
        infected_means[j] = np.mean(frac_list)

    df = pd.DataFrame({
        'Immune Fraction': immune,
        'Infected Fraction': infected_means
    })
    df.to_csv('immunity.csv', sep=',')
    graph_immunity()

def graph_immunity():
    df = pd.read_csv('immunity.csv')
    immune = df['Immune Fraction']
    infected = df['Infected Fraction']

    plt.figure(figsize=(16, 6))
    plt.plot(immune, infected, c='slateblue')
    plt.xlabel('Fraction of immune cells')
    plt.ylabel('Average infected population fraction')
    plt.savefig('immunity.png', dpi=300, bbox_inches='tight')
    # plt.show()




if __name__ == "__main__":

    # Handle command line arguments
    parser = argparse.ArgumentParser(description="SIRS Model")

    parser.add_argument("-a", "--action", help="What to do with model: 'animate' (animation) or 'phase' (calculate phases) or 'graph_phase' (create equilibration graph without recalculating; requires .csv files from 'phase') or 'variance' (calculates var(I/N) along P(s->i)=0.2-0.5 while keeping P(i->r)=P(r->s)=0.5) or 'graph_variance' (create variance graph without recalculating; requires .csv files from 'phase'). Default='animate'", type=str, default='animate')
    parser.add_argument("-L", "--length", help="LxL size of lattice. Default=50", type=int, default=50)
    parser.add_argument("-s", "--sweeps", help="Number of sweeps per measurement.", type=int, default=None)
    parser.add_argument("-psi", help="Probability of infection p(s->i)", type=float, default=0.3)
    parser.add_argument("-pir", help="Probability of recovery p(i->r)", type=float, default=0.3)
    parser.add_argument("-prs", help="Probability of becoming susceptible p(r->s)", type=float, default=0.3)
    parser.add_argument("--immune", help="Percentage of the inital population given full immunity. Default=0", type=float, default=0.)

    
    args = parser.parse_args()
    action = args.action
    L = args.length
    s = args.sweeps
    pi = args.psi
    pr = args.pir
    ps = args.prs
    immune = args.immune

    if action == 'animate':
        sirs(L, pi, pr, ps, immune)
    elif action == 'phase':
        phase(L, s)
    elif action == 'graph_phase':
        graph_sirs()
    elif action == 'variance':
        var_cut(L, s)
    elif action == 'graph_variance':
        graph_variance()
    elif action == 'immunity':
        immunity(L, s)
    elif action == 'graph_immunity':
        graph_immunity()
    
