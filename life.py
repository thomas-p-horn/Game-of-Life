import numpy as np
from numba import njit, prange
import time
import argparse
import matplotlib.pyplot as plt


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
    parser.add_argument("-t", "--temperature", help="Thermal energy of system (k_B * T). Default=1.5", type=float, default=1.5)
    parser.add_argument("-n", "--number", help="NxN size of lattice. Default=50", type=int, default=50)
    parser.add_argument("-s", "--state", help="Initialise the system with a specific state (eg. 'Glider')", default=None)
    parser.add_argument("-e", "--engine", help="Which update method to use ('roll' or 'numba')", default="roll")
    
    args = parser.parse_args()
    action = args.action
    temperature = args.temperature
    N = args.number
    state = args.state


    if action == 'plot':
        plot(state, N)
    
    
