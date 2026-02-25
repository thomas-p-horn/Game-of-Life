import numpy as np
from numba import njit, prange
import time
import argparse
import matplotlib.pyplot as plt
import csv


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
    lattice = init_state(L, 'glider')

    x_positions = []
    y_positions = []

    x_offset = 0
    y_offset = 0

    prev_x, prev_y = compute_com(lattice)

    for t in range(200):
        x, y = compute_com(lattice)

        dx = x - prev_x
        print(dx)
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

    plt.plot(x_positions, label='x')
    plt.plot(y_positions, label='y')
    plt.legend()
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
    
    args = parser.parse_args()
    action = args.action
    N = args.number
    state = args.state
    engine = args.engine


    if action == 'plot':
        plot(state, N)
    elif action == 'equilibrate':
        equilibrate(state, N, runs=10000)
    elif action == 'graph':
        graph()
    elif action == 'glider_speed':
        glider_speed()
    
    
