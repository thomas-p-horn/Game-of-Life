import numpy as np
import argparse
import matplotlib.pyplot as plt
import csv
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit

custom_cmap = ListedColormap(['mintcream', 'tomato', 'darkgreen'])

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


def animate(state=None, N=50):

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

def equilibrate(state=None, N=50, runs=100, max_run_length=15000):
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
    
    graph()
    

def graph():
    
    try:
        with open('equilibration times.csv', newline='') as f:
            reader = csv.reader(f)
            times = next(reader)
            times = [int(time) for time in times]
    except:
        print("File doesn't exist. Try running 'equilibrate' first.")

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



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Conway's Game of Life")
    parser.add_argument("-a", "--action", help="What to do with model: 'animate' (animation) or 'equilibrate' (calculate equilibration time) or 'graph' (plot equilibration graph, requires .csv files from 'measure'). Default='animate'", type=str, default='animate')
    parser.add_argument("-l", "--length", help="LxL size of lattice. Default=50", type=int, default=50)
    parser.add_argument("-r", "--runs", help="Number of runs when calculating equilibration time. Default='10000", type=int, default=10000)
    parser.add_argument("-s", "--state", help="Initialise the system with a specific state (eg. 'Glider')", default=None)

    
    args = parser.parse_args()
    action = args.action
    L = args.length
    r = args.runs
    state = args.state

    if action == 'animate':
        animate(state, L)
    elif action == 'equilibrate':
        equilibrate(state, L, runs=r)
    elif action == 'graph':
        graph()
    elif action == 'glider_speed':
        glider_speed()
    
