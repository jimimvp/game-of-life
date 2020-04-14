from .rendering import SimpleImageViewer
import numpy as np
from numpy.random import RandomState
from skimage.transform import resize
import time




def neighbour_count(grid):
    count = np.zeros_like(grid).astype(np.uint8)
    count[1:] += grid[:-1]
    count[:-1] += grid[1:]
    count[:, 1:] += grid[:, :-1]
    count[:, :-1] += grid[:, 1:]

    # diagonal neighbours
    count[1:, 1:] += grid[:-1, :-1]
    count[1:, :-1] += grid[:-1, 1:]
    count[:-1, 1:] += grid[1:, :-1]
    count[:-1, :-1] += grid[1:, 1:]

    return count

class CellularAutomata:



    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size)).astype(np.bool)
        self.background = np.ones((grid_size, grid_size, 3), dtype=np.uint8)*255

        ints = np.random.randint(0, 244, grid_size**2*3).astype(np.uint8)

        repeating = np.random.randint(0, 244, 30)
        repeats = int(grid_size/10*grid_size)
        repeating = np.tile(repeating, (1, repeats)).reshape(grid_size, grid_size, 3)

        self.colors = ints.reshape(grid_size, grid_size, 3)
        self.colors = repeating

        self.viewer = None

    def random_init(self, seed=None):
        if seed is None:
            seed = np.random.randint(int(1e6))
        np_random = RandomState(seed)
        self.grid = np_random.choice([0,1], self.grid_size**2).reshape(self.grid_size, self.grid_size).astype(np.bool)


    def random_init_middle(self, side_offset, seed=None):
        if seed is None:
            seed = np.random.randint(int(1e6))
        np_random = RandomState(seed)
        subgrid_size = self.grid_size - side_offset*2
        self.grid[side_offset:-side_offset, side_offset:-side_offset] = np_random.choice([0,1], subgrid_size**2).reshape(subgrid_size, subgrid_size)

    def simulate(self, steps=1):
        prev_grid = self.grid
        for i in range(steps):
            next_gen = np.zeros_like(self.grid)
            ncounts = neighbour_count(self.grid)

            # Any live cell with two or three live neighbors survives.
            next_gen[((ncounts==3) + (ncounts==2)) * self.grid] = 1
            # Any dead cell with three live neighbors becomes a live cell.
            next_gen[(self.grid==0) * (ncounts==3)] = 1
            # All the other cells are dead

            self.grid = next_gen

        self.grid_changed = np.any(self.grid != prev_grid)
        self.amount_of_change = np.sum(self.grid != prev_grid)


        return self.grid

    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size)).astype(np.bool)
        self.viewer = None
        self.grid_changed=True
        self.amount_of_change = 0

    def render(self):

        if self.viewer is None:
            self.viewer = SimpleImageViewer(0, maxwidth=900)



        # convert to w x h x 3
        img = np.bitwise_not(self.grid)[..., None].astype(np.float32)*255
        img = np.tile(img, (1, 3))
        #img[self.grid == 1] = self.colors[self.grid==1]


        img = resize(img, (900, 900, 3), order=0).astype(np.uint8)

        self.viewer.imshow(img)


def search_seeds(grid_size, side_offset):
    automata = CellularAutomata(grid_size)


    # search for cellular automata that does something
    steps = []
    amount_of_change = []
    prev_max = -1
    for seed in range(100000):
        automata.reset()
        automata.random_init_middle(seed=seed, side_offset=side_offset)

        reward = 0
        for i in range(500):
            automata.simulate()
            reward += 1
            if not automata.grid_changed: break
        steps.append(i)
        amount_of_change.append(reward)
        if prev_max < reward:
            prev_max = reward
            print("New maximum change seed:", seed)
        #print(f"Seed {seed} does something for {i} steps!")

    max_steps = np.max(steps)
    print("Good seeds:")
    print(np.argwhere(max_steps==np.array(steps)))
    print("Maximum amount of change:")
    max_change = np.max(amount_of_change)
    print(np.argwhere(np.array(amount_of_change) == max_change))

def execute_seed(seed, grid_size, side_offset):
    automata = CellularAutomata(grid_size)

    automata.reset()
    automata.random_init_middle(seed=seed, side_offset=side_offset)

    for i in range(10000):
        automata.simulate()
        automata.render()
        time.sleep(0.002)


def execute_test(grid_size=10):
    automata = CellularAutomata(grid_size)

    automata.reset()
    automata.grid[5, 3:6] = 1

    for i in range(1000):
        automata.simulate()
        automata.render()
        time.sleep(.1)

if __name__ == '__main__':
    #search_seeds(25, 10)
    #execute_test()
    execute_seed(1, 280, 120)
