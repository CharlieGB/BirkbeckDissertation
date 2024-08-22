import numpy as np
import matplotlib.pyplot as plt
import CTDL

class SOMLayer():

    def __init__(self, maze_dim, input_dim, size, learning_rate, sigma, sigma_const):

        self.size = size
        self.num_units = size * size
        self.num_dims = input_dim
        self.num_weights = input_dim

        self.learning_rate = learning_rate
        self.sigma = sigma
        self.sigma_const = sigma_const

        self.units = {'xy': [], 'w': [], 'etas': np.zeros(self.num_units), 'errors': np.zeros(self.num_units)}
        self.ConstructMap(maze_dim)

        return

    def ConstructMap(self, maze_dim):

        x = 0
        y = 0

        # Construct map
        for u in range(self.num_units):

            self.units['xy'].append([x, y])
            self.units['w'].append(np.random.rand(self.num_weights) * maze_dim - .5)

            x += 1
            if (x >= self.size):
                x = 0
                y += 1

        self.units['xy'] = np.array(self.units['xy'])
        self.units['w'] = np.array(self.units['w'])

        return

    def Update(self, state, unit, delta, update_mask):

        # self.units['w'][unit, :] = state
        # self.units['errors'][unit] = error


        diffs = self.units['xy'] - self.units['xy'][unit, :]
        location_distances = np.sqrt(np.sum(np.square(diffs), axis=-1))

        # sigma = np.clip(reward_value, 0, .01)
        # neighbourhood_values = np.exp(
        #     -np.square(location_distances) / (2.0 * (self.sigma_const + sigma)))

        neighbourhood_values = np.exp(-np.square(location_distances) / (
                2.0 * (self.sigma_const + (delta * self.sigma))))

        # lr = np.clip(reward_value, 0, .01)
        # self.units['w'] += lr * np.expand_dims(neighbourhood_values, axis=-1) * (state - self.units['w'])

        self.units['w'] += np.squeeze((delta * self.learning_rate) * \
                           np.expand_dims(neighbourhood_values, axis=-1) * (
                                   state - self.units['w'])) * \
                           np.tile(np.expand_dims(update_mask, axis=1), (1, 2))

        return

    def GetBestUnit(self, state):

        best_unit = np.argmin(np.sum((self.units['w'] - state) ** 2, axis=-1), axis=0)

        return best_unit
    

class SOM(object):

    def __init__(self, directory, maze_width, maze_height, input_dim, map_size, learning_rate, sigma, sigma_const):

        self.directory = directory
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.SOM_layer = SOMLayer(np.amax([maze_width, maze_height]), input_dim, map_size, learning_rate, sigma, sigma_const)

        self.location_counts = np.zeros((maze_height, maze_width))

        return

    def Update(self, state, unit, error, update_mask):

        self.SOM_layer.Update(state, unit, error, update_mask)

        return

    def GetOutput(self, state):

        best_unit = self.SOM_layer.GetBestUnit(state)

        return best_unit


    def PlotResults(self, plot_num):

        #self.PlotMap(plot_num)
        #self.PlotLocations(plot_num)

        return

    def PlotMap(self, plot_num):

        width = np.unique(self.SOM_layer.units['xy']).shape[0]
        height = width
        im_grid = np.zeros((width, height, 3))

        for i in range(width * height):
            image = np.zeros(3)
            image[:2] = self.SOM_layer.units['w'][i, :]
            image = np.clip(np.array(image) / np.amax([self.maze_width, self.maze_height]), 0, 1)
            im_grid[self.SOM_layer.units['xy'][i, 0], self.SOM_layer.units['xy'][i, 1], :] = image
        plt.figure()
        plt.imshow(im_grid)
        plt.savefig(self.directory + 'SOM%06d.pdf' % plot_num)
        plt.close()

        return

    def PlotLocations(self, plot_num):

        im_grid = np.zeros((self.maze_height, self.maze_width))

        for i in range(self.SOM_layer.num_units):
            y = int(np.rint(np.clip(self.SOM_layer.units['w'][i, 0], 0, self.maze_height-1)))
            x = int(np.rint(np.clip(self.SOM_layer.units['w'][i, 1], 0, self.maze_width-1)))
            im_grid[y, x] = 1

        plt.figure()
        plt.imshow(im_grid)
        plt.savefig(self.directory + 'SOMLocations%06d.png' % plot_num)
        plt.close()

        np.save(self.directory + 'SOMLocations', im_grid)

        return

    def RecordLocationCounts(self):

        for i in range(self.SOM_layer.num_units):
            y = int(np.clip(self.SOM_layer.units['w'][i, 0], 0, self.maze_height-1))
            x = int(np.clip(self.SOM_layer.units['w'][i, 1], 0, self.maze_width-1))
            self.location_counts[y, x] += 1

        return