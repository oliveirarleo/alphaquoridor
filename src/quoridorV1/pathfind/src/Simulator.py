#!/usr/bin/env python3
import os
import sys

sys.path.append(os.getcwd() + '/src')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import randint
import random


class Simulator(object):
    resolution: int
    map_width: int
    map_height: int
    value_non_obs: int
    value_obs: int
    size_obs_width: int
    size_obs_height: int
    obs_left_top_corner_list: list

    def __init__(self,
                 map_width_meter: float,
                 map_height_meter: float):
        """
        Constructor

        the cell is empty if value_non_obs, the cell is blocked if value_obs.
        """

        # map resolution, how many cells per meter
        # how many cells for width and height
        map_width = map_width_meter
        map_height = map_height_meter

        # check if these are integers
        assert (map_width.is_integer() == True)
        assert (map_height.is_integer() == True)

        self.map_width = int(map_width)
        self.map_height = int(map_height)

        # create an empty map
        # self.map_array = np.array([self.value_non_obs] * (self.map_width * self.map_height)).reshape(-1, self.map_width)
        self.map_array = np.zeros((self.map_width, self.map_height), int)
        # a 2D list, each element is x and y coordinates of the obstacle's left top corner
        self.obs_left_top_corner_list = []
        self.goals = [(0, 8), (8, 0), (16, 8), (8, 16), ]

    def quoridor_generate_random_obs(self, num_obs: int):

        for idx_obs in range(0, num_obs):
            for _ in range(10):
                colision = False
                dims = random.choice([(1, 3), (3, 1)])
                if dims[0] == 3:
                    x = random.randrange(0, self.map_array.shape[0]-2, 2)
                    y = random.randrange(1, self.map_array.shape[1]-1, 2)
                else:
                    x = random.randrange(1, self.map_array.shape[0]-1, 2)
                    y = random.randrange(0, self.map_array.shape[1]-2, 2)

                for i in range(dims[0]):
                    for j in range(dims[1]):
                        if self.map_array[x+i][y+j] == 9:
                            colision = True

                if not colision:
                    for i in range(dims[0]):
                        for j in range(dims[1]):
                            self.map_array[x + i][y + j] = 9
                    break


    def quoridor_generate_blocking_obs(self):
            for i in range(17):
                self.map_array[1][i] = 9

    def plot_single_path(self, *arguments):
        """
        Simulator.visualize(path) # plot a path
        Simulator.visualize(path_full, path_short) # plot two paths

        path is a list for the trajectory. [x[0], y[0], x[1], y[1], ...]
        """

        fig_map, ax_map = plt.subplots(1, 1)

        # plot retangle obstacles
        for idx, x in np.ndenumerate(self.map_array):
            idx = (idx[1], idx[0])
            if x > 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='k')
                # Add the patch to the Axes
                ax_map.add_patch(rect)
            elif idx[0]%2 == 1 or idx[1]%2 == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1,  facecolor='lightgray')
                # Add the patch to the Axes
                ax_map.add_patch(rect)

        points = list(zip(arguments[0], arguments[0][1:]))[::2]
        for i, p in enumerate(points):
            if i == 0:
                rect = patches.Rectangle(p, 1, 1,
                                         linewidth=1,  facecolor='r')
            elif i == len(points) - 1:
                rect = patches.Rectangle(p, 1, 1,
                                         linewidth=1, facecolor='g')
            else:
                rect = patches.Rectangle(p, 1, 1,
                                         linewidth=1, facecolor='b')
            ax_map.add_patch(rect)

        # if len(arguments) == 2:
        #     ax_map.plot(arguments[1][0::2], arguments[1][1::2], label="short path")
        # ax_map.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_map.set_xlabel("x")
        ax_map.set_ylabel("y")
        ax_map.set_aspect('equal')
        ax_map.set_yticks(np.arange(0, 17, 2))
        ax_map.set_xticks(np.arange(0, 17, 2))
        ax_map.set_xlim([0, 17])
        ax_map.set_ylim([0, 17])
        plt.show()

    def quoridor_generate_start(self):
        start = [random.randrange(0, 17, 2), random.randrange(0, 17, 2)]
        while self.map_array[start[1]][start[0]] > 1:
            start = [random.randrange(0, 17, 2), random.randrange(0, 17, 2)]
            print("Start is inside an obstacle. Re-generate a new start.")

        return start

    def quoridor_generate_random_goals(self):

        goal = random.choice(self.goals)
        while self.map_array[goal[1]][goal[0]] > 1:
            print("Target is inside an obstacle. Re-generate a new target.")
            goal = random.choice(self.goals)
        return goal

    def quoridor_generate_goals(self):
        for g in self.goals:
            yield g
        return self.quoridor_generate_random_goals()

    def quoridor_generate_start_and_goals(self, num_tests):
        targets = []
        starts = []
        goals = self.quoridor_generate_goals()
        for _ in range(num_tests):
            starts.extend(self.quoridor_generate_start())
            targets.extend(next(goals))

        return starts, targets
