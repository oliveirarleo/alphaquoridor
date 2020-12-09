#!/usr/bin/env python3
import os
import sys
from pprint import pprint

sys.path.append(os.getcwd() + '/build')
sys.path.append(os.getcwd() + '/src')
import QuoridorAStarPython
import numpy as np
import time
import matplotlib.pyplot as plt

import Simulator as helper

if __name__ == "__main__":
    # define the world
    map_width_meter = 17.0
    map_height_meter = 17.0
    # create a simulator
    Simulator = helper.Simulator(map_width_meter, map_height_meter)
    # number of obstacles
    num_obs = 20
    # [width, length] size of each obstacle [meter]
    size_obs = [1, 1]
    # generate random obstacles
    Simulator.quoridor_generate_random_obs(num_obs)
    # Simulator.quoridor_generate_blocking_obs()

    # convert 2D numpy array to 1D list
    world_map = Simulator.map_array.flatten().tolist()

    for i in Simulator.map_array:
        print(i)
    print(Simulator.map_array.shape)

    # define the start and goal
    num_targets = 1
    start, end = Simulator.quoridor_generate_start_and_goals(4)
    # solve it
    print('start ', start)
    print('end ', end)
    start_points = list(zip(start, start[1:]))[::2]
    end_points = list(zip(end, end[1:]))[::2]
    for s, e in zip(start_points, end_points):
        print(s, e)

        t0 = time.time()
        path_short, steps_used = QuoridorAStarPython.FindPath(s, e, world_map, Simulator.map_width,
                                                              Simulator.map_height)
        t1 = time.time()
        print("Time used for a single path is [sec]:")
        print(t1 - t0)
        print("This is the path. " + "Steps used:" + str(steps_used))
        if s != tuple(path_short[:2]):
            path_short = list(s) + path_short
        if e != tuple(path_short[-2:]):
            path_short = path_short + list(e)

        # visualization (uncomment next line if you want to visualize a single path)
        Simulator.plot_single_path(path_short)

    print(QuoridorAStarPython.PathExistsAll(start, end, world_map, Simulator.map_width, Simulator.map_height))
