import copy
import os

import numpy as np
from functools import partial

from matplotlib import patches
import matplotlib.pyplot as plt


class QuoridorBoard:
    def __init__(self, n, board=None):
        assert n >= 3

        self.n = n
        self.history = {}

        midpoint_red = self.n // 2 + 1 - n % 2
        midpoint_blue = self.n // 2 - 1 + n % 2
        lastpoint = self.n - 1

        self.red_goal = lastpoint
        self.blue_goal = 0

        if board:
            self.setBoard(board)
        else:
            self.red_board = np.zeros((self.n, self.n), np.int16)
            self.blue_board = np.zeros((self.n, self.n), np.int16)
            self.v_walls = np.zeros((self.n - 1, self.n - 1), np.int16)
            self.h_walls = np.zeros((self.n - 1, self.n - 1), np.int16)
            self.draw = False

            self.max_walls = (self.n + 1) ** 2 // 10
            # red player
            self.red_position = (midpoint_red, 0)
            self.red_walls = self.max_walls
            self.red_board[self.red_position[0], self.red_position[1]] = 1

            # blue player
            self.blue_position = (midpoint_blue, lastpoint)
            self.blue_walls = self.max_walls
            self.blue_board[self.blue_position[0], self.blue_position[1]] = 1

        self.actions = {
            # NORTH
            0: partial(self.move, dx=+0, dy=+1),
            # SOUTH
            1: partial(self.move, dx=+0, dy=-1),
            # EAST
            2: partial(self.move, dx=+1, dy=+0),
            # WEST
            3: partial(self.move, dx=-1, dy=+0),
            # JN
            4: partial(self.move, dx=+0, dy=+2),
            # JS
            5: partial(self.move, dx=+0, dy=-2),
            # JE
            6: partial(self.move, dx=+2, dy=+0),
            # JW
            7: partial(self.move, dx=-2, dy=+0),
            # JNE
            8: partial(self.move, dx=+1, dy=+1),
            # JSW
            9: partial(self.move, dx=-1, dy=-1),
            # JNW
            10: partial(self.move, dx=-1, dy=+1),
            # JSE
            11: partial(self.move, dx=+1, dy=-1),
            # PLACE VERTICAL WALL
            'vw': self.placeVerticalWall,
            # PLACE HORIZONTAL WALL
            'hw': self.placeHorizontalWall,
        }

        self.convert_action = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10] + list(
            np.flip(np.arange(12, 12 + (self.n - 1) ** 2).reshape((self.n - 1, self.n - 1)), (0, 1)).ravel()) + list(
            np.flip(np.arange(12 + (self.n - 1) ** 2, 12 + 2 * (self.n - 1) ** 2).reshape((self.n - 1, self.n - 1)),
                    (0, 1)).ravel())

    def getGameEnded(self, player):
        if self.red_position[1] == self.red_goal:
            return player
        elif self.blue_position[1] == self.blue_goal:
            return -player
        elif self.draw:
            return 1e-4
        return 0

    def addToHistory(self):
        s = self.getBoardHashable()
        if s in self.history:
            self.history[s] += 1
        else:
            self.history[s] = 1

        if self.history[s] > 2:
            self.draw = True

    def getBoard(self):
        return self.red_board, self.blue_board, self.v_walls, self.h_walls, self.red_walls, self.blue_walls, self.draw

    def getBoardDist(self, is_red=True):
        dists = np.zeros((self.n, self.n), dtype=int)

        for i in range(self.n):
            for j in range(self.n):
                dists[i][j] = 255

        return dists

    def getBoardHashable(self):
        return hash((self.red_board.tostring(), self.blue_board.tostring(), self.v_walls.tostring(),
                     self.h_walls.tostring(), self.draw))

    def setBoard(self, board):
        self.history = copy.deepcopy(board.history)
        self.red_board = np.array(board.red_board, copy=True)
        self.blue_board = np.array(board.blue_board, copy=True)
        self.v_walls = np.array(board.v_walls, copy=True)
        self.h_walls = np.array(board.h_walls, copy=True)
        self.draw = board.draw

        self.red_position = board.red_position
        self.red_walls = board.red_walls
        self.red_goal = board.red_goal

        self.blue_position = board.blue_position
        self.blue_walls = board.blue_walls
        self.blue_goal = board.blue_goal

    def flipBoard(self):
        self.red_position, self.blue_position = (self.n - 1 - self.blue_position[0],
                                                 self.n - 1 - self.blue_position[1]), \
                                                (self.n - 1 - self.red_position[0],
                                                 self.n - 1 - self.red_position[1])
        self.red_walls, self.blue_walls = self.blue_walls, self.red_walls

        self.red_board, self.blue_board = self.blue_board, self.red_board

        self.red_board = np.flip(self.red_board, (0, 1))
        self.blue_board = np.flip(self.blue_board, (0, 1))
        self.v_walls = np.flip(self.v_walls, (0, 1))
        self.h_walls = np.flip(self.h_walls, (0, 1))

    def makeCanonical(self, player):
        if player != 1:
            self.flipBoard()
        return self

    def getValidActions(self, player):
        pass
        # red(x,y) hw bloqueia N ->     (x-1,y)     (x,y)
        # red(x,y) hw bloqueia NN ->    (x-1,y+1)   (x,y+1)
        # red(x,y) vw bloqueia NE ->    (x,y)       (x,y+1)
        # red(x,y) vw bloqueia NW ->    (x-1,y)     (x-1,y+1)

        # red(x,y) hw bloqueia S ->     (x-1,y-1)   (x,y-1)
        # red(x,y) hw bloqueia SS ->    (x-1,y-2)   (x,y-2)
        # red(x,y) vw bloqueia SW ->    (x-1,y-1)   (x-1,y-2)
        # red(x,y) vw bloqueia SE ->    (x,y-1)     (x,y-2)

        # red(x,y) vw bloqueia E ->     (x,y-1)     (x,y)
        # red(x,y) vw bloqueia EE ->    (x+1,y-1)   (x+1,y)

        # red(x,y) vw bloqueia W ->     (x-1,y-1)   (x-1,y)
        # red(x,y) vw bloqueia WW ->    (x-2,y-1)   (x-2,y)

    def executeAction(self, player, action):
        self.addToHistory()
        pawn_moves = 12
        vertical_wall_moves = pawn_moves + (self.n - 1) ** 2
        if player == -1:
            action = self.convert_action[action]

        # Pawn Moves
        if 0 <= action < pawn_moves:
            x, y = self.red_position if player == 1 else self.blue_position
            self.actions[action](player, x, y)

        # Vertical Walls
        elif pawn_moves <= action < vertical_wall_moves:
            x = int((action - pawn_moves) / (self.n - 1))
            y = int((action - pawn_moves) % (self.n - 1))
            self.actions['vw'](player, x, y)

        # Horizontal Walls
        else:
            x = int((action - vertical_wall_moves) / (self.n - 1))
            y = int((action - vertical_wall_moves) % (self.n - 1))
            self.actions['hw'](player, x, y)

    def move(self, player, x, y, dx=0, dy=0):
        if player == 1:
            player_board = self.red_board
            self.red_position = (x + dx, y + dy)
        else:
            player_board = self.blue_board
            self.blue_position = (x + dx, y + dy)

        player_board[x, y] = 0
        player_board[x + dx, y + dy] = 1

    def placeVerticalWall(self, player, x, y):
        if player == 1:
            self.red_walls -= 1
        else:
            self.blue_walls -= 1

        self.v_walls[x, y] = 1

    def placeHorizontalWall(self, player, x, y):
        if player == 1:
            self.red_walls -= 1
        else:
            self.blue_walls -= 1

        self.h_walls[x, y] = 1

    def plot_board(self, invert_yaxis=False, path=None, name=None, save=True):
        if path is None:
            path = []
        if name is None:
            name = str(self.getBoardHashable())

        fig_map, ax_map = plt.subplots(1, 1)

        if invert_yaxis:
            color1 = 'tab:red'
            color2 = 'tab:blue'
            color2w = 'gray'
        else:
            color1 = 'tab:blue'
            color2 = 'tab:red'
            color2w = 'gray'

        board_len = self.n * 2 - 1
        for y in range(board_len):
            for x in range(board_len):
                if y % 2 == 1 or x % 2 == 1:
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor='lightgray')
                    ax_map.add_patch(rect)

        for y in range(board_len):
            for x in range(board_len):
                # Walls gray background
                if x % 2 == 1 and y % 2 == 1:
                    # Red Walls
                    if self.v_walls[(x + 1) // 2 - 1, (y + 1) // 2 - 1] == 1:
                        rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=color2w)
                        ax_map.add_patch(rect)
                        rect = patches.Rectangle((x, y + 1), 1, 1, linewidth=0, facecolor=color2w)
                        ax_map.add_patch(rect)
                        rect = patches.Rectangle((x, y - 1), 1, 1, linewidth=0, facecolor=color2w)
                        ax_map.add_patch(rect)
                    if self.h_walls[(x + 1) // 2 - 1, (y + 1) // 2 - 1] == 1:
                        rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=color2w)
                        ax_map.add_patch(rect)
                        rect = patches.Rectangle((x + 1, y), 1, 1, linewidth=0, facecolor=color2w)
                        ax_map.add_patch(rect)
                        rect = patches.Rectangle((x - 1, y), 1, 1, linewidth=0, facecolor=color2w)
                        ax_map.add_patch(rect)
                elif x % 2 == 0 and y % 2 == 0:
                    # Red Player
                    if self.red_board[x // 2, y // 2] == 1:
                        rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=color2)
                        ax_map.add_patch(rect)
                    # Blue Player
                    if self.blue_board[x // 2, y // 2] == 1:
                        rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=color1)
                        ax_map.add_patch(rect)

        points = list(zip(path, path[1:]))[::2]
        for i, p in enumerate(points):
            if i != 0 and i != len(points) - 1:
                rect = patches.Rectangle(p, 1, 1, linewidth=0, facecolor='tab:green', alpha=0.5)
                ax_map.add_patch(rect)

        ax_map.set_aspect('equal')
        ax_map.set_yticks(np.arange(0, board_len, 1))
        ax_map.set_xticks(np.arange(0, board_len, 1))
        ax_map.set_xlim([0, board_len])
        ax_map.set_ylim([0, board_len])
        if invert_yaxis:
            ax_map.invert_yaxis()
            ax_map.invert_xaxis()

        if save:
            if not os.path.exists('./games'):
                os.makedirs('./games')
            plt.savefig('./games/' + name)
        else:
            plt.show()
        plt.close(fig_map)