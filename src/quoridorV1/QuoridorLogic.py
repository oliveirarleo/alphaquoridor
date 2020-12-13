import copy
import os

import numpy as np
from functools import partial

from matplotlib import patches
import matplotlib.pyplot as plt
import QuoridorUtils


class QuoridorBoard:
    def __init__(self, n, board=None):
        assert n >= 3

        self.n = n
        self.board_len = 2 * self.n - 1
        self.board_z = 6

        self.history = {}

        if n % 2 == 1:
            midpoint_red = midpoint_blue = int(self.board_len / 2)
        else:
            midpoint_red = int(self.board_len / 2) + 1
            midpoint_blue = int(self.board_len / 2) - 1
        lastpoint = self.board_len - 1

        self.red_goal = (midpoint_red, lastpoint)
        self.blue_goal = (midpoint_blue, 0)

        if board:
            self.setBoard(board)
        else:
            self.red_board = np.zeros((self.board_len, self.board_len), np.int16)
            self.blue_board = np.zeros((self.board_len, self.board_len), np.int16)
            self.red_walls_board = np.zeros((self.board_len, self.board_len), np.int16)
            self.blue_walls_board = np.zeros((self.board_len, self.board_len), np.int16)
            self.draw = np.zeros((self.board_len, self.board_len), np.int16)

            num_walls = (self.n + 1) ** 2 // 10
            # red player
            self.red_position = (midpoint_red, 0)
            self.red_walls = num_walls
            self.red_board[self.red_position[0], self.red_position[1]] = 1

            # blue player
            self.blue_position = (midpoint_blue, lastpoint)
            self.blue_walls = num_walls
            self.blue_board[self.blue_position[0], self.blue_position[1]] = 1

        self.actions = {
            # NORTH
            0: partial(self.move, dx=+0, dy=+2),
            # SOUTH
            1: partial(self.move, dx=+0, dy=-2),
            # EAST
            2: partial(self.move, dx=+2, dy=+0),
            # WEST
            3: partial(self.move, dx=-2, dy=+0),
            # JN
            4: partial(self.move, dx=+0, dy=+4),
            # JS
            5: partial(self.move, dx=+0, dy=-4),
            # JE
            6: partial(self.move, dx=+4, dy=+0),
            # JW
            7: partial(self.move, dx=-4, dy=+0),
            # JNE
            8: partial(self.move, dx=+2, dy=+2),
            # JNW
            9: partial(self.move, dx=-2, dy=+2),
            # JSE
            10: partial(self.move, dx=+2, dy=-2),
            # JSW
            11: partial(self.move, dx=-2, dy=-2),
            # PLACE VERTICAL WALL
            'vw': self.placeVerticalWall,
            # PLACE HORIZONTAL WALL
            'hw': self.placeHorizontalWall,
        }

        self.convert_action = [1, 0, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8]
        cvw = list(np.flip(np.arange(12, 12 + (self.n - 1) ** 2).reshape((self.n - 1, self.n - 1)), (0, 1)).ravel())
        chw = list(
            np.flip(np.arange(12 + (self.n - 1) ** 2, 12 + 2 * (self.n - 1) ** 2).reshape((self.n - 1, self.n - 1)),
                    (0, 1)).ravel())
        self.convert_action = self.convert_action + cvw + chw

    def getGameEnded(self, player):
        if self.red_position[1] == self.red_goal[1]:
            return player
        elif self.blue_position[1] == self.blue_goal[1]:
            return -player
        elif self.draw[0][0] == 1:
            return 1e-4
        return 0

    def addToHistory(self):
        s = self.getBoardHashable()
        if s in self.history:
            self.history[s] += 1
        else:
            self.history[s] = 1

        if self.history[s] > 2:
            self.draw = np.ones((self.board_len, self.board_len))

    def getRepetitions(self):
        s = self.getBoardHashable()
        if s in self.history:
            return self.history[s]
        return 0

    def getBoard(self):
        board = np.zeros((self.board_z, self.board_len, self.board_len), dtype=int)
        board[0] = self.red_walls_board + self.blue_walls_board
        board[1] = self.getBoardDist()
        board[2] = self.red_board
        board[3] = self.getBoardDist(is_red=False)
        board[4] = self.blue_board
        board[5] = self.draw
        return board

    def getBoardDist(self, is_red=True):
        w = self.red_walls_board + self.blue_walls_board
        dists = np.ones((self.board_len, self.board_len), dtype=int)

        for i in range(self.board_len):
            for j in range(self.board_len):
                dists[i][j] = 255

        if is_red:

            changed = True
            while changed:
                changed = False
                for i in range(0, self.board_len, 2):
                    for j in range(self.board_len - 1, -1, -2):
                        if j == self.board_len - 1:
                            dists[i][j] = 0
                        else:
                            min = 254
                            if i < self.board_len - 2 and min > dists[i + 2][j] and w[i + 1][j] == 0:
                                min = dists[i + 2][j]
                            if i > 1 and min > dists[i - 2][j] and w[i - 1][j] == 0:
                                min = dists[i - 2][j]
                            if min > dists[i][j + 2] and w[i][j+1] == 0:
                                min = dists[i][j + 2]
                            if j > 1 and min > dists[i][j - 2] and w[i][j-1] == 0:
                                min = dists[i][j - 2]
                            if dists[i][j] != min + 1:
                                dists[i][j] = min + 1
                                changed = True
        else:
            changed = True
            while changed:
                changed = False
                for i in range(0, self.board_len, 2):
                    for j in range(0, self.board_len, 2):
                        if j == 0:
                            dists[i][j] = 0
                        else:
                            min = 254
                            if i < self.board_len - 2 and min > dists[i + 2][j] and w[i + 1][j] == 0:
                                min = dists[i + 2][j]
                            if i > 1 and min > dists[i - 2][j] and w[i - 1][j] == 0:
                                min = dists[i - 2][j]
                            if j < self.board_len - 2 and min > dists[i][j + 2] and w[i][j + 1] == 0:
                                min = dists[i][j + 2]
                            if j > 1 and min > dists[i][j - 2] and w[i][j - 1] == 0:
                                min = dists[i][j - 2]
                            if dists[i][j] != min + 1:
                                dists[i][j] = min + 1
                                changed = True
        return dists/255

    def getBoardHashable(self):
        board = np.zeros((5, self.board_len, self.board_len), dtype=int)
        board[0] = self.red_walls_board
        board[1] = self.blue_walls_board
        board[2] = self.red_board
        board[3] = self.blue_board
        board[4] = self.draw
        return hash(board.tostring())

    def setBoard(self, board):
        self.history = copy.deepcopy(board.history)
        self.red_board = np.array(board.red_board, copy=True)
        self.blue_board = np.array(board.blue_board, copy=True)
        self.red_walls_board = np.array(board.red_walls_board, copy=True)
        self.blue_walls_board = np.array(board.blue_walls_board, copy=True)
        self.draw = np.array(board.draw, copy=True)
        self.red_position = board.red_position
        self.red_walls = board.red_walls
        self.blue_position = board.blue_position
        self.blue_walls = board.blue_walls

    def flipBoard(self):
        l = self.board_len - 1
        self.red_position, self.blue_position = (l - self.blue_position[0], l - self.blue_position[1]), \
                                                (l - self.red_position[0], l - self.red_position[1])
        self.red_goal, self.blue_goal = (l - self.blue_goal[0], l - self.blue_goal[1]), \
                                        (l - self.red_goal[0], l - self.red_goal[1])
        self.red_walls, self.blue_walls = self.blue_walls, self.red_walls

        self.red_board, self.blue_board = self.blue_board, self.red_board
        self.red_walls_board, self.blue_walls_board = self.blue_walls_board, self.red_walls_board

        self.red_board = np.flip(self.red_board, (0, 1))
        self.blue_board = np.flip(self.blue_board, (0, 1))
        self.red_walls_board = np.flip(self.red_walls_board, (0, 1))
        self.blue_walls_board = np.flip(self.blue_walls_board, (0, 1))

    def makeCanonical(self, player):
        if player != 1:
            self.flipBoard()
        return self

    def getValidActions(self, player):
        if player == 1:
            player_x = self.red_position[0]
            player_y = self.red_position[1]
            player_end_x = self.red_goal[0]
            player_end_y = self.red_goal[1]
            opponent_x = self.blue_position[0]
            opponent_y = self.blue_position[1]
            opponent_goal_x = self.blue_goal[0]
            opponent_goal_y = self.blue_goal[1]
            walls = self.red_walls
        else:
            player_x = self.blue_position[0]
            player_y = self.blue_position[1]
            player_end_x = self.blue_goal[0]
            player_end_y = self.blue_goal[1]
            opponent_x = self.red_position[0]
            opponent_y = self.red_position[1]
            opponent_goal_x = self.red_goal[0]
            opponent_goal_y = self.red_goal[1]
            walls = self.blue_walls

        return QuoridorUtils.GetValidActions(player_x, player_y, player_end_x, player_end_y,
                                             opponent_x, opponent_y, opponent_goal_x, opponent_goal_y,
                                             self.red_walls_board + self.blue_walls_board, walls)

    def findPlayer(self, player):
        player_board = self.red_board if player == 1 else self.blue_board
        pos = np.where(player_board == 1)
        return pos[0][0], pos[1][0]

    def countWalls(self, player):
        wall_board = self.red_walls_board if player == 1 else self.blue_walls_board
        return np.sum(wall_board) / 3

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
            y = int((action - pawn_moves) % (self.n - 1)) * 2 + 1
            x = int((action - pawn_moves) / (self.n - 1)) * 2 + 1
            self.actions['vw'](player, x, y)
        # Horizontal Walls
        else:
            y = int((action - vertical_wall_moves) % (self.n - 1)) * 2 + 1
            x = int((action - vertical_wall_moves) / (self.n - 1)) * 2 + 1
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
            wall_board = self.red_walls_board
            self.red_walls -= 1
        else:
            wall_board = self.blue_walls_board
            self.blue_walls -= 1

        wall_board[x, y] = 1
        wall_board[x, y + 1] = 1
        wall_board[x, y - 1] = 1

    def placeHorizontalWall(self, player, x, y):
        if player == 1:
            wall_board = self.red_walls_board
            self.red_walls -= 1
        else:
            wall_board = self.blue_walls_board
            self.blue_walls -= 1

            # boardsize = board.shape[1]-1
            # y = boardsize-y
            # x = boardsize-x

        wall_board[x, y] = 1
        wall_board[x + 1, y] = 1
        wall_board[x - 1, y] = 1

    def plot_board(self, invert_yaxis=False, path=None, name=None):
        if path is None:
            path = []
        if name is None:
            name = str(self.getBoardHashable())

        fig_map, ax_map = plt.subplots(1, 1)

        if invert_yaxis:
            color1 = 'tab:red'
            color2 = 'tab:blue'
            color1w = 'darkred'
            color2w = 'darkblue'
        else:
            color1 = 'tab:blue'
            color2 = 'tab:red'
            color1w = 'darkblue'
            color2w = 'darkred'

        for y in range(self.board_len):
            for x in range(self.board_len):
                # Walls gray background
                if y % 2 == 1 or x % 2 == 1:
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor='lightgray')
                    ax_map.add_patch(rect)
                # Red Walls
                if self.red_walls_board[x, y] == 1:
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=color2w)
                    ax_map.add_patch(rect)
                # Blue Walls
                if self.blue_walls_board[x, y] == 1:
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=color1w)
                    ax_map.add_patch(rect)
                # Red Player
                if self.red_board[x, y] == 1:
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=color2)
                    ax_map.add_patch(rect)
                # Blue Player
                if self.blue_board[x, y] == 1:
                    rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor=color1)
                    ax_map.add_patch(rect)

        points = list(zip(path, path[1:]))[::2]
        for i, p in enumerate(points):
            if i != 0 and i != len(points) - 1:
                rect = patches.Rectangle(p, 1, 1, linewidth=0, facecolor='tab:green', alpha=0.5)
                ax_map.add_patch(rect)

        ax_map.set_aspect('equal')
        ax_map.set_yticks(np.arange(0, self.board_len, 1))
        ax_map.set_xticks(np.arange(0, self.board_len, 1))
        ax_map.set_xlim([0, self.board_len])
        ax_map.set_ylim([0, self.board_len])
        if invert_yaxis:
            ax_map.invert_yaxis()
            ax_map.invert_xaxis()

        if not os.path.exists('./games'):
            os.makedirs('./games')
        plt.savefig('./games/' + name)
        plt.close(fig_map)
