import numpy as np
from functools import partial

from matplotlib import patches
import matplotlib.pyplot as plt


class QuoridorBoard:
    def __init__(self, n, board=None):
        assert n >= 3

        self.n = n
        self.board_len = 2 * self.n - 1

        # Board indices
        self.RED_PLAYER_IDX = 0
        self.BLUE_PLAYER_IDX = 1
        self.RED_WALLS_IDX = 2
        self.BLUE_WALLS_IDX = 3
        self.DRAW_IDX = 4

        self.history = {}

        if n % 2 == 0:
            midpoint_red = midpoint_blue = int(self.board_len / 2)
        else:
            midpoint_red = int(self.board_len / 2) + 1
            midpoint_blue = int(self.board_len / 2) - 1
        lastpoint = self.board_len - 1

        self.red_goal = (lastpoint, midpoint_red)
        self.blue_goal = (0, midpoint_blue)

        if board:
            self.setBoard(board)
        else:
            self.board = np.zeros((5, self.board_len, self.board_len), np.int16)

            # red player
            self.red_position = (0, midpoint_red)
            self.red_walls = 10
            self.board[self.RED_PLAYER_IDX, self.red_position[0], self.red_position[1]] = 1

            # blue player
            self.blue_position = (lastpoint, midpoint_blue)
            self.blue_walls = 10
            self.board[self.BLUE_PLAYER_IDX, self.blue_position[0], self.blue_position[1]] = 1

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

    def getGameEnded(self):
        if self.red_position[1] == self.red_goal[1]:
            return 1
        elif self.blue_position[1] == self.blue_goal[1]:
            return -1
        elif self.getRepetitions() >= 3:
            return 0
        return 0

    def getRepetitions(self):
        s = self.getBoardHashable()
        if s in self.history:
            return self.history[s]
        return 0

    def getBoardHashable(self):
        return hash(self.board.data)

    def setBoard(self, board):
        self.board = np.array(board, copy=True)
        self.red_position = self.findPlayer(1)
        self.red_walls = 10 - self.countWalls(1)
        self.blue_position = self.findPlayer(-1)
        self.blue_walls = 10 - self.countWalls(-1)

    def flipBoard(self):
        b = np.zeros(self.board.shape, dtype=int)
        b[2, :, :] = np.array(self.board[3, :, :], copy=True)
        b[3, :, :] = np.array(self.board[2, :, :], copy=True)
        b[0, :, :] = np.array(self.board[1, :, :], copy=True)
        b[1, :, :] = np.array(self.board[0, :, :], copy=True)

        for i in range(4):
            b[i, :, :] = np.flip(b[i, :, :], (0, 1))
        return b

    def makeCanonical(self, player):
        if player != 1:
            self.flipBoard()

    def getValidActions(self, player):
        return QuoridorUtils.GetValidMoves(self.board, player)

    def findPlayer(self, player):
        player_idx = self.RED_PLAYER_IDX if player == 1 else self.BLUE_PLAYER_IDX
        pos = np.where(self.board[player_idx, :, :] == 1)
        return pos[0][0], pos[1][0]

    def countWalls(self, player):
        player_idx = self.RED_PLAYER_IDX if player == 1 else self.BLUE_PLAYER_IDX
        return np.sum(self.board[player_idx, :, :]) / 3

    def executeAction(self, player, action):
        pawn_moves = 12
        vertical_wall_moves = pawn_moves + (self.n - 1) ** 2
        # Pawn Moves
        if 0 <= action < pawn_moves:
            y, x = self.findPlayer(player)
            self.actions[action](self.board, player, x, y)
        # Vertical Walls
        elif pawn_moves <= action < vertical_wall_moves:
            y = int((action - pawn_moves) % (self.n - 1)) * 2 + 1
            x = int((action - pawn_moves) / (self.n - 1)) * 2 + 1
            self.actions['vw'](self.board, player, x, y)
        # Horizontal Walls
        else:
            y = int((action - vertical_wall_moves) % (self.n - 1)) * 2 + 1
            x = int((action - vertical_wall_moves) / (self.n - 1)) * 2 + 1
            self.actions['hw'](self.board, player, x, y)

        return self.board

    def move(self, player, x, y, dx=0, dy=0):
        if player == 1:
            player_idx = self.RED_PLAYER_IDX
            self.red_position = (y + dy, x + dx)
        else:
            player_idx = self.BLUE_PLAYER_IDX
            self.blue_position = (y + dy, x + dx)

            # dy = -dy
            # dx = -dx

        self.board[player_idx, y, x] = 0
        self.board[player_idx, y + dy, x + dx] = 1

    def placeVerticalWall(self, player, x, y):
        if player == 1:
            player_idx = self.RED_WALLS_IDX
            self.red_walls -= 1
        else:
            player_idx = self.BLUE_WALLS_IDX
            self.blue_walls -= 1

            # boardsize = board.shape[1]-1
            # y = boardsize-y
            # x = boardsize-x

        self.board[player_idx, y, x] = 1
        self.board[player_idx, y + 1, x] = 1
        self.board[player_idx, y - 1, x] = 1

    def placeHorizontalWall(self, player, x, y):

        if player == 1:
            player_idx = self.RED_WALLS_IDX
            self.red_walls -= 1
        else:
            player_idx = self.BLUE_WALLS_IDX
            self.blue_walls -= 1

            # boardsize = board.shape[1]-1
            # y = boardsize-y
            # x = boardsize-x

        self.board[player_idx, y, x] = 1
        self.board[player_idx, y, x + 1] = 1
        self.board[player_idx, y, x - 1] = 1

    def board_pretty(self, invert_yaxis=False, path=None):
        """
        Simulator.visualize(path) # plot a path
        Simulator.visualize(path_full, path_short) # plot two paths

        path is a list for the trajectory. [x[0], y[0], x[1], y[1], ...]
        """

        if path is None:
            path = []

        fig_map, ax_map = plt.subplots(1, 1)

        # plot retangle obstacles
        for idx, x in np.ndenumerate(self.board[self.RED_WALLS_IDX, :, :]):
            idx = (idx[1], idx[0])
            if idx[0] % 2 == 1 or idx[1] % 2 == 1:
                rect = patches.Rectangle(idx, 1, 1, linewidth=0, facecolor='lightgray')
                ax_map.add_patch(rect)
            if x == 1:
                rect = patches.Rectangle(idx, 1, 1, linewidth=0, facecolor='darkred')
                ax_map.add_patch(rect)

        for idx, x in np.ndenumerate(self.board[self.BLUE_WALLS_IDX, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                rect = patches.Rectangle(idx, 1, 1, linewidth=0, facecolor='darkblue')
                ax_map.add_patch(rect)

        for idx, x in np.ndenumerate(self.board[self.RED_PLAYER_IDX, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                rect = patches.Rectangle(idx, 1, 1, linewidth=0, facecolor='r')
                ax_map.add_patch(rect)

        for idx, x in np.ndenumerate(self.board[self.BLUE_PLAYER_IDX, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                rect = patches.Rectangle(idx, 1, 1, linewidth=0, facecolor='b')
                ax_map.add_patch(rect)

        points = list(zip(path, path[1:]))[::2]
        for i, p in enumerate(points):
            if i != 0 and i != len(points) - 1:
                rect = patches.Rectangle(p, 1, 1, linewidth=0, facecolor='g', alpha=0.5)
                ax_map.add_patch(rect)

        ax_map.set_aspect('equal')
        ax_map.set_yticks(np.arange(0, self.board_len, 1))
        ax_map.set_xticks(np.arange(0, self.board_len, 1))
        ax_map.set_xlim([0, self.board_len])
        ax_map.set_ylim([0, self.board_len])
        if invert_yaxis:
            ax_map.invert_yaxis()
        plt.show()
