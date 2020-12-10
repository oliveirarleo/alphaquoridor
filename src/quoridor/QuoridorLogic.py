import numpy as np
from functools import partial


class QuoridorBoard:
    def __init__(self, n):
        self.n = n
        self.board_len = 2 * self.n - 1

        self.board = None

        self.red_position = None
        self.red_goal = None
        self.red_walls = None

        self.blue_position = None
        self.blue_goal = None
        self.blue_walls = None

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
            8: partial(self.move, dx=-2, dy=+2),
            # JNW
            9: partial(self.move, dx=+2, dy=+2),
            # JSE
            10: partial(self.move, dx=-2, dy=-2),
            # JSW
            11: partial(self.move, dx=+2, dy=-2),
            # PLACE VERTICAL WALL
            'vw': self.placeVerticalWall,
            # PLACE HORIZONTAL WALL
            'hw': self.placeHorizontalWall,
        }

    def setInitBoard(self):
        # empty board
        self.board = np.zeros((4, self.board_len, self.board_len), np.int16)

        midpoint = int(self.board_len / 2)
        lastpoint = self.board_len - 1

        # red player position
        self.red_position = (0, midpoint)
        self.red_goal = (lastpoint, midpoint)
        self.red_walls = 10
        self.board[2, self.red_position[0], self.red_position[1]] = 1

        # blue player position
        self.blue_position = (lastpoint, midpoint)
        self.blue_goal = (0, midpoint)
        self.blue_walls = 10
        self.board[3, self.blue_position[0], self.blue_position[1]] = 1

        return self.board

    def setBoard(self, board):
        self.board = np.copy(board)

    def findPlayer(self, player):
        player = 2 if player == 1 else 3
        pos = np.where(self.board[player, :, :] == 1)
        return pos[0][0], pos[1][0]

    def countRedWalls(self):
        return np.sum(self.board[0, :, :]) / 3

    def countBlueWalls(self):
        return np.sum(self.board[1, :, :]) / 3

    def executeAction(self, player, action):
        pawn_moves = 12
        vertical_wall_moves = pawn_moves + (self.n - 1) ** 2
        # Pawn Moves
        if 0 <= action < pawn_moves:
            y, x = self.findPlayer(player)
            self.actions[action](self.board, player, x, y)
        elif pawn_moves <= action < vertical_wall_moves:
            y = int((action-pawn_moves) % (self.n - 1)) * 2 + 1
            x = int((action-pawn_moves) / (self.n - 1)) * 2 + 1
            self.actions['vw'](self.board, player, x, y)
        else:
            y = int((action-vertical_wall_moves) % (self.n - 1)) * 2 + 1
            x = int((action-vertical_wall_moves) / (self.n - 1)) * 2 + 1
            self.actions['hw'](self.board, player, x, y)

        return self.board

    @staticmethod
    def move(board, player, x, y, dx=0, dy=0):
        player_idx = 2 if player == 1 else 3
        board[player_idx, y, x] = 0
        board[player_idx, y + dy, x + dx] = 1
        return board

    @staticmethod
    def placeVerticalWall(board, player, x, y):
        player_idx = 0 if player == 1 else 1
        board[player_idx, y, x] = 1
        board[player_idx, y + 1, x] = 1
        board[player_idx, y - 1, x] = 1
        return board

    @staticmethod
    def placeHorizontalWall(board, player, x, y):
        player_idx = 0 if player == 1 else 1
        board[player_idx, y, x] = 1
        board[player_idx, y, x + 1] = 1
        board[player_idx, y, x - 1] = 1
        return board

