import numpy as np


class Board:
    def __init__(self):
        self.board_width = 17
        self.board_height = 17
        self.num_players = 2

        # empty board
        self.board = np.zeros((self.board_width, self.board_height, self.num_players + 1), int)

        # red player position
        self.red_position = (int(self.board_width / 2), 0)
        self.red_goal = (int(self.board_width / 2), self.board_height - 1)
        self.red_walls = 10
        self.board[self.red_position[0]][self.red_position[1]][1] = 1

        # blue player position
        self.blue_position = (int(self.board_width / 2), self.board_height - 1)
        self.blue_goal = (int(self.board_width / 2), 0)
        self.blue_walls = 10
        self.board[self.blue_position[0]][self.blue_position[1]] = 1

    # @staticmethod
    # def flipBoard(board):
    #     board[:]
    #     return np.flip(board, axis=1)
