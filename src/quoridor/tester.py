import math

import numpy as np

from quoridor.QuoridorGame import QuoridorGame
from quoridor.QuoridorLogic import QuoridorBoard
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class QuoridorEngineTester:
    def __init__(self, n):
        self.n = n
        self.n_walls = n - 1

        self.game = QuoridorGame(self.n)
        self.init_board = self.game.getInitBoard()
        self.board = self.init_board

        self.pawn_actions = 12
        self.vwall_actions = self.pawn_actions + self.n_walls * self.n_walls

        self.pawn_action_translator = {
            'N': 0,
            'S': 1,
            'E': 2,
            'W': 3,
            'JN': 4,
            'JS': 5,
            'JE': 6,
            'JW': 7,
            'JNE': 8,
            'JNW': 9,
            'JSE': 10,
            'JSW': 11,
        }

    def board_to_string(self):
        walls = self.board[0, :, :] + self.board[1, :, :] + 2 * self.board[2, :, :] + 3 * self.board[3,:, :]
        board_string = 'board:' + str(walls.shape) + '\n'
        for line in walls:
            board_string += str(line) + '\n'
        return board_string

    def board_pretty(self, invert_yaxis=False):
        """
        Simulator.visualize(path) # plot a path
        Simulator.visualize(path_full, path_short) # plot two paths

        path is a list for the trajectory. [x[0], y[0], x[1], y[1], ...]
        """

        fig_map, ax_map = plt.subplots(1, 1)

        # plot retangle obstacles
        for idx, x in np.ndenumerate(self.board[0, :, :]):
            idx = (idx[1], idx[0])
            if idx[0] % 2 == 1 or idx[1] % 2 == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='lightgray')
                # Add the patch to the Axes
                ax_map.add_patch(rect)
            if x == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='darkred')
                # Add the patch to the Axes
                ax_map.add_patch(rect)

        for idx, x in np.ndenumerate(self.board[1, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='darkblue')
                # Add the patch to the Axes
                ax_map.add_patch(rect)
                # Add the patch to the Axes
                ax_map.add_patch(rect)

        for idx, x in np.ndenumerate(self.board[2, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='r')
                # Add the patch to the Axes
                ax_map.add_patch(rect)

        for idx, x in np.ndenumerate(self.board[3, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='b')
                # Add the patch to the Axes
                ax_map.add_patch(rect)

        ax_map.set_aspect('equal')
        ax_map.set_yticks(np.arange(0, 17, 2))
        ax_map.set_xticks(np.arange(0, 17, 2))
        ax_map.set_xlim([0, 17])
        ax_map.set_ylim([0, 17])
        if invert_yaxis:
            ax_map.invert_yaxis()
        plt.show()

    def placeWall(self, player, x, y, is_vertical):
        if is_vertical:
            print('VW' + str(x) + ' ' + str(y))
            self.board = self.game.getNextState(self.board, player, self.pawn_actions + x * self.n_walls + y)

        else:
            print('HW' + str(x) + ' ' + str(y))
            self.board = self.game.getNextState(self.board, player, self.vwall_actions + x * self.n_walls + y)

    def move(self, player, direction):
        self.board = self.game.getNextState(self.board, player, self.pawn_action_translator[direction])

    def printValidActions(self, player):
        valid_actions = self.game.getValidMoves(self.board, player)

        print('Valid Moves', sum(valid_actions), '/', len(valid_actions))

        pawn_moves = self.pawn_action_translator.keys()
        for i, m in enumerate(pawn_moves):
            print(m, valid_actions[i], end=', ')
        print()

        # Print vwalls
        print('Vwalls:')
        for i in range(self.n_walls):
            print(i, end=' ')
            for j in range(self.n_walls):
                print(valid_actions[self.pawn_actions + self.n_walls * i + j], end=' ')
            print()
        print(' ', end=' ')
        for i in range(self.n_walls):
            print(i, end=' ')
        print()

        # Print hwalls
        print('Hwalls:')
        num_vwalls = self.pawn_actions + self.n_walls ** 2
        for i in range(self.n_walls):
            print(i, end=' ')
            for j in range(self.n_walls):
                print(valid_actions[num_vwalls + self.n_walls * i + j], end=' ')
            print()

        print(' ', end=' ')
        for i in range(self.n_walls):
            print(i, end=' ')
        print()

    def getValidActions(self, player):
        return self.game.getValidMoves(self.board, player)

    def executeAction(self, action, player):
        self.board = self.game.getNextState(self.board, player, action)

    def printActionType(self, action, player):
        if player == 1:
            ptype = 'red'
        else:
            ptype = 'blu'

        if action < self.pawn_actions:
            print(ptype, 'Move', action)
        elif action < self.vwall_actions:
            print(ptype, 'VWall x:', int((action - self.pawn_actions) / self.n_walls), 'y:', (action - self.pawn_actions) % self.n_walls)
        else:
            print(ptype, 'HWall x:', int((action - self.vwall_actions) / self.n_walls), 'y:', (action - self.vwall_actions) % self.n_walls)

def main():
    tester = QuoridorEngineTester(9)
    # tester.printValidActions(1)
    player = 1
    for _ in range(30):
        valid_actions = tester.getValidActions(player)
        pi = [a / sum(valid_actions) for a in valid_actions]
        action = np.random.choice(len(valid_actions), p=pi)
        print('Valid Actions', sum(valid_actions), '/', len(valid_actions))

        tester.printActionType(action, player)

        tester.executeAction(action, player)
        # tester.printValidActions(player)
        player = - player
        # print()

    tester.board_pretty(True)
    # print(tester.board_to_string())


if __name__ == "__main__":
    main()
