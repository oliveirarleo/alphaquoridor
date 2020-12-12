import sys
import os

import numpy as np
import logging
from utils import dotdict
import coloredlogs

sys.path.append(os.path.join(os.path.dirname(__file__), 'pathfind/build'))
import QuoridorUtils
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from alphazero_general.MCTS import MCTS

from quoridor.pytorch.NNet import NNetWrapper as nn
from quoridor.QuoridorGame import QuoridorGame as Game

log = logging.getLogger(__name__)


class QuoridorEngineTester:
    def __init__(self, n):
        self.n = n
        self.n_walls = n - 1

        self.game = Game(self.n)
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
        walls = self.board[0, :, :] + self.board[1, :, :] + 2 * self.board[2, :, :] + 3 * self.board[3, :, :]
        board_string = 'board:' + str(walls.shape) + '\n'
        for line in walls:
            board_string += str(line) + '\n'
        return board_string

    @staticmethod
    def board_pretty(board, invert_yaxis=False, path=[]):
        """
        Simulator.visualize(path) # plot a path
        Simulator.visualize(path_full, path_short) # plot two paths

        path is a list for the trajectory. [x[0], y[0], x[1], y[1], ...]
        """

        fig_map, ax_map = plt.subplots(1, 1)

        # plot retangle obstacles
        for idx, x in np.ndenumerate(board[0, :, :]):
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

        for idx, x in np.ndenumerate(board[1, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='darkblue')
                # Add the patch to the Axes
                ax_map.add_patch(rect)
                # Add the patch to the Axes
                ax_map.add_patch(rect)

        for idx, x in np.ndenumerate(board[2, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='r')
                # Add the patch to the Axes
                ax_map.add_patch(rect)

        for idx, x in np.ndenumerate(board[3, :, :]):
            idx = (idx[1], idx[0])
            if x == 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(idx, 1, 1,
                                         linewidth=1, facecolor='b')
                # Add the patch to the Axes
                ax_map.add_patch(rect)

        points = list(zip(path, path[1:]))[::2]
        for i, p in enumerate(points):
            if i != 0 and i != len(points) - 1:
                # Create a Rectangle patch
                rect = patches.Rectangle(p, 1, 1,
                                         linewidth=1, facecolor='g', alpha=0.5)
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
        valid_actions = self.game.getValidActions(self.board, player)

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
                print(valid_actions[self.pawn_actions + self.n_walls * j + i], end=' ')
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
                print(valid_actions[num_vwalls + self.n_walls * j + i], end=' ')
            print()

        print(' ', end=' ')
        for i in range(self.n_walls):
            print(i, end=' ')
        print()

    def getValidActions(self, player):
        return self.game.getValidActions(self.board, player)

    def executeAction(self, action, player):
        self.board, next_player = self.game.getNextState(self.board, player, action)
        return next_player

    def printActionType(self, action, player):
        if player == 1:
            ptype = 'red'
        else:
            ptype = 'blu'

        if action < self.pawn_actions:
            print(ptype, 'Move', action)
        elif action < self.vwall_actions:
            print(ptype, 'VWall x:', int((action - self.pawn_actions) / self.n_walls), 'y:',
                  (action - self.pawn_actions) % self.n_walls)
        else:
            print(ptype, 'HWall x:', int((action - self.vwall_actions) / self.n_walls), 'y:',
                  (action - self.vwall_actions) % self.n_walls)

    def printPath(self, player):
        end = [self.n - 1, 2 * self.n - 2] if player == 1 else [self.n - 1, 0]

        player_idx = 2 if player == 1 else 3
        pos = np.where(self.board[player_idx, :, :] == 1)
        walls = self.board[1, :, :] + self.board[0, :, :]

        start = [pos[1][0], pos[0][0]]

        print('pos', start)
        print('goal', end)

        path, steps = QuoridorUtils.FindPath(start, end, walls, walls.shape[0], walls.shape[0])

        print('steps', steps)
        self.board_pretty(self.board, True, path)


def main():
    # tester = QuoridorEngineTester(3)
    # # tester.printValidActions(1)
    # player = 1
    # for _ in range(40):
    #     valid_actions = tester.getValidActions(player)
    #     if sum(valid_actions) == 0:
    #         break
    #     pi = [a / sum(valid_actions) for a in valid_actions]
    #     action = np.random.choice(len(valid_actions), p=pi)
    #     print('Valid Actions', sum(valid_actions), '/', len(valid_actions))
    #     # tester.printValidActions(player)
    #
    #     tester.printActionType(action, player)
    #     tester.board_pretty()
    #
    #     player = tester.executeAction(action, player)
    #     if tester.game.getGameEnded(tester.board, player) != 0:
    #         print('GAME ENDED')
    #         tester.board_pretty()
    #         break

    # game = QuoridorGame(3)
    # ww = 0
    # bw = 0
    # for j in range(200):
    #     board = game.getInitBoard()
    #     i = 1
    #     # QuoridorEngineTester.board_pretty(board)
    #     while True:
    #         # print(i)
    #         i += 1
    #         valid_actions = game.getValidActions(board, 1)
    #         pi = [a / sum(valid_actions) for a in valid_actions]
    #         action = np.random.choice(len(valid_actions), p=pi)
    #         board, player = game.getNextState(board, 1, action)
    #
    #         board = game.getCanonicalForm(board, player)
    #         # QuoridorEngineTester.board_pretty(board)
    #         if game.getGameEnded(board, 1) != 0:
    #             # print('GAME ENDED', game.getGameEnded(board, 1))
    #             break
    #     # print(i)
    #     if i % 2:
    #         ww += 1
    #     else:
    #         bw+=1
    # QuoridorEngineTester.board_pretty(board)
    # # print(tester.board_to_string())
    # print(ww, bw)

    args = dotdict({
        'numIters': 1000,
        'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,  #
        'updateThreshold': 0.6,
        # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
        'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 1,

        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
        'numItersForTrainExamplesHistory': 20,

    })

    log.info('Loading %s...', Game.__name__)
    game = Game(3)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(game)
    mcts = MCTS(game, nnet, args)
    board = game.getInitBoard()

    episodeStep = 0
    curPlayer = 1
    while True:
        print(episodeStep)
        QuoridorEngineTester.board_pretty(board)
        episodeStep += 1
        canonicalBoard = game.getCanonicalForm(board, curPlayer)
        temp = int(episodeStep < args.tempThreshold)

        pi = mcts.getActionProb(canonicalBoard, temp=temp)
        print('pi', pi)
        la = game.getValidActions(board, curPlayer)
        action = np.random.choice(len(pi), p=pi)
        board, curPlayer = game.getNextState(board, curPlayer, action)

        r = game.getGameEnded(board, curPlayer)

        if r != 0:
            break


if __name__ == "__main__":
    main()
