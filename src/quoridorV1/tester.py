import sys
import os

import coloredlogs
import numpy as np
import logging

from alphazero_general.Arena import Arena
from utils import dotdict

sys.path.append(os.path.join(os.path.dirname(__file__), 'pathfind/build'))
import QuoridorUtils

from alphazero_general.MCTS import MCTS

from quoridorV1.pytorch.NNet import NNetWrapper as nn
from quoridorV1.QuoridorGame import QuoridorGame as Game, QuoridorGame

log = logging.getLogger(__name__)

pawn_action_translator = {
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


class QuoridorEngineTester:

    def __init__(self, n):
        self.n = n
        self.n_walls = n - 1

        self.game = Game(self.n)

        self.pawn_actions = 12
        self.vwall_actions = self.pawn_actions + self.n_walls * self.n_walls

    @staticmethod
    def placeWall(game, board, player, x, y, is_vertical):
        n_walls = game.n - 1
        pawn_actions = 12
        vwall_actions = pawn_actions + n_walls ** 2
        if is_vertical:
            print('VW: ' + str(x) + ' ' + str(y))
            return game.getNextState(board, player, pawn_actions + y * n_walls + x)

        else:
            print('HW: ' + str(x) + ' ' + str(y))
            return game.getNextState(board, player, vwall_actions + y * n_walls + x)

    @staticmethod
    def printValidActions(game, board, player):
        n_walls = game.n - 1
        pawn_actions = 12

        valid_actions = game.getValidActions(board, player)
        num_vwalls = pawn_actions + n_walls ** 2

        print('Valid Moves', sum(valid_actions), '/', len(valid_actions))

        pawn_moves = pawn_action_translator.keys()
        for i, m in enumerate(pawn_moves):
            print(m, valid_actions[i], end=', ')
        print()

        # Print vwalls
        print('Vwalls:')
        for i in range(n_walls-1, -1, -1):
            print(i, end=' ')
            for j in range(n_walls):
                print(valid_actions[pawn_actions + n_walls * j + i], end=' ')
            print()
        print(' ', end=' ')
        for i in range(n_walls):
            print(i, end=' ')
        print()

        # Print hwalls
        print('Hwalls:')
        for i in range(n_walls-1, -1, -1):
            print(i, end=' ')
            for j in range(n_walls):
                print(valid_actions[num_vwalls + n_walls * j + i], end=' ')
            print()

        print(' ', end=' ')
        for i in range(n_walls):
            print(i, end=' ')
        print()

    @staticmethod
    def printActionType(game, board, action, player):
        if player == 1:
            ptype = '--Red'
        else:
            ptype = '--Blu'
        n_walls = game.n - 1
        pawn_actions = 12
        vwall_actions = pawn_actions + n_walls ** 2

        if action < pawn_actions:
            print(ptype, 'Move', action)
        elif action < vwall_actions:
            print(ptype, 'VWall x:', int((action - pawn_actions) / n_walls), 'y:',
                  (action - pawn_actions) % n_walls)
            x = int((action - pawn_actions) / n_walls) * 2 +1
            y = ((action - pawn_actions) % n_walls) * 2 +1
            print(QuoridorUtils.pathExistsForPlayers((board.red_walls_board + board.blue_walls_board),
                                                     board.red_position[0], board.red_position[1], board.red_goal[0],
                                                     board.red_goal[1],
                                                     board.blue_position[0], board.blue_position[1], board.blue_goal[0],
                                                     board.blue_goal[1],
                                                     x, y, True))
        else:
            print(ptype, 'HWall x:', int((action - vwall_actions) / n_walls), 'y:',
                  (action - vwall_actions) % n_walls)
            x = int((action - vwall_actions) / n_walls) * 2 + 1
            y = ((action - vwall_actions) % n_walls) * 2 + 1
            print(QuoridorUtils.pathExistsForPlayers((board.red_walls_board + board.blue_walls_board),
                                                     board.red_position[0], board.red_position[1], board.red_goal[0],
                                                     board.red_goal[1],
                                                     board.blue_position[0], board.blue_position[1], board.blue_goal[0],
                                                     board.blue_goal[1],
                                                     x, y, True))


def main():
    args = dotdict({
        'numIters': 1000,
        'numEps': 200,  # Number of complete self-play games to simulate during a new iteration.
        'tempThreshold': 15,  #
        'updateThreshold': 0.60,
        # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
        'numMCTSSims': 1000,  # Number of games moves for MCTS to simulate.
        'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
        'cpuct': 2.5,
        'cpuct_base': 19652,
        'cpuct_mult': 2,

        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('./dev/models/v0_n9', 'quoridor_n9_v0_nnetv0_torch_checkpoint_1.pth.tar'),
        'numItersForTrainExamplesHistory': 20,

    })

    coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
    log.info('Loading %s...', Game.__name__)
    g = Game(5)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    pnet = nn(g)

    nnet.load_checkpoint(folder='/home/leleco/proj/pfg/alphaquoridor/pretrained_models/quoridor/v1_n5/', filename='quoridor_n5_v1_nnetv0_torch_best.pth.tar')
    pnet.load_checkpoint(folder='/home/leleco/proj/pfg/alphaquoridor/pretrained_models/quoridor/v1_n5/', filename='quoridor_n5_v1_nnetv0_torch_best.pth.tar')

    pmcts = MCTS(g, pnet, args)
    nmcts = MCTS(g, nnet, args)
    log.info('PITTING AGAINST PREVIOUS VERSION')
    temp = 0
    arena = Arena(lambda x: np.random.choice(g.getActionSize(), p=pmcts.getActionProb(x, temp=temp)),
                  lambda x: np.random.choice(g.getActionSize(), p=pmcts.getActionProb(x, temp=temp)), g, g.display)
    pwins, nwins, draws = arena.playGames(40, verbose=True)

    log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
    # game = QuoridorGame(5)
    # board = game.getInitBoard()
    # board, player = game.getNextState(board, 1, 0)
    # board, player = QuoridorEngineTester.placeWall(game, board, 1, 0, 1, False)
    # # QuoridorEngineTester.printValidActions(game, board, 1)
    # # board, player = game.getNextState(board, -1, 3)
    # path, len = QuoridorUtils.FindPath(board.blue_position, board.blue_goal, (board.red_walls_board+board.blue_walls_board),game.board_len,game.board_len)
    # board.plot_board(path=path)
    # print(path)
    # print(board.blue_position)
    # print(board.blue_goal)
    # print(QuoridorUtils.PathExists(board.blue_position[0], board.blue_position[1],
    #                                board.blue_goal[0], board.blue_goal[1], (board.red_walls_board+board.blue_walls_board),game.board_len,game.board_len))
    # print(np.rot90(board.red_walls_board+board.blue_walls_board))

    # game = QuoridorGame(5)
    # ww = 0
    # bw = 0
    # for _ in tqdm(range(1), desc="Self Play"):
    #     board = game.getInitBoard()
    #     i = 1
    #     while True:
    #         # print(i)
    #         flip = i%2==0
    #         path, lendd = QuoridorUtils.FindPath(board.blue_position, board.blue_goal,
    #                                            (board.red_walls_board + board.blue_walls_board), game.board_len,
    #                                            game.board_len)
    #         board.plot_board(path=path)
    #         i += 1
    #         valid_actions = game.getValidActions(board, 1)
    #         pi = [a / sum(valid_actions) for a in valid_actions]
    #         action = np.random.choice(len(valid_actions), p=pi)
    #         QuoridorEngineTester.printActionType(game, board, action, 1)
    #         # print(board.red_position)
    #         # print(board.blue_position)
    #         # QuoridorEngineTester.printValidActions(game, board, 1)
    #
    #         board, player = game.getNextState(board, 1, action)
    #
    #         path, lendd = QuoridorUtils.FindPath(board.blue_position, board.blue_goal,
    #                                              (board.red_walls_board + board.blue_walls_board), game.board_len,
    #                                              game.board_len)
    #         board.plot_board(path=path)
    #         # print(QuoridorUtils.PathExists(board.blue_position[0], board.blue_position[1],
    #         #                                board.blue_goal[0], board.blue_goal[1], (board.red_walls_board+board.blue_walls_board),game.board_len,game.board_len))
    #         #
    #         board = game.getCanonicalForm(board, player)
    #
    #         if game.getGameEnded(board, 1) != 0:
    #             # print('GAME ENDED', game.getGameEnded(board, 1))
    #             break
    #     # print(i)
    #     if i % 2:
    #         ww += 1
    #     else:
    #         bw += 1
    #
    #     board.plot_board(invert_yaxis=(not flip))
    # print(ww, bw)

    # args = dotdict({
    #     'numIters': 1000,
    #     'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    #     'tempThreshold': 15,  #
    #     'updateThreshold': 0.6,
    #     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    #     'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    #     'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
    #     'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    #     'cpuct': 1,
    #
    #     'checkpoint': './temp/',
    #     'load_model': False,
    #     'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    #     'numItersForTrainExamplesHistory': 20,
    #
    # })
    #
    # log.info('Loading %s...', Game.__name__)
    # game = Game(3)
    #
    # log.info('Loading %s...', nn.__name__)
    # nnet = nn(game)
    # mcts = MCTS(game, nnet, args)
    # board = game.getInitBoard()
    #
    # episodeStep = 0
    # curPlayer = 1
    # while True:
    #     print(episodeStep)
    #     board.plot_board()
    #     episodeStep += 1
    #     canonicalBoard = game.getCanonicalForm(board, curPlayer)
    #     temp = int(episodeStep < args.tempThreshold)
    #
    #     pi = mcts.getActionProb(canonicalBoard, temp=temp)
    #     # if curPlayer == -1:
    #     #     pi = board.piSymmetries(pi)
    #     print('pi', pi)
    #     la = game.getValidActions(board, curPlayer)
    #     action = np.random.choice(len(pi), p=pi)
    #     QuoridorEngineTester.printActionType(game, action, curPlayer)
    #     board, curPlayer = game.getNextState(board, curPlayer, action)
    #
    #     r = game.getGameEnded(board, curPlayer)
    #
    #     if r != 0:
    #         print('ENDE')
    #         break
    # board.plot_board()

if __name__ == "__main__":
    main()
