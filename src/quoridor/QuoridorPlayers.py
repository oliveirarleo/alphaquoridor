import numpy as np

from alphazero_general.MCTS import MCTS
from alphazero_general.utils import dotdict
from quoridor.pytorch.NNet import NNetWrapper as nn
from quoridor.pytorchv2.NNet import NNetWrapper as nnv2


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        actions = self.game.getValidActions(board, 1)
        # print(actions)
        return np.random.choice(len(actions), p=np.array(actions) / sum(actions))


class HumanQuoridorPlayer:
    def __init__(self, game):
        self.game = game
        self.pawn_action_translator = ['N', 'S', 'E', 'W', 'JN', 'JS', 'JE', 'JW', 'JNE', 'JSW', 'JNW', 'JSE', ]

    def play(self, board):
        board.plot(save=False)
        valid = self.game.getValidActions(board, 1)
        num_walls = self.game.n - 1

        while True:
            input_move = input().upper()
            try:
                if input_move in self.pawn_action_translator:
                    action = self.pawn_action_translator.index(input_move)
                    return action
                elif input_move.startswith('V'):
                    input_list = input_move.split(" ")
                    action = 12 + int(input_list[1]) * num_walls + int(input_list[2])
                    if valid[action] == 1:
                        return action
                    raise Exception
                elif input_move.startswith('H'):
                    input_list = input_move.split(" ")
                    action = 12 + num_walls ** 2 + int(input_list[1]) * num_walls + int(input_list[2])
                    if valid[action] == 1:
                        return action
                    raise Exception
                else:
                    raise Exception
            except:
                print('INVALID MOVE!', input_move)


class GreedyQuoridorPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        ac = board.shortestPathActions()
        greedy_actions = np.argwhere(ac == np.amax(ac))
        return greedy_actions[0][0]


class AlphaQuoridor:
    def __init__(self, game, nn_folder, nn_name, args=None, temp=0):

        self.game = game
        if args is None:
            self.args = dotdict({
                'tempThreshold': 15,
                'updateThreshold': 0.60,
                'maxlenOfQueue': 200000,
                'numMCTSSims': 1000,
                'cpuct': 2.5,
                'cpuct_base': 19652,
                'cpuct_mult': 2,
            })
        else:
            self.args = args

        nnet = nn(self.game)
        nnet.load_checkpoint(folder=nn_folder, filename=nn_name)
        self.nmcts = MCTS(self.game, nnet, args)
        self.temp = temp

    def play(self, board):
        return np.random.choice(self.game.getActionSize(), p=self.nmcts.getActionProb(board, temp=self.temp))


class AlphaQuoridorV2:
    def __init__(self, game, nn_folder, nn_name, args=None, temp=0):

        self.game = game
        if args is None:
            self.args = dotdict({
                'tempThreshold': 15,
                'updateThreshold': 0.60,
                'maxlenOfQueue': 200000,
                'numMCTSSims': 1000,
                'arenaCompare': 40,
                'cpuct': 2.5,
                'cpuct_base': 19652,
                'cpuct_mult': 2,
            })
        else:
            self.args = args

        nnet = nnv2(self.game)
        nnet.load_checkpoint(folder=nn_folder, filename=nn_name)
        self.nmcts = MCTS(self.game, nnet, args)
        self.temp = temp

    def play(self, board):
        return np.random.choice(self.game.getActionSize(), p=self.nmcts.getActionProb(board, temp=self.temp))
