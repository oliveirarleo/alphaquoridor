""""

    This is a Regression Test Suite to automatically test all combinations of games and ML frameworks. Each test
    plays two quick games using an untrained neural network (randomly initialized) against a random player.

    In order for the entire test suite to run successfully, all the required libraries must be installed.  They are:
    Pytorch, Keras, Tensorflow.

     [ Games ]      Pytorch     Tensorflow  Keras
      -----------   -------     ----------  -----
    - Othello       [Yes]       [Yes]       [Yes]
    - TicTacToe                             [Yes]
    - Connect4                  [Yes]
    - Gobang                    [Yes]       [Yes]

"""

import unittest

from src.alphazero_general import Arena
from src.alphazero_general.MCTS import MCTS

from src.tictactoe.TicTacToeGame import TicTacToeGame
from src.tictactoe.TicTacToePlayers import *
from src.tictactoe.keras.NNet import NNetWrapper as TicTacToeKerasNNet

from src.tictactoe_3d.TicTacToePlayers import *

from src.othello.OthelloGame import OthelloGame
from src.othello.pytorch import NNetWrapper as OthelloPytorchNNet
from src.othello.tensorflow import NNetWrapper as OthelloTensorflowNNet
from src.othello.keras import NNetWrapper as OthelloKerasNNet

from src.connect4.Connect4Game import Connect4Game
from src.connect4.tensorflow.NNet import NNetWrapper as Connect4TensorflowNNet

from src.gobang.GobangGame import GobangGame
from src.gobang import NNetWrapper as GobangKerasNNet
from src.gobang import NNetWrapper as GobangTensorflowNNet

import numpy as np
from utils import *

class TestAllGames(unittest.TestCase):

    @staticmethod
    def execute_game_test(game, neural_net):
        rp = RandomPlayer(game).play

        args = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        mcts = MCTS(game, neural_net(game), args)
        n1p = lambda x: np.argmax(mcts.getActionProb(x, temp=0))

        arena = Arena.Arena(n1p, rp, game)
        print(arena.playGames(2, verbose=False))

    def test_othello_pytorch(self):
        self.execute_game_test(OthelloGame(6), OthelloPytorchNNet)

    def test_othello_tensorflow(self):
        self.execute_game_test(OthelloGame(6), OthelloTensorflowNNet)

    def test_othello_keras(self):
        self.execute_game_test(OthelloGame(6), OthelloKerasNNet)

    def test_tictactoe_keras(self):
        self.execute_game_test(TicTacToeGame(), TicTacToeKerasNNet)

    def test_connect4_tensorflow(self):
        self.execute_game_test(Connect4Game(), Connect4TensorflowNNet)

    def test_gobang_keras(self):
        self.execute_game_test(GobangGame(), GobangKerasNNet)

    def test_gobang_tensorflow(self):
        self.execute_game_test(GobangGame(), GobangTensorflowNNet)


if __name__ == '__main__':
    unittest.main()