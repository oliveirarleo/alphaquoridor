from quoridor.QuoridorPlayers import RandomPlayer, GreedyQuoridorPlayer, HumanQuoridorPlayer
from src.alphazero_general import Arena
from src.alphazero_general.MCTS import MCTS
from quoridor.QuoridorGame import QuoridorGame as Game
from quoridor.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = Game(5)

# all players
rp = RandomPlayer(g).play
gp = GreedyQuoridorPlayer(g).play
hp = HumanQuoridorPlayer(g).play

n1p = gp

n2p = hp


arena = Arena.Arena(n1p, n2p, g, display=g.display)

print(arena.playGames(10, verbose=True))
