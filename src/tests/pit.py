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


# nnet players
n1 = NNet(g)
n1.load_checkpoint('/run/media/leleco/4EB5CC9A2FD2A5F9/dev/models/n5_v3/100s5niftehnegdraw/', 'quoridor_n5_v3_nnet_v2_torch_best.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct': 2.5, 'cpuct_base': 19652, 'cpuct_mult': 2})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

n2 = NNet(g)
n2.load_checkpoint('/run/media/leleco/4EB5CC9A2FD2A5F9/dev/models/n5_v3/50s5niftehnegdraw/', 'quoridor_n5_v3_nnet_v2_torch_best.pth.tar')
args2 = dotdict({'numMCTSSims': 50, 'cpuct': 2.5, 'cpuct_base': 19652, 'cpuct_mult': 2})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))


arena = Arena.Arena(n1p, n2p, g, display=g.display)

print(arena.playGames(10, verbose=True))
