from functools import partial

from quoridor.QuoridorPlayers import RandomPlayer, GreedyQuoridorPlayer, HumanQuoridorPlayer, AlphaQuoridor
from src.alphazero_general import Arena
from quoridor.QuoridorGame import QuoridorGame as Game

from alphazero_general.utils import *

g = Game(5)

hp = HumanQuoridorPlayer(g).play

args = dotdict({
                'numMCTSSims': 200,
                'arenaCompare': 2,
                'cpuct': 2.5,
                'cpuct_base': 19652,
                'cpuct_mult': 2,
                'dirichlet_alpha': 0.3,
                'eps': 0.25,
            })

aq = AlphaQuoridor(g, './git_models/', 'quoridor_n5v5_nnet_v3_1600_100.pth.tar', args=args, temp=0)
arena = Arena.Arena(aq.play, hp, g, display=partial(g.display))

print(arena.playGames(args.arenaCompare, verbose=False))
