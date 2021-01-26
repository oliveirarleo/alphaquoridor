from functools import partial

from quoridor.QuoridorPlayers import RandomPlayer, GreedyQuoridorPlayer, HumanQuoridorPlayer, AlphaQuoridor
from src.alphazero_general import Arena
from quoridor.QuoridorGame import QuoridorGame as Game

from alphazero_general.utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = Game(5)

# all players
rp = RandomPlayer(g).play
gp = GreedyQuoridorPlayer(g).play
hp = HumanQuoridorPlayer(g).play


folder = '/run/media/leleco/4EB5CC9A2FD2A5F9/dev/models/pit_models/'

args = dotdict({
                'numMCTSSims': 200,
                'arenaCompare': 200,
                'cpuct': 2.5,
                'cpuct_base': 19652,
                'cpuct_mult': 2,
                'dirichlet_alpha': 0.3,
                'eps': 0.25,
            })

# aq = AlphaQuoridor(g, folder, 'quoridor_n5_v3_nnet_v2_1600x50x28.pth.tar', args=args, temp=0)
# aqv2 = AlphaQuoridorV2(g, folder, 'quoridor_n5_v3_nnet_v4_800x100x26.pth.tar', args=args, temp=0)
# arena = Arena.Arena(aq.play, aqv2.play, g, display=partial(g.display, save_folder='teste4'))
# print(arena.playGames(args.arenaCompare, verbose=False))

aq = AlphaQuoridor(g, folder, 'quoridor_n5_v3_nnet_v2_600x50x100.pth.tar', args=args, temp=0)
aqv2 = AlphaQuoridor(g, folder, 'quoridor_n5_v3_nnet_v2_600x100x43.pth.tar', args=args, temp=0)
arena = Arena.Arena(aq.play, aqv2.play, g, display=partial(g.display, save_folder='teste6'))

