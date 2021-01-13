import logging
import sys
import coloredlogs

sys.path.append('quoridor/pathfind/build')
from alphazero_general.Coach import Coach
from quoridor.pytorch.NNet import NNetWrapper as nn
from quoridor.QuoridorGame import QuoridorGame as Game
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,
    'numEps': 600,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,  #
    'updateThreshold': 0.60,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 100,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 2.5,
    'cpuct_base': 19652,
    'cpuct_mult': 2,

    'checkpoint': '/run/media/leleco/4EB5CC9A2FD2A5F9/dev/models/n5_v3/100epsT2',
    'load_model': False,
    'load_examples': False,
    'load_folder_file': ('/run/media/leleco/4EB5CC9A2FD2A5F9/dev/models/n5_v1/cpuct_new/', 'quoridor_n5_v1_nnetv0_torch_checkpoint.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(5)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', *args.load_folder_file)
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_examples:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')

    c.learn()


if __name__ == "__main__":
    main()
