import logging
import sys
import coloredlogs

sys.path.append('quoridor/pathfind/build')
from alphazero_general.Coach import Coach
from quoridor.pytorch.NNet import NNetWrapper as nn
from quoridor.QuoridorGame import QuoridorGame as Game
from alphazero_general.utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 80,
    'numEps': 1600,
    'tempThreshold': 8,
    'updateThreshold': 0.60,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 100,
    'arenaCompare': 40,
    'cpuct': 4,
    'cpuct_base': 19652,
    'cpuct_mult': 2,
    'dirichlet_alpha': 0.3,
    'eps': 0.25,

    # 'checkpoint': '/run/media/leleco/4EB5CC9A2FD2A5F9/dev/models/n5_v5/test',
    'checkpoint': '/run/media/leleco/4EB5CC9A2FD2A5F9/dev/models/n5v5/1600x100',
    'load_model': True,
    'load_examples': True,
    'load_folder_file': (
    '/run/media/leleco/4EB5CC9A2FD2A5F9/dev/models/n5v5/1600x100', 'quoridor_n5_v3_nnet_v2_torch_best.pth.tar'),
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
