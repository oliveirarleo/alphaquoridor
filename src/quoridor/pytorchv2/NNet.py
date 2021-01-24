import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from alphazero_general.NeuralNet import NeuralNet

import torch
import torch.optim as optim

from .QuoridorBigNNet import QuoridorNNet as qnnet

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 128,
    'cuda': torch.cuda.is_available(),
    'num_channels': 256,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game, nn_args=None):
        super().__init__(game)

        if nn_args is None:
            self.nn_args = args
        else:
            self.nn_args = nn_args

        self.nnet = qnnet(game, self.nn_args)
        self.boards, self.walls, self.values = game.getBoardSize()
        self.action_size = game.getActionSize()
        if self.nn_args.cuda:
            self.nnet.cuda()

    def __str__(self):
        return 'nnet_v2_torch'

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.nn_args.epochs):
            # print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.nn_args.batch_size)

            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.nn_args.batch_size)
                nn_input, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards, walls, values = list(zip(*[i for i in nn_input]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                walls = torch.FloatTensor(np.array(walls).astype(np.float64))
                values = torch.FloatTensor(np.array(values).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.nn_args.cuda:
                    boards = boards.contiguous().cuda()
                    walls = walls.contiguous().cuda()
                    values = values.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet(boards, walls, values)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, board):
        """
        board: np array with board
        """
        board, wall, value = board.getBoard()
        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        wall = torch.FloatTensor(wall.astype(np.float64))
        value = torch.FloatTensor(value.astype(np.float64))
        if self.nn_args.cuda:
            board = board.contiguous().cuda()
            wall = wall.contiguous().cuda()
            value = value.contiguous().cuda()
        board = board.view(self.boards[2], self.boards[0], self.boards[1])
        wall = wall.view(self.walls[2], self.walls[0], self.walls[1])
        value = value.view(self.values)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board, wall, value)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if self.nn_args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
