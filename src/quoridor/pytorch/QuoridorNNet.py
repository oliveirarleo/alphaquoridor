import sys

sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuoridorNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.boards, self.walls, self.values = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(QuoridorNNet, self).__init__()
        self.conv1 = nn.Conv2d(self.boards[2], self.args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1)

        self.conv5 = nn.Conv2d(self.walls[2], self.args.num_channels, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.args.num_channels)
        self.bn2 = nn.BatchNorm2d(self.args.num_channels)
        self.bn3 = nn.BatchNorm2d(self.args.num_channels)
        self.bn4 = nn.BatchNorm2d(self.args.num_channels)

        self.bn5 = nn.BatchNorm2d(self.args.num_channels)
        self.bn6 = nn.BatchNorm2d(self.args.num_channels)
        self.bn7 = nn.BatchNorm2d(self.args.num_channels)

        # print(self.args.num_channels * (self.boards[0] - 4) * (self.boards[1] - 4), self.args.num_channels * (self.walls[0] - 2) * (self.walls[1] - 2))
        size = self.args.num_channels * (self.boards[0] - 4) * (self.boards[1] - 4) + self.args.num_channels * (
                    self.walls[0] - 2) * (self.walls[1] - 2) + self.values
        # print(size)
        self.fc1 = nn.Linear(size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, board, walls, values):
        # s: batch_size x board_x x board_y
        s = board.view(-1, self.boards[2], self.boards[0], self.boards[1])  # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels * (self.boards[0] - 4) * (self.boards[1] - 4))

        s2 = walls.view(-1, self.walls[2], self.walls[0], self.walls[1])  # batch_size x 1 x board_x x board_y
        s2 = F.relu(self.bn5(self.conv5(s2)))  # batch_size x num_channels x board_x x board_y
        s2 = F.relu(self.bn6(self.conv6(s2)))  # batch_size x num_channels x board_x x board_y
        s2 = F.relu(self.bn7(self.conv7(s2)))  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s2 = s2.view(-1, self.args.num_channels * (self.walls[0] - 2) * (self.walls[1] - 2))

        s3 = values.view(-1, self.values)  # batch_size x 1 x board_x x board_y

        # print(s.shape)
        # print(s2.shape)
        # print(s3.shape)
        s = torch.cat((s, s2, s3), 1)
        # print(s.shape)
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        # return F.log_softmax(pi, dim=1), F.tanh(v)
        return F.log_softmax(pi, dim=1), torch.tanh(v)  # new
