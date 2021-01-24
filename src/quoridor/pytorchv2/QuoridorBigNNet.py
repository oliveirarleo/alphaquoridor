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
        self.conv3 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(self.walls[2], self.args.num_channels, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(self.args.num_channels, self.args.num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(self.args.num_channels)
        self.bn2 = nn.BatchNorm2d(self.args.num_channels)
        self.bn3 = nn.BatchNorm2d(self.args.num_channels)
        self.bn4 = nn.BatchNorm2d(self.args.num_channels)

        self.bn5 = nn.BatchNorm2d(self.args.num_channels)
        self.bn6 = nn.BatchNorm2d(self.args.num_channels)
        self.bn7 = nn.BatchNorm2d(self.args.num_channels)
        self.bn8 = nn.BatchNorm2d(self.args.num_channels)

        size = self.args.num_channels * self.boards[0] * self.boards[1] + self.args.num_channels * self.walls[0] * self.walls[1] + self.values
        self.fc1 = nn.Linear(size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        # Policy head
        self.fc3 = nn.Linear(512, 256)
        self.fc_bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, self.action_size)

        # Value head
        self.fc5 = nn.Linear(512, 256)
        self.fc_bn5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, board, walls, values):
        s = board.view(-1, self.boards[2], self.boards[0], self.boards[1])
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(-1, self.args.num_channels * self.boards[0] * self.boards[1])

        s2 = walls.view(-1, self.walls[2], self.walls[0], self.walls[1])
        s2 = F.relu(self.bn5(self.conv5(s2)))
        s2 = F.relu(self.bn6(self.conv6(s2)))
        s2 = F.relu(self.bn7(self.conv7(s2)))
        s2 = s2.view(-1, self.args.num_channels * self.walls[0] * self.walls[1])

        s3 = values.view(-1, self.values)

        s = torch.cat((s, s2, s3), 1)
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        s_pi = F.dropout(F.relu(self.fc_bn3(self.fc3(s))), p=self.args.dropout, training=self.training)
        pi = self.fc4(s_pi)

        s_v = F.dropout(F.relu(self.fc_bn5(self.fc5(s))), p=self.args.dropout, training=self.training)
        v = self.fc6(s_v)

        return F.log_softmax(pi, dim=1), torch.tanh(v)
