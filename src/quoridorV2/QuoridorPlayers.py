import numpy as np


class RandomPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        actions = self.game.getValidActions(board, 1)
        print(actions)
        return np.random.choice(len(actions), p=np.array(actions)/sum(actions))


class HumanQuoridorPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidActions(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i / self.game.n), int(i % self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x, y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyQuoridorPlayer:
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidActions(board, 1)
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]
