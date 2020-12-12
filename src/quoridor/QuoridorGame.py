import sys
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.join(os.path.dirname(__file__), 'pathfind/build'))
import QuoridorUtils
from src.alphazero_general.Game import Game
from .QuoridorLogic import QuoridorBoard


def board_pretty(board, invert_yaxis=False, path=[]):
    """
    Simulator.visualize(path) # plot a path
    Simulator.visualize(path_full, path_short) # plot two paths

    path is a list for the trajectory. [x[0], y[0], x[1], y[1], ...]
    """

    fig_map, ax_map = plt.subplots(1, 1)

    # plot retangle obstacles
    for idx, x in np.ndenumerate(board[0, :, :]):
        idx = (idx[1], idx[0])
        if idx[0] % 2 == 1 or idx[1] % 2 == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='lightgray')
            # Add the patch to the Axes
            ax_map.add_patch(rect)
        if x == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='darkred')
            # Add the patch to the Axes
            ax_map.add_patch(rect)

    for idx, x in np.ndenumerate(board[1, :, :]):
        idx = (idx[1], idx[0])
        if x == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='darkblue')
            # Add the patch to the Axes
            ax_map.add_patch(rect)
            # Add the patch to the Axes
            ax_map.add_patch(rect)

    for idx, x in np.ndenumerate(board[2, :, :]):
        idx = (idx[1], idx[0])
        if x == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='r')
            # Add the patch to the Axes
            ax_map.add_patch(rect)

    for idx, x in np.ndenumerate(board[3, :, :]):
        idx = (idx[1], idx[0])
        if x == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='b')
            # Add the patch to the Axes
            ax_map.add_patch(rect)

    points = list(zip(path, path[1:]))[::2]
    for i, p in enumerate(points):
        if i != 0 and i != len(points) - 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(p, 1, 1,
                                     linewidth=1, facecolor='g', alpha=0.5)
            # Add the patch to the Axes
            ax_map.add_patch(rect)

    ax_map.set_aspect('equal')
    ax_map.set_yticks(np.arange(0, 17, 2))
    ax_map.set_xticks(np.arange(0, 17, 2))
    ax_map.set_xlim([0, 17])
    ax_map.set_ylim([0, 17])
    if invert_yaxis:
        ax_map.invert_yaxis()
    plt.show()


class QuoridorGame(Game):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.board_len = 2 * self.n - 1
        self.action_size = 12 + 2 * (self.n - 1) ** 2

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return QuoridorBoard(self.n).setInitBoard()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return self.board_len, self.board_len

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # 4 pawn moves, 8 jumps, 64 vertical walls, 64 horizontal walls
        return self.action_size

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        quoridor_board = QuoridorBoard(self.n)
        quoridor_board.setBoard(board)
        next = quoridor_board.executeAction(player, action)
        return (self.flipBoard(next), -player)

    def getValidActions(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        return QuoridorUtils.GetValidMoves(board, player)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        if np.any(board[2, self.board_len - 1, :]):
            return 1
        elif np.any(board[3, 0, :]):
            return -1
        return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        if player == 1:
            return board
        else:
            return self.flipBoard(board)

    def flipBoard(self, board):
        b = np.zeros(board.shape, dtype=int)
        b[2, :, :] = np.array(board[3, :, :], copy=True)
        b[3, :, :] = np.array(board[2, :, :], copy=True)
        b[0, :, :] = np.array(board[1, :, :], copy=True)
        b[1, :, :] = np.array(board[0, :, :], copy=True)

        for i in range(4):
            b[i, :, :] = np.flip(b[i, :, :], (0, 1))
        return b

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """

        return hash(board.tostring())
