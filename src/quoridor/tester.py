import numpy as np

from quoridor.QuoridorGame import QuoridorGame
from quoridor.QuoridorLogic import QuoridorBoard
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def tostring_board(board):
    walls = board[:, :, 0] + board[:, :, 1] + 2 * board[:, :, 2] + 3 * board[:, :, 3]
    board_string = 'board:' + str(walls.shape) + '\n'
    for line in walls:
        board_string += str(line) + '\n'
    return board_string


def board_pretty(board):
    """
    Simulator.visualize(path) # plot a path
    Simulator.visualize(path_full, path_short) # plot two paths

    path is a list for the trajectory. [x[0], y[0], x[1], y[1], ...]
    """

    fig_map, ax_map = plt.subplots(1, 1)

    walls = (board[:, :, 0] + board[:, :, 1])
    # plot retangle obstacles
    for idx, x in np.ndenumerate(walls):
        idx = (idx[1], idx[0])
        if x == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='k')
            # Add the patch to the Axes
            ax_map.add_patch(rect)
        elif idx[0] % 2 == 1 or idx[1] % 2 == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='lightgray')
            # Add the patch to the Axes
            ax_map.add_patch(rect)

    for idx, x in np.ndenumerate(board[:, :, 2]):
        idx = (idx[1], idx[0])
        if x == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='r')
            # Add the patch to the Axes
            ax_map.add_patch(rect)

    for idx, x in np.ndenumerate(board[:, :, 3]):
        idx = (idx[1], idx[0])
        if x == 1:
            # Create a Rectangle patch
            rect = patches.Rectangle(idx, 1, 1,
                                     linewidth=1, facecolor='b')
            # Add the patch to the Axes
            ax_map.add_patch(rect)

    # if len(arguments) == 2:
    #     ax_map.plot(arguments[1][0::2], arguments[1][1::2], label="short path")
    # ax_map.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_map.set_aspect('equal')
    ax_map.set_yticks(np.arange(0, 17, 2))
    ax_map.set_xticks(np.arange(0, 17, 2))
    ax_map.set_xlim([0, 17])
    ax_map.set_ylim([0, 17])
    plt.show()


def main():
    n = 5
    quoridor_game = QuoridorGame(n)
    init_board = quoridor_game.getInitBoard()

    print('Init Board', tostring_board(init_board))
    print('Board Size', quoridor_game.getBoardSize())
    print('Action Size', quoridor_game.getActionSize())
    print('Valid Moves', quoridor_game.getValidMoves(init_board, 1))

    # TEST GET GAME ENDED
    # b = init_board
    # for _ in range(n-1):
    #     b = quoridor_game.getNextState(b, 1, 0)
    #     b = quoridor_game.getNextState(b, -1, 1)
    #     print('Red UP')
    # b = quoridor_game.getNextState(b, 1, 2)
    # b = quoridor_game.getNextState(b, 1, 1)
    # b = quoridor_game.getNextState(b, -1, 0)
    # board_pretty(b)
    # print('Red won?', quoridor_game.getGameEnded(b, 1))

    # TEST PLACING WALLS
    pawn_moves = 12
    vwx = 2
    vwy = 0
    print('VW'+str(vwx)+str(vwy))
    b = quoridor_game.getNextState(init_board, 1, pawn_moves + vwx*(n-1) + vwy)

    vertical_wall_moves = pawn_moves + (n-1)*(n-1)
    hwx = 1
    hwy = 3
    print('HW' + str(hwx) + str(hwy))
    b = quoridor_game.getNextState(b, 1, vertical_wall_moves + hwx * 8 + hwy)

    # TEST MOVING
    print('Red N')
    b = quoridor_game.getNextState(b, 1, 0)
    print('Blue S')
    b = quoridor_game.getNextState(b, -1, 1)
    print('Blue E')
    b = quoridor_game.getNextState(b, -1, 2)
    board_pretty(b)

    # quoridor_board = QuoridorBoard(9)
    # quoridor_board.setInitBoard()
    # print('Red Position', quoridor_board.findPlayer(1))
    # print('Blue Position', quoridor_board.findPlayer(-1))
    # print('Red Walls', quoridor_board.countRedWalls())
    # print('Blue Walls', quoridor_board.countRedWalls())


if __name__ == "__main__":
    main()
