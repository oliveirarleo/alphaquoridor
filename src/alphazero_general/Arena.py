import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena:
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False, name=None):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                self.display(board, name=name+'_'+str(it))

            c = self.game.getCanonicalForm(board, curPlayer)
            action = players[curPlayer + 1](c)
            valids = self.game.getValidActions(c, 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.info(f'valids = {valids}')
                log.info(f'player = {curPlayer}')
                log.info(f'red_walls = {board.red_walls}')
                log.info(f'blue_walls = {board.blue_walls}')
                board.plot(save=False)
                c.plot(save=False)

                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert self.display
            # print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board, name+'_over')
        # print(it, curPlayer * self.game.getGameEnded(board, curPlayer))
        return curPlayer * self.game.getGameEnded(board, curPlayer)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        progress_bar = tqdm(range(num), desc="Arena.playGames (1)")
        for i in progress_bar:
            gameResult = self.playGame(verbose=verbose, name='game_'+str(i))
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            progress_bar.set_description(f'Arena.playGames (1): {oneWon} / {twoWon} : {draws}')

        self.player1, self.player2 = self.player2, self.player1

        progress_bar = tqdm(range(num), desc="Arena.playGames (2)")
        for i in progress_bar:
            gameResult = self.playGame(verbose=verbose, name='game_'+str(i+num))
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

            progress_bar.set_description(f'Arena.playGames (2): {oneWon} / {twoWon} : {draws}')

        return oneWon, twoWon, draws
