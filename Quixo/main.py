import random
from game import Game, Move, Player
from random_player import RandomPlayer
import vinz_player
import A2c
import reinforced_player


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move


if __name__ == '__main__':


    g = Game()
    vinz_player.gen_pos(g)
    vinz_player.gen_moves()
    g.print()
    #player1 = RandomPlayer()
    #player1 = vinz_player.VinzPlayer(0)
    #player1=vinz_player.MontecarloPlayer(5,100)

    #player1=A2c.BigBrainPlayer(0,.8)
    player1=reinforced_player.ReinforcedPlayer(0)
    player1.train(5000)
    # player1=vinz_player.AlphaPlayer(0,1)
    player2 = RandomPlayer()
    player1_wins=0
    for i in range(0,1000):
        winner = g.play(player1, player2)
        if winner==0:
            player1_wins+=1
        print(f"Game: {i}, wins: {player1_wins}")
        g=Game()
    g.print()
    A2c.onehot_encode(g)

    print(f"Wins: {player1_wins}/1000")
    print(vinz_player.loss_v1(g, 0))
    print(f"Winner: Player {winner}")


#10,6 50%
#