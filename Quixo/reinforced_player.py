from copy import deepcopy
import math
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from game import Game, Move
from main import Player
from collections import deque, namedtuple
from random_player import RandomPlayer

CMove=namedtuple("CMove", ["position","move"])
Transition=namedtuple("Transition", ["state", "action", "reward", "next_state","is_final"])
BATCH_SIZE=128
ALPHA=.9
LEARN_RATE=.0001

from_poss=[]
moves : list[CMove]=[]



def is_move_acceptable(from_pos_inv, slide):
    '''Slide the other pieces'''
    from_pos=(from_pos_inv[1],from_pos_inv[0])
    # define the corners
    SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
    # if the piece position is not in a corner
    if from_pos not in SIDES:
        # if it is at the TOP, it can be moved down, left or right
        acceptable_top: bool = from_pos[0] == 0 and (
            slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
        )
        # if it is at the BOTTOM, it can be moved up, left or right
        acceptable_bottom: bool = from_pos[0] == 4 and (
            slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
        )
        # if it is on the LEFT, it can be moved up, down or right
        acceptable_left: bool = from_pos[1] == 0 and (
            slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
        )
        # if it is on the RIGHT, it can be moved up, down or left
        acceptable_right: bool = from_pos[1] == 4 and (
            slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
        )
    # if the piece position is in a corner
    else:
        # if it is in the upper left corner, it can be moved to the right and down
        acceptable_top: bool = from_pos == (0, 0) and (
            slide == Move.BOTTOM or slide == Move.RIGHT)
        # if it is in the lower left corner, it can be moved to the right and up
        acceptable_left: bool = from_pos == (4, 0) and (
            slide == Move.TOP or slide == Move.RIGHT)
        # if it is in the upper right corner, it can be moved to the left and down
        acceptable_right: bool = from_pos == (0, 4) and (
            slide == Move.BOTTOM or slide == Move.LEFT)
        # if it is in the lower right corner, it can be moved to the left and up
        acceptable_bottom: bool = from_pos == (4, 4) and (
            slide == Move.TOP or slide == Move.LEFT)
    # check if the move is acceptable
    acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
    return acceptable

def is_tile_acceptable(from_pos, board, player_id):
    acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (board[from_pos] < 0 or board[from_pos] == player_id)
    return acceptable

def gen_pos(game:Game):
    global from_poss
    board=game.get_board()

    for (i,j),_ in np.ndenumerate(board):
        if(i==0 or i==4 or j==0 or j==4):
            from_poss.append((i,j))
    
def gen_moves():
    global from_poss, moves
    #board=game.get_board()
    for pos in from_poss:
        for m in Move:
            if(is_move_acceptable(pos,m)):
                moves.append(CMove(pos,m))
    
    return moves

class ReplayMemory():
    def __init__(self, max_capacity):
        self.mem=deque([], max_capacity)
    def __len__(self):
        return len(self.mem)
    def push(self, elem):
        self.mem.append(elem)
    def sample(self, batch_size):
        samples=random.sample(self.mem, batch_size)
        self.mem.clear()
        return samples 

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq=nn.Sequential(nn.Linear(75,128),
                               nn.ReLU(),
                               nn.Linear(128,64),
                               nn.ReLU(),
                               nn.Linear(64,44))
        
    def forward(self, data) -> torch.Tensor:
        return self.seq(data)

def one_hot(game:Game):
    return F.one_hot(torch.tensor(game.get_board()+1).long(),3).float()
class ReinforcedPlayer(Player):
    def __init__(self, player_id, discount=.9, eps_decay=500):
        self.player_id=player_id

        self.discount=discount
        self.eps_decay=eps_decay
        self.epsilon=eps_decay
        self.memory=ReplayMemory(5000)
        self.agent=QNetwork()
        self.target_agent=QNetwork()
        
        self.target_agent.load_state_dict(self.agent.state_dict())

        self.training=True
        gen_pos(Game())
        gen_moves()

    def play_game(self,other:Player):
        game=Game()
        player=[self, self]
        player[self.player_id]=self
        player[1-self.player_id]=other

        winner=-1
        state=None
        while winner<0:
            
            game.current_player_idx = 1-game.current_player_idx

            if game.current_player_idx==self.player_id:
                state=one_hot(game).flatten()

            ok=False

            while not ok:
                move=player[game.current_player_idx].make_move(game)
                ok=game._Game__move(move[0], move[1], game.current_player_idx)

            winner=game.check_winner()

            
            if state!=None and (game.current_player_idx!=self.player_id or winner>=0):
                if winner==self.player_id:
                    reward=1
                elif winner==1-self.player_id:
                    reward=-1
                else:
                    reward=0
                # print(f" {game.current_player_idx!=self.player_id} {winner}")
                self.memory.push(Transition(state,self.last_action, reward, one_hot(game).flatten(), winner>=0))

        return

    def train(self, episodes=500):
        optimizer=optim.Adam(self.agent.parameters(),0.0001)
        criterion=nn.MSELoss()

        for i in range(episodes):
            print(f"Episode: {i}")
            self.play_game(RandomPlayer())

            #Bellman: Q(s,a) = r + gamma*max_a Q(s', a)

            if len(self.memory)<BATCH_SIZE:
                continue

            batch=self.memory.sample(BATCH_SIZE)
            batch=Transition(*zip(*batch))

            batch_state=torch.cat(batch.state).view(BATCH_SIZE,-1) 
            batch_action=torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1)
            batch_reward=torch.tensor(batch.reward, dtype=torch.float).unsqueeze(1)
            batch_next_state=torch.cat(batch.next_state).view(BATCH_SIZE,-1)
            batch_is_final=torch.tensor(batch.is_final, dtype=torch.bool).flatten()

            Q_sa=batch_reward.clone()
            with torch.no_grad():
                print(~batch_is_final)
                Q_sa[~(batch_is_final.numpy())]+=(self.target_agent(batch_next_state).max(1).values.unsqueeze(1))[~(batch_is_final.numpy())]
            pred=self.agent(batch_state).gather(1, batch_action)

            loss=criterion(pred, Q_sa)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            agent_dict=self.agent.state_dict()
            target_dict=self.target_agent.state_dict()

            for k in agent_dict.keys():
                target_dict[k]=ALPHA*target_dict[k]+(1-ALPHA)*agent_dict[k]
            self.target_agent.load_state_dict(target_dict)

        print("Trained")            


    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        g=deepcopy(game)
        encoded_game=one_hot(g)

        eps_threshold=math.log(1+10*(self.epsilon/self.eps_decay))
        if(random.random()<eps_threshold):
            action_indices=torch.randperm(44)
        else:
            action=self.agent(encoded_game.float().flatten())
            _,action_indices=torch.sort(action,descending=True)

        self.epsilon-=1
        if self.epsilon<0:
            self.epsilon=0

        for i in action_indices:
            m=moves[i]
            ok=g._Game__move(m.position, m.move, self.player_id)
            if ok:
                self.last_action=i
                return m
        assert ok, "No more moves"
        return 