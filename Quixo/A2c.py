
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from copy import deepcopy
import random
from game import Game, Move
import numpy as np
from game import Player
from math import log, sqrt
from random_player import RandomPlayer
from matplotlib import pyplot as plt

CMove=namedtuple("CMove", ["position","move"])
Transition=namedtuple("Transition", ["state", "action", "reward", "next_state","is_final"])
BATCH_SIZE=1024
ALPHA=.0001
LEARN_RATE_ACTOR=.0001
LEARN_RATE_CRITIC=.0001

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

class ActorNet(nn.Module):
    def __init__(self, inputs=1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(25*inputs, 256),
                               nn.LeakyReLU(),
                               nn.Linear(256, 64),
                               nn.LeakyReLU(),
                               nn.Linear(64,44),

                               nn.Softmax()
                               )
        
    def forward(self, data):
        return self.net(data)
    
class CriticNet(nn.Module):
    def __init__(self, inputs=1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(25*inputs, 256),
                               nn.LeakyReLU(),
                               nn.Linear(256, 64),
                               nn.LeakyReLU(),
                               nn.Linear(64,1)
                               )
        
    def forward(self, data):
        return self.net(data)
    
def onehot_encode(game:Game): #   -1< x <=1
    board=deepcopy(game.get_board())
    board+=1


    flat=board.flatten()
    ohenc=np.zeros([3,flat.shape[0]])
    ohenc[flat,np.arange(25)]=1
    ohenc=ohenc.reshape([3,5,5])
    return ohenc
    
def onehot_encode_action(act_index):
    # action=np.zeros(44)
    action=torch.zeros(44)
    action[act_index]=1
    return action

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

class BigBrainPlayer(Player):
    def __init__(self, player_id, discount=0.7):
        gen_pos(Game())
        gen_moves()
        self.player_id=player_id
        self.actor=ActorNet(3)
        self.critic=CriticNet(3)
        self.memory=ReplayMemory(5000)
        self.last_move=None
        self.discount=discount
        self.move_id=None

        self.training=True

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        global moves

        state=onehot_encode(game)
        self.actor_probs:torch.Tensor=self.actor(torch.tensor(state, dtype=torch.float32).flatten())

        softmax=self.actor_probs.detach()
        dist=torch.distributions.Categorical(softmax)
        choice=dist.sample()
        assert game.current_player_idx==self.player_id

        it=0
        while True:
            move=moves[choice]
            if game.get_board()[(move.position[1],move.position[0])]!=1-self.player_id:
                self.move_id=choice
                break
            else:
                choice=dist.sample()
            it+=1

        self.last_move=self.move_id
        return move
    
    def play_turn_as_player1(self, game:Game, player2=RandomPlayer()) -> Transition:
        global moves

        state=torch.tensor(onehot_encode(game),dtype=torch.float)

        game.current_player_idx=1-game.current_player_idx
        it=0
        ok=False
        
        while not ok:
            assert(game.current_player_idx==self.player_id), "Wrong Turn"

            from_pos, move = self.make_move(game)
            ok=game._Game__move(from_pos, move, self.player_id)

            #SANITY CHECKS
            assert ok
            it+=1
            if(it>20000):
                print(f"Too much time in play as player1 {game.get_last_error()} while {game.get_board()[from_pos]}")

        if game.check_winner()==-1:
            game.current_player_idx=1-game.current_player_idx

            ok=False
            while not ok:
                from_pos, move=player2.make_move(game)
                ok=game._Game__move(from_pos, move, 1)
            
        winner=game.check_winner()
        reward=0

        if winner==-1:
            reward=0
        elif winner==self.player_id:
            reward=1
        else:
            reward=-1

        next_state=torch.tensor(onehot_encode(game),dtype=torch.float)
        t=Transition(state.flatten(), self.last_move, reward, next_state.flatten(), winner!=-1)

        return t

    def play_turn_as_player2(self, game:Game, player1=RandomPlayer()) -> Transition:
        ''' Simulate one step of the game and save the transition (state,action,reward,next state and check if it was a final move) '''
        global moves

        state=torch.tensor(onehot_encode(game),dtype=torch.float)
        
        game.current_player_idx=1-game.current_player_idx

        ok=False
        while not ok:
            from_pos, move=player1.make_move(game)
            ok=game._Game__move(from_pos, move, 1)

        game.current_player_idx=1-game.current_player_idx
        it=0
        ok=False
        
        while not ok:
            assert(game.current_player_idx==self.player_id), "Wrong Turn"

            from_pos, move = self.make_move(game)
            ok=game._Game__move(from_pos, move, self.player_id)
            it+=1
            if(it>20000):
                print(f"Too much time in play as player1 {game.get_last_error()} while {game.get_board()[from_pos]}")

        
        winner=game.check_winner()
        reward=0
        if winner==-1:
            reward=0

        elif winner==self.player_id:
            reward=1
        else:
            reward=-1


        next_state=torch.tensor(onehot_encode(game),dtype=torch.float)
        t=Transition(state.flatten(), self.last_move, reward, next_state.flatten(), winner!=-1)

        return t

    def train_batch(self, batch, critic_criterion, critic_optimizer, actor_criterion, actor_optimizer):


        critic_optimizer.zero_grad()
        actor_optimizer.zero_grad()

        batch_data=Transition(*zip(*batch))
        #print(batch_data)
        batch_reward=torch.tensor(batch_data.reward, dtype=torch.float).unsqueeze(1)
        batch_state=torch.cat(batch_data.state).view(len(batch),-1)
        batch_next_state=torch.cat(batch_data.next_state).view(len(batch),-1)
        batch_action=torch.tensor(batch_data.action, dtype=torch.float).unsqueeze(1)
        batch_done=torch.tensor(batch_data.is_final,dtype=bool).unsqueeze(1)

        batch_targets=torch.zeros(len(batch), dtype=torch.float).unsqueeze(1)

        val=0
        for i in reversed(range(len(batch_reward))):
            val=batch_reward[i]+self.discount*val
            batch_targets[i]=val

        # TD_target=self.discount*self.critic(batch_next_state).detach()
        # TD_target[batch_done]=0
        # TD_target+=batch_reward
        # TD_target=TD_target.detach()
        TD_target=batch_targets.detach()


        batch_action_encoded=F.one_hot(batch_action.long(), 44).squeeze(1)

        critic_pred=self.critic(batch_state)
        loss_c=critic_criterion(critic_pred, TD_target)
        loss_c.backward()
        critic_optimizer.step()

        advantage = TD_target-critic_pred.detach()
        
        actor_pred=self.actor(batch_state)
        gradient=batch_action_encoded-actor_pred.detach()
        actor_target=ALPHA*gradient*advantage + actor_pred.detach()

        loss_a=actor_criterion(actor_pred, actor_target)
        loss_a.backward()
        actor_optimizer.step()
        
    def train(self):
        global moves
        game=Game()
        #print(f"First turn: {game.get_current_player()}")
        critic_criterion=nn.MSELoss() #watch out for the reduction
        critic_optimizer=optim.Adam(self.critic.parameters(), LEARN_RATE_CRITIC)

        actor_criterion=nn.CrossEntropyLoss() #watch out for the reduction
        actor_optimizer=optim.Adam(self.actor.parameters(), LEARN_RATE_ACTOR)

        steps=0
        td_cumulative=0
        agent_wins=0
        
        self.critic.train()
        self.actor.train()


        loss_values=[]
        closs_values=[]
        for i in range(1000):
            transitions=[]
            while True:
                if(self.player_id==0):
                    t=self.play_turn_as_player1(game)
                else:
                    t=self.play_turn_as_player2(game)
                # self.memory.push(t)
                transitions.append(t)
                if t.is_final:
                    if (i+1)%50==0:
                        print(f"Agent wins: {agent_wins}/50 with an average of {td_cumulative/steps:.1} error at iteration {i}")
                        steps=0
                        td_cumulative=0
                        agent_wins=0
                        
                    if(game.check_winner()==self.player_id):
                        agent_wins+=1
                    game=Game()
                steps+=1
                if t.is_final:
                    break

            
            

            # TD_target=torch.tensor([t.reward], dtype=torch.float)   
            # if not t.is_final:
            #     TD_target+= self.discount*self.critic(t.next_state).detach()
            
            # critic_pred=self.critic(t.state)
            # critic_optimizer.zero_grad()
            # loss_critic=critic_criterion(critic_pred, TD_target)
            # closs_values.append(loss_critic.item())
            # loss_critic.backward()
            # critic_optimizer.step()


            # advantage = TD_target - critic_pred.detach()

            # td_cumulative+=advantage.item()

            # actor_pred=self.actor_probs
            # enc_action=onehot_encode_action(self.last_move)
            # gradient=enc_action - actor_pred.detach()
            # actor_target=ALPHA*gradient* advantage + actor_pred.detach()

            # actor_optimizer.zero_grad()
            # #loss_actor=(-actor_pred*advantage).mean()

            # loss_actor=actor_criterion(actor_pred,actor_target)
            # # print(f"{loss_actor.item():.2} and {advantage.item():.2}")
            # # loss_actor=loss_actor*advantage
            # # print(f"{loss_actor.item():.2}")
            # loss_actor.backward()
            # actor_optimizer.step()
            # loss_values.append(loss_actor.item())
                


            self.train_batch(transitions,critic_criterion, critic_optimizer, actor_criterion, actor_optimizer)
            transitions.clear()

        plt.plot(loss_values)
        plt.plot(closs_values)
        plt.show()
        plt.pause(0.5)
        self.actor.eval()
        self.critic.eval()            



        
            

        

    

        