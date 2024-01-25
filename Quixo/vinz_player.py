from collections import namedtuple
from copy import deepcopy
import random
from game import Game, Move
import numpy as np
from game import Player
from math import log, sqrt
from random_player import RandomPlayer

CMove=namedtuple("CMove", ["position","move"])
from_poss=[]
moves : list[CMove]=[]

def is_move_acceptable(from_pos, slide):
    '''Slide the other pieces'''
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
            if(is_move_acceptable((pos[1],pos[0]),m)):
                moves.append(CMove(pos,m))
    
    return moves

def loss_v1(game:Game, symbol=0)-> int: #max symbols on a line/diagonal
    b=game.get_board()

    max=0
    for i in range(0,5):
        count=np.count_nonzero(b[i,:]==symbol)
        if (1-symbol) not in b[i,:] and count>max:
            max=count

    for i in range(0,5):
        count=np.count_nonzero(b[:,i]==symbol)
        if (1-symbol) not in b[:,i] and count>max:
            max=count
    main_diag=np.array([ [i,j] for i in range(0,5) for j in range(0,5) if i==j])

    second_diag=np.array([[i,j] for i in range(0,5) for j in range(0,5) if 4-i==j])

    count=np.count_nonzero(b[main_diag[:,0], main_diag[:,1]]==symbol)

    if (1-symbol) not in b[main_diag[:,0],main_diag[:,1]] and count>max:
        max=count

    count=np.count_nonzero(b[second_diag[:,0], second_diag[:,1]]==symbol)

    if (1-symbol)not in b[second_diag[:,0],second_diag[:,1]] and count>max:
        max=count
    
    return max

class AlphaPlayer(Player):
    def __init__(self, player_id, depth=2):
        self.player_id=player_id
        self.depth=depth

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        print("Move")
        t=None
        outs=dict()

        for m in moves:
            temp_g=deepcopy(game)
            ok=temp_g._Game__move(m.position, m.move,self.player_id)
            if ok:
                temp_g.current_player_idx=1-temp_g.current_player_idx
                t=self.__alphabeta_node(temp_g, t, self.depth, 1-self.player_id)
                outs[m]=t
            
        return max(outs, key=outs.get)
    
    def __alphabeta_node(self, game:Game, comparison_val, n=4, turn=1) -> int:
            global moves
            if n<0:
                return loss_v1(game)
            
            outs=[]
            for m in moves:
                temp_g=deepcopy(game)

                assert turn==temp_g.get_current_player(),"Wrong turn"
                ok=temp_g._Game__move(m.position, m.move,turn)
                
                t=None
                if ok:
                    temp_g.current_player_idx=1-temp_g.current_player_idx
                    t=self.__alphabeta_node(temp_g, t, n-1, 1-turn)
                    if comparison_val is None:
                        outs.append(t)
                    elif turn!=self.player_id:
                        if t<comparison_val:
                            return t
                        else:
                            outs.append(t)
                    elif turn==self.player_id:
                        if t>comparison_val:
                            return t
                        else:
                            outs.append(t)
                    #print(outs)
                        

            if len(outs)==0:
                print(f"Hit bottom at {n}")
                return -1
            
            if turn==1-self.player_id:
                return min(outs)
            elif turn==self.player_id:
                return max(outs)
            



class MiniPlayer(Player): #minimax with alpha-beta (TODO)
    def __init__(self, player_id=0):
        self.player_id=player_id
        pass
    
    def make_move(self, game: Game, alphabeta=False) -> tuple[tuple[int, int], Move]:
        global moves
        
        values=np.zeros(len(moves))
        for i, m in enumerate(moves):
            temp_g=deepcopy(game)
            ok=temp_g._Game__move(m.position, m.move,self.symbol)
            temp_g.current_player_idx=1-temp_g.current_player_idx
            if ok:
                values[i]=self.__minimax_node(temp_g, 1, 1-self.symbol)
                 
        
        idx=np.argmax(values)

        #print("returned")
        return moves[idx]
    


    def __minimax_node(self, game:Game,n=4, turn=1) -> int:
        global moves
        if n<0:
            return loss_v1(game)
        
        outs=[]
        for m in moves:
            temp_g=deepcopy(game)

            assert turn==temp_g.get_current_player(),"Wrong turn"
            ok=temp_g._Game__move(m.position, m.move,turn)
            
            temp_g.current_player_idx=1-temp_g.current_player_idx
            if ok:
                t=self.__minimax_node(temp_g, n-1, 1-turn)
                outs.append(t)
                #print(outs)

        if len(outs)==0:
            print(f"Hit bottom at {n}")
            return -1
        
        if turn==1-self.symbol:
            return min(outs)
        elif turn==self.symbol:
            return max(outs)
        
        

class MontecarloPlayer(Player): #https://towardsdatascience.com/monte-carlo-tree-search-an-introduction-503d8c04e168 this helped me understand thus the code may be very similar
    
    def __init__(self, n_plays, iterations):
        self.iterations=iterations
        self.n_plays=n_plays

    def make_move(self,game:Game) -> tuple[tuple[int,int], Move]:
        
        self.root_node=MontecarloNode(deepcopy(game),None, None)
        wins,simulations=self.root_node.rollout(self.n_plays)
        self.root_node.backpropagate(wins, simulations)
      
        for _ in range(0,self.iterations):
            l,_=self.root_node.ret_max_child()
            c=l.expand()
            if c is None:
                break
            wins,simulations=c.rollout(self.n_plays)
            c.backpropagate(wins,simulations)
        #print("returned")
        move=self.root_node.get_move()
        assert move is not None and is_move_acceptable(*move) and deepcopy(game)._Game__move(*move, game.get_current_player()), f"{move}, \n {game.get_board()}"

        return move

class MontecarloNode():
    
    def __init__(self, game_state:Game, move:CMove=None, parent=None, turn=0, expandable=True):
        self.move=move
        self.turn=turn
        self.parent=parent
        self.children_nodes:list=[]
        self.w=0 #number of wins
        self.n=0 #number of explorations
        self.game_state=game_state
        self.expandable=expandable
        self.val=0

    def get_move(self):
        if self.move is not None:
            return self.move
        else:
            min=self.children_nodes[0]
            for i in range(1,len(self.children_nodes)):
                if self.children_nodes[i].get_value()<min.get_value():
                    min=self.children_nodes[i]

            return min.move
        
    def get_move_backtrack(self, root_node):
        if self.parent==root_node:
            return self.move
        else:
            return self.parent.get_move_backtrack(root_node)
    def get_value(self):
        if self.n==0:
            return 0
        return self.w/self.n
    
    def calc_value_UCT(self, c=2, with_heuristic=False):
        if self.n==0:
            print("Shouldn't happen")
            return 
        N=self.n
        if self.parent!=None:
            N=self.parent.n
        
        val=self.w/self.n + sqrt(c*log(N)/self.n)
        if with_heuristic:
            val+= loss_v1(self.game_state)/5
        return val
    

    def ret_max_child(self,N=None, rec=0):
        if N is None:
            N=self.n
        
        max_val=self.val
        o=self

        if(len(self.children_nodes)>100):
            print(f"At rec {rec} children: {len(self.children_nodes)}")
        # print(f"{self.w}/{self.n}, N={N}")

        for i in self.children_nodes: 
            assert i!=self, "Recursion on itself"

            if not i.expandable:
                continue

            temp,temp_val=i.ret_max_child(self.n, rec+1)
            if(temp_val>max_val):
                max_val=temp_val
                o=temp

        
        return o, max_val
            
    def expand(self)-> tuple[object, bool]: 
        global moves

        if not self.expandable:
            return None
        
        moves_ls=deepcopy(moves)

        random.shuffle(moves_ls)
        moves_loss=dict()
        for rand_move in moves_ls:
            game_cpy=deepcopy(self.game_state)
            assert self.turn==game_cpy.get_current_player(), f"Wrong turn {self.turn}, {self.depth}, {game_cpy.get_current_player()}"
            
            ok=game_cpy._Game__move(rand_move.position, rand_move.move, self.turn)
            
            if ok: 
                moves_loss[rand_move]=loss_v1(game_cpy)
        
        # assert ok, "No moves available"
        assert len(moves_loss)!=0, "No moves available"
        
        game_cpy.current_player_idx+=1
        game_cpy.current_player_idx%=2
        expandable=game_cpy.check_winner()==-1

        new_child=MontecarloNode(game_cpy, max(moves_loss, key=moves_loss.get), self, 1-self.turn, expandable)
        assert new_child!=self, "Child is itself"
        
        self.children_nodes.append(new_child)
        return new_child

    def rollout(self, n_plays=1):
        simulations=n_plays
        wins=0
        for i in range(n_plays):
            if(deepcopy(self.game_state).play(RandomPlayer(),RandomPlayer())==self.turn):
                wins+=1
        return wins, simulations


    def backpropagate(self,wins=0, simulations=1):
        self.n+=simulations
        self.w+=wins

        self.val=self.calc_value_UCT(6, False)
        if(self.parent!=None):

            assert type(self.parent)==MontecarloNode, f"Parent is {type(self.parent)}"
            self.parent.backpropagate(simulations-wins, simulations)



    


    
    
