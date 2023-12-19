import os
import pathlib
import numpy as np
import random
from time import sleep
import parser
from configparser import ConfigParser
import snake
from itertools import count
import torch
import AI
import matplotlib.pyplot as plt


table=np.zeros([3,parser.SCREEN_SIZE[0],parser.SCREEN_SIZE[1]], dtype='float32')
table[2,random.randint(1,parser.SCREEN_SIZE[0]-2), random.randint(1,parser.SCREEN_SIZE[1]-2)]=3
table[0,0,:]=1
table[0,:,0]=1
table[0,parser.SCREEN_SIZE[0]-1,:]=1
table[0,:,parser.SCREEN_SIZE[1]-1]=1


def init_table():
    global table
    
    table=np.zeros([3,parser.SCREEN_SIZE[0],parser.SCREEN_SIZE[1]], dtype='float32')
    table[0,0,:]=1
    table[0,:,0]=1
    table[0,parser.SCREEN_SIZE[0]-1,:]=1
    table[0,:,parser.SCREEN_SIZE[1]-1]=1
    table[2,random.randint(1,parser.SCREEN_SIZE[0]-2), random.randint(1,parser.SCREEN_SIZE[1]-2)]=3

def print_screen(table:np.ndarray):
    for i in table:
        print(i)

durations=[]
random_moves_per_episode=[]

plt.ion()
def plot():
    plt.figure(1)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations)
    plt.plot(random_moves_per_episode)
    
    plt.pause(0.001)    


p=np.array([1,2,3,4])
a=torch.tensor(p)
print(a)
p[1]=0
b=torch.tensor(p)
print(b,"\n",a)

plot()
epochs=600
for i in range(epochs):
    init_table()
    random_moves=0
    player=snake.Snake(parser.SCREEN_SIZE, table)
    state=torch.tensor(table, device='cuda', dtype=torch.float32).unsqueeze(0)
    for t in count() :
        #state=torch.tensor(table, device='cuda', dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        action, was_random=AI.select_action(state, 0.05, 0.90, 1000)
        player.change_dir(action)

        if was_random:
            random_moves+=1
        result=player.step(table)
        if result==2:
            table[2,random.randint(1,parser.SCREEN_SIZE[0]-2), random.randint(1,parser.SCREEN_SIZE[1]-2)]=3
        new_state=torch.tensor(table, device='cuda', dtype=torch.float32).unsqueeze(0)
        
        #print("\nCurrent: \n",state,"\nNext: \n",new_state)
        # assert np.any((state!=new_state).cpu().numpy()), "State not changing"

        if result==0: #Collide
            AI.update_memory(state, 
                             torch.tensor([player.dir], device='cuda').view(1,1), 
                             None, 
                             torch.tensor([-100], device='cuda'))
        elif result==1: #Nothing
            AI.update_memory(state, 
                             torch.tensor([player.dir], device='cuda').view(1,1), 
                             new_state, 
                             torch.tensor([1], device='cuda'))
        elif result==2: #Eat
            AI.update_memory(state, 
                             torch.tensor([player.dir], device='cuda').view(1,1), 
                             new_state, 
                             torch.tensor([+100], device='cuda'))
        # os.system('cls')
        # print_screen(table)
        # sleep(.2)
        AI.train_step(100,1024, 0.99, 0.005)  
        state=new_state      
        if result==0:
            durations+= [t+1]
            random_moves_per_episode+=[random_moves]
            if i%100==0:
                plot()  
            break

          
while(True):
    init_table()      
    player=snake.Snake(parser.SCREEN_SIZE, table)      
    while(True):
        player.change_dir(AI.select_action(state, 0.05, 0.95, 1000)[0])
        result=player.step(table)
        # os.system('cls')
        plt.figure(2)
        plt.imshow(table.transpose(1,2,0)/3)
        plt.pause(0.001)
        sleep(.2)
        if result==2:
            table[2,random.randint(1,parser.SCREEN_SIZE[0]-2), random.randint(1,parser.SCREEN_SIZE[1]-2)]=3
        if result==0:
            break
    


