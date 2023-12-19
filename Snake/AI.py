import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import parser
from collections import deque, namedtuple

class AI_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3, 32, 4,2,1)
        self.flat=nn.Flatten()
        self.lin0=nn.Linear((parser.SCREEN_SIZE[0]*parser.SCREEN_SIZE[1])*8, 512)
        self.lin1=nn.Linear(512,4)

    def forward(self, in_data) ->torch.Tensor:
        x=self.conv1(in_data)
        x=F.relu(x)
        #x=F.batch_norm(x)
        x=self.flat(x)
        x=F.relu(self.lin0(x))
        return self.lin1(x)
    
Transition=namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, max_capacity):
        self.memory=deque([], max_capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
memory=ReplayMemory(10000)

def update_memory(*args):
    global memory
    memory.push(*args)

ai_model=AI_net().cuda()
target_ai_model=AI_net().cuda()
target_ai_model.load_state_dict(ai_model.state_dict())
optimizer=optim.AdamW(ai_model.parameters(),float(parser.parser.get('Default', 'LEARN_RATE')), amsgrad=True)
criterion=nn.MSELoss()
steps_done=0


def select_action(state, eps_end, eps_start, eps_decay):
    global steps_done

    eps_threshold=eps_end + math.exp(-1*(steps_done)/eps_decay)*(eps_start-eps_end)
    steps_done+=1

    if(random.random()<eps_threshold):
        return random.randint(0,3), True
    else:
        return ai_model(state).max(1).indices.view(1,1).item(), False

def train_step(epochs:int, batch_size:int, gamma, blend_factor):
    global memory

    if(batch_size>len(memory)):
        return
    transitions=memory.sample(batch_size)

    batch=Transition(*zip(*transitions))
    
    state_batch=torch.cat(batch.state)
    action_batch=torch.cat(batch.action)
    reward_batch=torch.cat(batch.reward)

    state_action_values=ai_model(state_batch).gather(1, action_batch)
    
    non_final_mask=torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device='cuda', dtype=torch.bool)
    non_final_next_states=torch.cat([s for s in batch.next_state if s is not None])

    next_state_values=torch.zeros(tuple([batch_size]), device='cuda')
    with torch.no_grad():
        next_state_values[non_final_mask]=ai_model(non_final_next_states).max(1).values
    

    expected_state_action_values=next_state_values*gamma + reward_batch
    
    
    optimizer.zero_grad()
    loss=criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    loss.backward()
    torch.nn.utils.clip_grad_value_(ai_model.parameters(), 100)
    optimizer.step()

    target_ai_model_state_dict=target_ai_model.state_dict()
    ai_model_state_dict=ai_model.state_dict()
    for key in ai_model_state_dict:
        target_ai_model_state_dict[key]=blend_factor*ai_model_state_dict[key]+(1-blend_factor)*target_ai_model_state_dict[key]
    target_ai_model.load_state_dict(target_ai_model_state_dict)

        

    

        