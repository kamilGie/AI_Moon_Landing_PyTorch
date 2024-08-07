# !pip install gymnasium
# !pip install "gymnasium[atari, accept-rom-license]"
# !apt-get install -y swig     
# !pip install gymnasium[box2d]

import os 
import random 
import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque,namedtuple

class Network (nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 =nn.Linear(state_size, 64)
        self.fc2 =nn.Linear(64, 64)
        self.fc3 =nn.Linear(64, action_size)

    def forward(self,state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)
    
import gymnasium as gym 
env = gym.make('LunarLander-v2')
state_shape = env.observation_space.shape
state_size =  env.observation_space.shape[0]
number_actions = env.action_space.n
print(state_shape)
print(state_size)
print(number_actions)

learning_rate = 5e-4
minBatch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3

class ReplayMemory(object):
    def __init__(self, capacity):
        self.device  = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self,event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences = random.sample(self.memory,batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states,next_states,actions,rewards,dones
    
class agent():
    def __init__(self,states_size,action_size):
        self.device  = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
        self.states_size = states_size
        self.action_size = action_size

        self.local_qnetork = Network(state_size,action_size).to(self.device)
        self.target_qnetork = Network(state_size,action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetork.parameters(), lr = learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step= (self.t_step+1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minBatch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences,discount_factor)

    def act(self,state, epsilon= 0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetork.eval()
        with torch.no_grad():
            action_values = self.local_qnetork(state)
        self.local_qnetork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experienes, discount_factor):#nie ogariam tego xd 
        states,next_state,action,rewards,dones = experienes
        next_q_target = self.target_qnetork(next_state).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + (discount_factor * next_q_target * (1- dones))
        q_expected = self.local_qnetork(states).gather(1,actions)
        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetork,self.target_qnetork, interpolation_parameter)

    def soft_update(self ,localModel, TargetModel, interpolation_parameter):
        for targetParam, local_param in zip(TargetModel.parameters(),localModel.parameters()):
            targetParam.data.copy_(interpolation_parameter* local_param.data + (1.0-interpolation_parameter) * targetParam.data) 