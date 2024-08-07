import random 
import numpy as np 
import torch 
import torch.optim as optim
import torch.nn.functional as F
from network import Network
from replayMemory import ReplayMemory

learning_rate = 5e-4
min_batch_size = 100
discount_factor = 0.99
replay_buffer_size = int(1e5)
interpolation_parameter = 1e-3

class Agent():
    def __init__(self, states_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.states_size = states_size
        self.action_size = action_size

        self.local_qnetwork = Network(states_size, action_size).to(self.device)
        self.target_qnetwork = Network(states_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > min_batch_size:
                experiences = self.memory.sample(100)
                self.learn(experiences, discount_factor)

    def act(self, state, epsilon=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self, experiences, discount_factor):
        states, next_state, action, rewards, dones = experiences
        next_q_target = self.target_qnetwork(next_state).detach().max(1)[0].unsqueeze(1)
        q_target = rewards + (discount_factor * next_q_target * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, action)
        loss = F.mse_loss(q_expected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, interpolation_parameter)

    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)