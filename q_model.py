import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np

MODEL_PATH = "./model/"
MODEL_FILE = "model_reduced_height.pth"

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        # Input: State
        # Output: Quality of every action (Q-Value of every action)
        super().__init__()

        self.linear1 = nn.Linear(input_size, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        return x

    def save(self):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        torch.save(self.state_dict(), MODEL_PATH + MODEL_FILE)
    
    def load(self):
        if (os.path.exists(MODEL_PATH + MODEL_FILE)):
            self.load_state_dict(torch.load(MODEL_PATH + MODEL_FILE))
            self.eval()
            print(MODEL_PATH+MODEL_FILE, "loaded")
        else:
            print("Path doesnt exists. Loading failed")

class QTrainer:
    def __init__(self, model, gamma, lr):
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr)
        self.loss_function = nn.MSELoss()

    def train_model(self, transition):
        prev_state, action, next_state, reward, finished = zip(*transition)
        
        prev_state = torch.tensor(np.array(prev_state), dtype=torch.double)
        next_state = torch.tensor(np.array(next_state), dtype=torch.double)

        memory_length = len(finished)

        q_old = self.model(prev_state)
        q_new = q_old.clone()
        for idx in range(memory_length):
            if (finished):
                q_new[idx][action[idx].argmax()] = reward[idx]
            else:
                # The label for quality of this action is current reward + maximum quality of next action
                # Quality is high if reward is high and there's a high quality possible move for next state
                q_new[idx][action[idx].argmax()] = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
        self.optimizer.zero_grad()
        loss = self.loss_function(q_old, q_new)
        loss.backward()

        self.optimizer.step()