import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random
np.set_printoptions(threshold=sys.maxsize)

from q_model import Model, QTrainer

MAX_MEMORY = 20_000
BATCH_SIZE = 5_000
GAMMA = 0.99
LEARNING_RATE = 0.0001
N_EXPLORATION = 100

class Agent:
    def __init__(self, state_size=None, action_size=None, shared_model=None, load_model=False):
        self.action_size = action_size
        if (shared_model == None):
            assert state_size != None and action_size != None, "No model. Either provide a model or state and action size must exists"
            if (load_model):
                if (not self.model.load()):
                    assert state_size != None and action_size != None, "Model failed to load. Either provide a model or state and action size must exists"
            else:
                self.model = Model(state_size, action_size).double()
        else:
            # ! No checking if model is valid
            self.model = shared_model
            load_model = False

        self.trainer = QTrainer(self.model, GAMMA, LEARNING_RATE)
        self.memory = deque(maxlen=MAX_MEMORY)

    def calculate_action(self, state : np.ndarray, n_game, open_ai_env=None):
        epsilon = N_EXPLORATION - n_game
        # print(random.randrange(0, N_EXPLORATION), '<', epsilon, '=', random.randrange(0, N_EXPLORATION) < epsilon)
        if (random.randrange(0, N_EXPLORATION) < epsilon):
            action = np.zeros(self.action_size)
            action[open_ai_env.action_space.sample()] = 1
        else:
            state = torch.tensor(state, dtype=torch.double)
            action : torch.Tensor = self.model(state)
            action = F.one_hot(torch.argmax(action), self.action_size)
            action = action.numpy() 

        return action
        # return np.array([1, 0, 0, 0, 0, 0, 0])

    def save_transition(self, transition):
        self.memory.append(transition)
    
    def train_short(self, transition):
        self.trainer.train_model([transition])

    def train_long(self):
        if (len(self.memory) > BATCH_SIZE):
            transitions = random.sample(self.memory, BATCH_SIZE)
        else:
            transitions = self.memory
        self.trainer.train_model(transitions)

    def save_model(self):
        self.model.save()
    
    def get_loss(self):
        return self.trainer.losses
