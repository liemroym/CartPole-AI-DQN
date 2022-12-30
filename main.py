import gym
from agent import Agent
from q_model import Model
from multiprocessing import Process
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import pygame

class TrainProcess:
    def __init__(self, id, model, max_n_game=1_000):
        # Implementation using shared model
        self.id = id
        self.env = gym.make('CartPole-v1', render_mode="human")   
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.agent = Agent(state_size=state_size, action_size=action_size, shared_model=model, load_model=True)
        # agent = Agent(env.get_state().shape[0], env.get_actions(), load_model=True)
        self.max_n_game = max_n_game
        
        self.process = Process(target=self.start_process)


    def start_process(self):
        n_game = 0
        prev_state, info = self.env.reset()
        # prev_state = 2.*(prev_state - np.min(prev_state))/np.ptp(prev_state)-1
        game_reward = 0
        max_reward = 0
        reward_list = []
        avg_reward_list = []
        # while True:
        while n_game < self.max_n_game:
            action = self.agent.calculate_action(prev_state, n_game, self.env)
            
            new_state, reward, finished, truncated, info  = self.env.step(action.argmax())
            # new_state = 2.*(new_state - np.min(new_state))/np.ptp(new_state)-1
            
            if (finished or truncated):
                reward = -5
            game_reward += reward

            transition = (prev_state, action, new_state, reward, finished or truncated)
            self.agent.save_transition(transition)
            self.agent.train_short(transition)

            if (finished or truncated):
                n_game +=  1
                self.agent.train_long()
                self.agent.save_model()

                reward_list.append(game_reward)
                avg_reward_list.append(np.mean(reward_list))
                
                # print(f"Game: {n_game} --> Score: {self.env.get_score()}, Fitness: {self.env.get_fitness()}")

                print(f"Game: {n_game} --> Score: {game_reward}")
                max_reward = max(max_reward, game_reward)
                game_reward = 0
                prev_state, info = self.env.reset()
            else:
                prev_state = new_state
        
        print(f"Process {self.id} finished. Max score: {max_reward}")

        while True:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            plt.clf()
            plt.title('Training...')
            plt.xlabel('Number of Games')
            plt.ylabel('Loss')
            plt.plot(reward_list)
            plt.plot(avg_reward_list)
            plt.show(block=False)
            plt.pause(.1)
        
if __name__ == "__main__":
    pygame.init()
    
    temp = gym.make('CartPole-v1')
    # env action --> Discrete --> int
    model = Model(temp.observation_space.shape[0], temp.action_space.n).double()
    model.load()
    
    del temp

    processes = []
    for i in range(3):
        p = TrainProcess(id=i, model=model, max_n_game=200)
        # p.start_process()
        p.process.start()
        processes.append(p)
    
    for p in processes:
        p.process.join()

        