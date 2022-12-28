from environment import Game
from agent import Agent
from q_model import Model
from multiprocessing import Process
import pygame

class TrainProcess:
    def __init__(self, id, model, max_iter):
        # Implementation using shared model
        self.id = id
        self.env = Game()    
        self.agent = Agent(shared_model=model)
        # agent = Agent(env.get_state().shape[0], env.get_actions(), load_model=True)
        self.max_iter = max_iter
        
        self.process = Process(target=self.start_process)


    def start_process(self):
        n_game = 0
        for _ in range(self.max_iter):
            curr_state = self.env.get_state()
            # action = np.array(self.env.handle_control())
            action = self.agent.calculate_action(curr_state, n_game)
            finished, reward = self.env.do_action(action)
            new_state = self.env.get_state()

            transition = (curr_state, action, new_state, reward, finished)
            self.agent.save_transition(transition)
            self.agent.train_short(transition)

            if (finished):
                n_game +=  1
                self.agent.train_long()
                self.agent.save_model()

                print(f"Game: {n_game} --> Score: {self.env.get_score()}, Fitness: {self.env.get_fitness()}")
                self.env.reset()
            
        print(f"Process {self.id} finished")

        
if __name__ == "__main__":
    pygame.init()
    temp = Game()
    model = Model(temp.get_state().shape[0], temp.get_actions()).double()
    model.load()
    del temp

    processes = []
    for i in range(2):
        p = TrainProcess(id=i, model=model, max_iter=100)
        p.process.start()
        processes.append(p)

        