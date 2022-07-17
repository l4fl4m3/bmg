import gym, random
from gym.spaces import Discrete
import numpy as np

class TwoColorGridWorld(gym.Env):
    def __init__ (self):

        self.l = 5
        self.size = self.l * self.l
        self.action_space = Discrete(4)
        self.random_squares = random.sample(range(self.size), 3)  # g,b,r == 0,1,2
        self.rewards = [-0.04, 1, -1] # g,b,r
        self.threshold = 100_000
        self.step_count = 0

        self.observation_space = self.make_space()

    def make_space(self):
        coordinates = []
        coordinates.append([self.random_squares[0] // self.l, self.random_squares[0] % self.l])
        coordinates.append([self.random_squares[1] // self.l, self.random_squares[1] % self.l])
        coordinates.append([self.random_squares[2] // self.l, self.random_squares[2] % self.l])
        obs_space = []
        for c in coordinates:
            one_hot = np.eye(self.l)[c]
            obs_space.append(one_hot)
        obs_space = np.array(obs_space).flatten()
        return obs_space

    def move_green(self, action):
        #up
        if action == 0:
            if self.random_squares[0] - self.l >= 0: 
                self.random_squares[0] = self.random_squares[0] - self.l
        #down
        elif action == 1:
            if self.random_squares[0] + self.l <= self.size - 1: 
                self.random_squares[0] = self.random_squares[0] + self.l
        #left        
        elif action == 2:
            if self.random_squares[0] % self.l != 0: 
                self.random_squares[0] -= 1
        #right
        elif action == 3:
            if (self.random_squares[0] + 1) % self.l != 0:                                                 
                self.random_squares[0] += 1 

        self.step_count += 1

    def move_captured(self, color):
        temp = [i for i in range(self.size) if i not in self.random_squares]
        if color == 'b':
            self.random_squares[1] = random.sample(temp, 1)[0]
        else:
            self.random_squares[2] = random.sample(temp, 1)[0]          
    
    def allocate_reward(self):
        if self.random_squares[0] == self.random_squares[1]:
            reward = self.rewards[1]
            self.move_captured('b')
            done = True
        elif self.random_squares[0] == self.random_squares[2]:
            reward = self.rewards[2]
            self.move_captured('r')
            done = True
        else:
            reward = self.rewards[0]
            done = False

        if self.step_count == self.threshold: 
            self.rewards[1], self.rewards[2] = self.rewards[2], self.rewards[1]
            self.step_count = 0

        return reward, done

    def step(self, action):

        # move green piece
        self.move_green(action)

        # allocate rewards
        reward, done = self.allocate_reward()

        # update space
        self.observation_space = self.make_space()

        return self.observation_space, reward, done, {}

    def reset(self):
        return self.observation_space

    '''
    def set_seed(self, random_seed):
        random.seed(random_seed)
        self.random_squares = random.sample(range(self.size), 3)  # g,b,r == 0,1,2
        self.observation_space = self.make_space()
    '''
    

#e = TwoColorGridWorld()