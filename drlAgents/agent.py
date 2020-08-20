# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:30:23 2020

@author: William Bankes
"""

from .replayMemory import Replay_Memory
from .logger import Logger

import torch

class DQN_Agent():
    
    def __init__(self, env, policy,
                 min_mem=10_000, max_mem=20_000,
                 batch_size=128, target_update = 20):
        
        self.env = env
        self.mem = Replay_Memory(min_mem, max_mem)
        self.policy = policy
        
        self.batch_size = batch_size
        self.target_update = target_update
                
    def fill_replay_memory(self):

        while self.mem.fill():
            
            done = False
            state = self.env.reset()
            
            while not done:
                
                action = self.env.action_space.sample()
                
                next_state, reward, done, _ = self.env.step(action) 
                
                self.mem.add((state, action, reward, next_state, done))
                
                state = next_state
                
        
    def train(self, epochs):
        
        #fill the replay memory
        self.fill_replay_memory()
                
        for epoch in range(epochs):
            
            print(epoch)
            done = False            
            state = self.env.reset()

            
            ep_reward = 0
            #Limit the length of the iteration
            for i in range(1000):
                
                action = self.policy.action(torch.tensor(state,
                                                         dtype=torch.float))
                
                next_state, reward, done, _ = self.env.step(action)
                
                self.mem.add((state, action, reward, next_state, done))
                                       
                if done:
                    Logger.getInstance().add('ep_duration', i)
                    break
                    
                state = next_state
                
                ep_reward += reward
                
                self.policy.train(self.mem.sample(self.batch_size))
                
            Logger.getInstance().add('ep_reward', ep_reward)
            
            if (epoch % self.target_update) == 0:
                self.policy.update_target()
                
    def play(self, length=500):
        
        state = self.env.reset()        
        done = False
        
        for _ in range(length):
            self.env.render()          
            
            action = self.policy.action(torch.tensor(state, dtype=torch.float))
            
            print(action)
            
            state, reward, done, _ = self.env.step(action)
            
        self.env.close()
        
    def play_random(self):
        
        self.env.reset()
        
        for _ in range(500):
            self.env.render()
            self.env.step(self.env.action_space.sample()) # take a random action
        self.env.close()