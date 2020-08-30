# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:30:23 2020

@author: William Bankes
"""

from .replayMemory import replayMemory
from .logger import Logger

import torch

class dqnAgent():
    """
    Implementation of the DQN agent described in the paper 'Playing Atari with 
    Deep Reinforcement Learning'. 
    
    env -> open ai gym environment the agent is to train on
    
    policy -> policy object
    
    min_mem -> (int) minimum memory before the policy net begins training from
                samples from the replay memory
                
    max_mem -> (int) maximum memory available in the replay memory cache
    
    batch_size -> (int) batch update size, for the policy net
    
    target_update -> (int) number of epochs between updates of the target net
    """
    
    
    def __init__(self, env, policy,
                 min_mem=10_000, max_mem=20_000,
                 batch_size=128, target_update=20):
        
        self.env = env
        self.mem = replayMemory(min_mem, max_mem)
        self.policy = policy
        
        self.batch_size = batch_size
        self.target_update = target_update
                
        
        
    def fill_replay_memory(self):
        """
        Fill the replay memory to min_mem by sampling random actions
        from the sample space 
        """
        
        
        while self.mem.fill():
            
            done = False
            state = self.env.reset()
            
            while not done:
                
                action = self.env.action_space.sample()
                
                next_state, reward, done, _ = self.env.step(action) 
                
                self.mem.add((state, action, reward, next_state, done))
                
                state = next_state
                
        
    def train(self, epochs):
        """
        Train the policy net on sampled experience from the environment for a
        predetermined number of epochs. Each epoch is one full simulation of the
        environment.
        
        epochs -> (int) number of epochs to train the model over 
        """        

        self.fill_replay_memory()
                
        for epoch in range(epochs):
            
            done = False            
            ep_reward = 0
            state = self.env.reset()
            
            #Limit the length of the simulation
            for i in range(1000):
                
                action = self.policy.action(torch.tensor(state, dtype=torch.float))
                
                next_state, reward, done, _ = self.env.step(action)
                self.mem.add((state, action, reward, next_state, done))
                                       
                if done:
                    Logger.getInstance().add('ep_duration', i)
                    break
                    
                state = next_state
                ep_reward += reward
                
                self.policy.train(self.mem.sample(self.batch_size))
                
            #Log the episode reward and update the target network
            Logger.getInstance().add('ep_reward', ep_reward)
                        
            if (epoch % self.target_update) == 0:
                self.policy.update_target()
                
                
                
                
                
                
    def play(self, length=500, output=False):
        """
        Simulate and render an environment from the beginning with the current policy
        providing greedy actions.
        
        length -> (int) run time of the simulation
        
        output -> (rgb_array) return the frames as an rgb_array of values that can
                    be plotted. (Might be more fitting to use the Logger)
        """        
        
        done = False
        state = self.env.reset()        
        frames = list()        
        
        for _ in range(length):
            if output:
                frames.append(self.env.render('rgb_array'))
            else:
                self.env.render()
            
            action = self.policy.action(torch.tensor(state, dtype=torch.float))           
            state, reward, done, _ = self.env.step(action)
            
        self.env.close()
        return frames
        
    def play_random(self, length=500):
        """
        Simulate and render an environment from the beginning with a random policy
        providing actions

        length -> (int) run time of the simulation        
        """
        
        
        self.env.reset()
        
        for _ in range(length):
            self.env.render()
            self.env.step(self.env.action_space.sample())
            
        self.env.close()