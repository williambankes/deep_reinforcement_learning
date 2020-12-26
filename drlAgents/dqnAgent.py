# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 18:34:16 2020

@author: William
"""

from .agents import Agent
from .policies import dqnPolicy
from .explorationFunctions import create_factor_min_exploration


#Should this inherit from Agent? Should we abstract this further

default_expl_params = {'eps_start':0.95,
                       'eps_end':0.1,
                       'eps_decay':0.995}

class DQNAgent:
    
    def __init__(self, env, net, min_mem=10_000, max_mem=50_000,
                 batch_size=128, target_update=10, learning_rate=1e-4,
                 momentum=0.5, use_learning_decay=True, lr_decay=1,
                 discount=0.999, create_expl_func=create_factor_min_exploration,
                 expl_params=default_expl_params):
        
        policy = dqnPolicy(net,
                           create_expl_func=create_expl_func,
                           expl_params=expl_params,
                           learning_rate=learning_rate,
                           momentum=momentum,
                           use_learning_decay=use_learning_decay,
                           lr_decay=lr_decay,
                           discount=discount)
        
        self.agent = Agent(env, policy, 
                           min_mem=min_mem,
                           max_mem=max_mem,
                           batch_size=batch_size,
                           target_update=target_update)
        
        
        #Need to access the logged data
    
    def train(self, epochs):
        self.agent.train(epochs=epochs)
        
    def play(self, length=200, output=False, random=False):
        return self.agent.play(length=length, output=output, random=random)
        