# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:26:22 2020

@author: William Bankes
"""
#%%

from drlAgents.policies import dqnPolicy
from drlAgents.agents import dqnAgent
from drlAgents.explorationFunctions import create_factor_min_exploration

import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
import imageio

torch.manual_seed(42)

#%%

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        a = 20
        
        self.in_layer = nn.Linear(4,a)
        self.h1 = nn.Linear(a,a)
        self.bn1 = nn.BatchNorm1d(a)
        self.h2 = nn.Linear(a,a)
        self.bn2 = nn.BatchNorm1d(a)
        self.out_layer = nn.Linear(a,2)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.bn1(self.h1(x)))
        x = F.relu(self.bn2(self.h2(x)))
        return self.out_layer(x)
    
policy_params = {
        'discount':0.999,
        'learning_rate': 0.0001,
        'momentum':0.5,
        'use_learning_decay':True,
        'lr_decay':1
        }

agent_params = {
        'min_mem':10_000,
        'max_mem':50_000,
        'batch_size':128,
        'target_update':10,
        }

env_params = {
        'env_string':'CartPole-v0',
        'seed':42}

expl_params = {'eps_start':0.9,
               'eps_end':0.1,
               'eps_decay':0.995}


#%%

env = gym.make(env_params['env_string'])
env.seed(env_params['seed'])
env.action_space.seed(env_params['seed'])

p = dqnPolicy(Net, create_factor_min_exploration, expl_params,
              **policy_params)
d = dqnAgent(env, p, **agent_params)

d.train(200)

#%%

frames = d.play(output=True)

#%%

kwargs_write = {'fps':1.0, 'quantizer':'nq'}
imageio.mimsave('./cartpole.gif', frames[0:250], fps=60)















