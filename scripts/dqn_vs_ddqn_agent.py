# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:02:58 2020

A comparison of dqn vs ddqn agents and there performance on the CartPole-v0 
environment

@author: William Bankes
"""

#%%
from drlAgents.policies import dqnPolicy, ddqnPolicy
from drlAgents.agents import dqnAgent
from drlAgents.logger import Logger
from drlAgents.explorationFunctions import create_factor_min_exploration

import gym
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
    
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.bn1(self.h1(x)))
        x = F.relu(self.bn2(self.h2(x)))
        return self.out_layer(x)
    
#%%
#Script: repeat each hyperparameter set 10 times


policy_params = {
        'discount':0.999,
        'learning_rate': 0.0001,
        'momentum':0.5,
        'use_learning_decay':True,
        'lr_decay':1}

agent_params = {
        'min_mem':10_000,
        'max_mem':50_000,
        'batch_size':128,
        'target_update':10,
        }

env_params = {
        'env_string':'CartPole-v0',
        'seed':42}

expl_params = {'eps_start':0.95,
               'eps_end':0.1,
               'eps_decay':0.995}


def dqn_experiment(policy_params, agent_params, expl_params, env_params):
    
    Logger.getInstance().clear()
    
    env = gym.make(env_params['env_string'])
    env._max_episode_steps = 4000
    env.seed(env_params['seed'])
    env.action_space.seed(env_params['seed'])
    
    p = dqnPolicy(Net, create_factor_min_exploration, expl_params,
                  **policy_params)
    d = dqnAgent(env, p, **agent_params)
    
    d.train(300)

def ddqn_experiment(policy_params, agent_params, expl_params, env_params):
    
    Logger.getInstance().clear()
    
    env = gym.make(env_params['env_string'])
    env._max_episode_steps = 4000
    env.seed(env_params['seed'])
    env.action_space.seed(env_params['seed'])
    
    p = ddqnPolicy(Net, create_factor_min_exploration, expl_params,
                  **policy_params)
    d = dqnAgent(env, p, **agent_params)
    
    d.train(300)
    
    
def do_experiments(policy_params, agent_params, expl_params, env_params, runs):
    
    dqn_reward, ddqn_reward = list(), list()
    
    for i in range(runs):
        
        print(i)
        dqn_experiment(policy_params, agent_params, expl_params, env_params)
        dqn_reward.append(Logger.getInstance().get('ep_reward'))
        ddqn_experiment(policy_params, agent_params, expl_params, env_params)
        ddqn_reward.append(Logger.getInstance().get('ep_reward'))
    
    return dqn_reward, ddqn_reward


dqn_reward, ddqn_reward = do_experiments(policy_params, agent_params,
                                         expl_params, env_params, 10)

#%%


def compare_dqn_ddqn(dqn_data, ddqn_data):
    
    #Compute averages
    mean_dqn_data = np.mean(np.array(dqn_data, np.float), axis=0)
    mean_ddqn_data = np.mean(np.array(ddqn_data, np.float), axis=0)

    fig, axs = plt.subplots()
    
    axs.plot(mean_dqn_data, label='mean_dqn_reward')
    axs.plot(mean_ddqn_data, label='mean_ddqn_reward')
    
    axs.set_title('Average episode reward of dqn and ddqn')
    axs.set_xlabel('episode')
    axs.set_ylabel('mean reward')
    
    plt.legend()
    
compare_dqn_ddqn(dqn_reward, ddqn_reward)



