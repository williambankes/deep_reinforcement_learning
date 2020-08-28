# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:02:58 2020

Understand how certain hyperparameters affect the performance of the DQN

#final exploration rate...?

@author: William Bankes
"""

#%%

from comet_ml import Experiment

from drlAgents.policies import dqnPolicy
from drlAgents.agents import dqnAgent
from drlAgents.logger import Logger
from drlAgents.explorationFunctions import create_exponential_exploration

import gym
import matplotlib.pyplot as plt
import numpy as np


import torch.nn as nn
import torch.nn.functional as F


#%%
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        a = 8
        
        self.in_layer = nn.Linear(4,a)
        self.h1 = nn.Linear(a,a)
        self.bn1 = nn.BatchNorm1d(a)
        self.h2 = nn.Linear(a,a)
        self.bn2 = nn.BatchNorm1d(a)
        self.h3 = nn.Linear(a,a)
        self.bn3 = nn.BatchNorm1d(a)
        self.out_layer = nn.Linear(a,2)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.bn1(self.h1(x)))
        x = F.relu(self.bn2(self.h2(x)))
        x = F.relu(self.bn3(self.h3(x)))
        return self.out_layer(x)
    

class Net_nb(nn.Module):

    def __init__(self):
        super(Net_nb, self).__init__()
        
        self.in_layer = nn.Linear(4,64)
        self.h1 = nn.Linear(64,32)
        self.h2 = nn.Linear(32,8)
        self.out_layer = nn.Linear(8,2)
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.in_layer(x))
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        return self.out_layer(x)


#%%

def rescale_stepwise(data, duration):
    
    current_index = 0
    rescaled_data = list()
    
    for dur in duration:
        
        rescaled_data.append(np.mean(data[current_index:dur + current_index]))
        
        current_index += dur
        
    return rescaled_data
    

def plot_data(data, label):
    
    fig, axs = plt.subplots()
    
    for l in data:
        axs.plot(l, alpha=0.2)
    
    mean = np.mean(np.array(data, np.float), axis=0)
    axs.plot(mean, label='average_' + label, alpha = 1)
    
    axs.set_xlabel('Epoch')
    axs.set_ylabel(label)

    plt.legend()

    return fig

#%%
#Script: repeat each hyperparameter set 10 times


policy_params = {
        'discount':0.99,
        'learning_rate': 0.001,
        'momentum':0.25,
        'use_learning_decay':True,
        'lr_decay':1}

agent_params = {
        'min_mem':10_000,
        'max_mem':80_000,
        'batch_size':64,
        'target_update':10,
        }

env_params = {
        'env_string':'CartPole-v1',
        'seed':42}

expl_params = {'eps_start':0.9,
               'eps_end':0.1,
               'eps_decay':100}

"""
experiment = Experiment(api_key = '', 
                        project_name='dqn-initial-hyperparameter-exploration',
                        workspace="williambankes")
experiment.log_parameters(hyper_params)
experiment.log_parameters(agent_params)
experiment.log_parameters(policy_params)
experiment.log_parameters(expl_params)
"""

def dqn_experiment(policy_params, agent_params, expl_params, env_params):
    
    Logger.getInstance().clear()
    
    env = gym.make(env_params['env_string'])
    env.seed(env_params['seed'])
    env.action_space.seed(env_params['seed'])
    
    p = dqnPolicy(Net, create_exponential_exploration, expl_params,
                  **policy_params)
    d = dqnAgent(env, p, **agent_params)
    
    d.train(200)
    
    
def do_experiments(policy_params, agent_params, expl_params, env_params, runs):
    
    loss, reward, bellman_target = list(), list(), list()
    eps_threshold, duration = list(), list()
    
    for i in range(runs):
        
        dqn_experiment(policy_params, agent_params, expl_params, env_params)
    
        loss.append(Logger.getInstance().get('loss'))
        reward.append(Logger.getInstance().get('ep_reward'))
        bellman_target.append(Logger.getInstance().get('bellman_target'))
        eps_threshold.append(Logger.getInstance().get('ep_threshold'))
        duration.append(Logger.getInstance().get('ep_duration'))
    
    return loss, reward, bellman_target, eps_threshold, duration


loss, reward, bellman_target, eps_threshold, duration = do_experiments(policy_params, agent_params, expl_params, env_params, 10)


loss_fig = plot_data([rescale_stepwise(loss[i], duration[i]) for i in range(len(loss))], 'loss')
reward_fig = plot_data(reward, 'reward')
bellman_target_fig = plot_data([rescale_stepwise(bellman_target[i], duration[i]) for i in range(len(loss))], 'bellman_target')
eps_threshold_fig = plot_data([rescale_stepwise(eps_threshold[i], duration[i]) for i in range(len(loss))], 'eps_threshold')

"""
experiment.log_figure(figure=loss_fig)
experiment.log_figure(figure=reward_fig)
experiment.log_figure(figure=bellman_target_fig)
experiment.log_figure(figure=eps_threshold_fig)


experiment.end()
"""



