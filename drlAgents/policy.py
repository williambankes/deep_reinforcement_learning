# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:27:29 2020

@author: William Bankes
"""

import torch
import torch.nn.functional as F

import random
import math

from .logger import Logger

class Policy():
    
    """
    Policy class handles the neural network training and manages the target and
    sample networks
    
    - Should the policy know the action_space_size or should the Net carry that info
    
    """
    
    def __init__(self, net, discount=0.999, eps_start=0.9,
                 eps_end=0.05, eps_decay=500, learning_rate=1e-4,
                 momentum=0.5):
        
        self.active_net = net()
        self.target_net = net()
        
        #Metrics       
        self.actions = [i for i in range(self.active_net.out_layer.out_features)]
        
        self.target_net.load_state_dict(self.active_net.state_dict())
        self.target_net.eval()
        
        self.discount = discount
        self.optimizer = torch.optim.RMSprop(self.active_net.parameters(),
                                             lr=learning_rate,
                                             momentum=momentum)     
                
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        
    def update_target(self):
        self.target_net.load_state_dict(self.active_net.state_dict())
        self.target_net.eval()
        
        #log values:
        Logger.getInstance().add('target_weights', self.target_net.state_dict())
        
        
    def action(self, state):
        """
        Return action given state, with random exploration
        """
        self.active_net.eval()
        r = random.random()
        
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                    math.exp(-1. * self.steps_done / self.eps_decay)
                    
        self.steps_done += 1
        Logger.getInstance().add('ep_threshold', eps_threshold)
        
        if r > eps_threshold:
            with torch.no_grad():
                action = self.active_net(state.unsqueeze(0)).max(1)[1].item()
            
        else:
            action = random.sample(self.actions, 1)[0]
            
        return action
                
    def greedy_action(self, state):
        
        self.active_net.eval()
        
        with torch.no_grad():
            return self.active_net(state.unsqueeze(0)).max(1)[1].item()
        
    def train(self, batch):
        """
        Could re-do batch as a dictionary to allow for more flexibility
        """
        self.active_net.train()
        #Load in the various states:
        states = batch['state']
        actions = batch['action'].view(-1,1)
        rewards = batch['reward']
        next_states = batch['next_state']
        done = batch['done']

        #Calculate the state_action_value
        state_action_value = self.active_net(states).gather(1, actions)
        
        #Create the next_state_action_values:
        next_state_max_action_value = torch.zeros(len(states))
        
        next_state_max_action_value[~(done)] =\
                                self.target_net(next_states[~(done)]).max(1)[0].float()   
  
        time_difference_eval = rewards + \
                                (self.discount * next_state_max_action_value)
                                                     
        loss = F.smooth_l1_loss(state_action_value, time_difference_eval.view(-1, 1))
        
        Logger.getInstance().add('loss', loss)
        Logger.getInstance().add('bellman_target', torch.mean(time_difference_eval))
        
        self.optimizer.zero_grad()
        loss.backward()
     
        for param in self.active_net.parameters():
            param.grad.data.clamp_(-1, 1)
           
        self.optimizer.step()
        return