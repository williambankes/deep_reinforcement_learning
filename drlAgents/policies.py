# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:27:29 2020

@author: William Bankes
"""

import torch
import torch.nn.functional as F

import random
from .logger import Logger



class dqnPolicy():
    
    """
    Policy class manages the training of the active and target networks as well
    as returning     
    """
    
    def __init__(self, net, create_expl_func, expl_params,
                 learning_rate=1e-4, momentum=0.5, 
                 use_learning_decay=True, lr_decay=0.9,
                 discount=0.999):
        
        self.active_net = net()
        self.target_net = net()
        
        #Metrics       
        self.actions = [i for i in range(self.active_net.out_layer.out_features)]
        
        self.target_net.load_state_dict(self.active_net.state_dict())
        self.target_net.eval()
        
        self.discount = discount
        self.use_learning_decay = use_learning_decay
        self.optimizer = torch.optim.RMSprop(self.active_net.parameters(),
                                             lr=learning_rate,
                                             momentum=momentum)
        
        self.batch_norm = False        
        self.exploration_func = create_expl_func(**expl_params)
        
        self.steps_done = 0
        
        if use_learning_decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                lr_decay)
        
    def update_target(self):
        
        """
        Update the target network by directly copying the weights of the active
        net work directly
        """       
        
        self.target_net.load_state_dict(self.active_net.state_dict())
        self.target_net.eval()
        
        
        
                
    def action(self, state):
        
        """
        Return action given state, with random exploration that decreases as
        training continues. The shape of this exploration is determined by the
        exploration_func
        
        state -> (torch.tensor) input observation state
        """
        
        
        self.active_net.eval()
        
        r = random.random()
        eps_threshold = self.exploration_func(self.steps_done)
        self.steps_done += 1
                
        if r > eps_threshold:
            with torch.no_grad():
                action = self.active_net(state.unsqueeze(0)).max(1)[1].item()
            
        else:
            action = random.sample(self.actions, 1)[0]
            
        return action
    
    
    
                
    def greedy_action(self, state):
        """
        Return the action with the greatest state-action value        
        
        state -> (torch.tensor) input observation state
        """
        
        self.active_net.eval()
        
        with torch.no_grad():
            return self.active_net(state.unsqueeze(0)).max(1)[1].item()
        
        
        
        
    def train(self, batch):
        """
        Train the active network from a batch of experience. Log the loss and 
        bellman target
        
        batch -> (dict) dictionary containing torch.tensors of the state, action,
                reward, next_state and done
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
        rewards[~(done)] += self.discount * self.target_net(next_states[~(done)]).max(1)[0].float()   
                               
        loss = F.smooth_l1_loss(state_action_value, rewards.view(-1, 1))
        
        with torch.no_grad():
            Logger.getInstance().add('loss', loss.item())
            Logger.getInstance().add('bellman_target', torch.mean(rewards).item())
        
        self.optimizer.zero_grad()
        loss.backward()
     
        for param in self.active_net.parameters():
            param.grad.data.clamp_(-1, 1)
           
        self.optimizer.step()
        
        if self.use_learning_decay:
            self.scheduler.step()        
        return