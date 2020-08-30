# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:27:29 2020

@author: William Bankes
"""

import torch
import torch.nn.functional as F

import random
from .logger import Logger


class eGreedyPolicy():
    
    """
    Parent policy that implements the action and greedy_action methods for more
    advanced policies that use an exploration threshold.
    """
    
    
    def __init__(self, net, create_expl_func, expl_params):
        
        self.func = net()

        self.actions = [i for i in range(self.func.out_layer.out_features)]
        
        self.exploration_func = create_expl_func(**expl_params)
        self.eps_threshold = 1
        
        self.steps_done = 0
        
    def action(self, state):
        
        """
        Return action given state, with random exploration that decreases as
        training continues. The shape of this exploration is determined by the
        exploration_func
        
        state -> (torch.tensor) input observation state
        """
        
        self.func.eval()
        
        r = random.random()
        self.eps_threshold = self.exploration_func(self.steps_done)
        
        self.steps_done += 1
                
        if r > self.eps_threshold:
            action = self.greedy_action(state)
            
        else:
            action = random.sample(self.actions, 1)[0]
            
        return action
    
    
    def greedy_action(self, state):
        """
        Return the action with the greatest state-action value        
        
        state -> (torch.tensor) input observation state
        """
        
        self.func.eval()
        
        with torch.no_grad():
            return self.func(state.unsqueeze(0)).max(1)[1].item() 
        
    

class ddqnPolicy(eGreedyPolicy):
    
    def __init__(self, net, create_expl_func, expl_params,
                 learning_rate=1e-4, momentum=0.5, 
                 use_learning_decay=True, lr_decay=0.9,
                 discount=0.999):
        
        
        super().__init__(net, create_expl_func, expl_params)

        self.target_net = net()
        
        self.target_net.load_state_dict(self.func.state_dict())
        self.target_net.eval()
        
        self.discount = discount
        self.use_learning_decay = use_learning_decay
        self.optimizer = torch.optim.RMSprop(self.func.parameters(),
                                                 lr=learning_rate,
                                                 momentum=momentum)
        
        if use_learning_decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                lr_decay)

    def update_target(self):
        
        """
        Update the target network by directly copying the weights of the active
        net work directly
        """       
        
        self.target_net.load_state_dict(self.func.state_dict())
        self.target_net.eval()

    def train(self, batch):
        """
        Train the active network from a batch of experience. Log the loss and 
        bellman target
        
        batch -> (dict) dictionary containing torch.tensors of the state, action,
                reward, next_state and done
        """
        
        
        self.func.train()
        #Load in the various states:
        states      =   batch['state']
        actions     =   batch['action'].view(-1,1)
        rewards     =   batch['reward'].view(-1,1)
        next_states =   batch['next_state']
        done        =   batch['done']

        #Calculate the state_action_value
        state_action_value = self.func(states).gather(1, actions)
        
        #Create the next_state_action_values:
        with torch.no_grad():
            active_actions = self.func(next_states[~(done)]).max(1)[1].view(-1,1).long()
            
        rewards[~(done)] += self.discount * self.target_net(next_states[~(done)]).gather(1, active_actions)
 
        loss = F.smooth_l1_loss(state_action_value, rewards)
        
        with torch.no_grad():
            Logger.getInstance().add('loss', loss.item())
            Logger.getInstance().add('bellman_target', torch.mean(rewards).item())
        
        self.optimizer.zero_grad()
        loss.backward()
     
        for param in self.func.parameters():
            param.grad.data.clamp_(-1, 1)
           
        self.optimizer.step()
        
        if self.use_learning_decay:
            self.scheduler.step()
            
        return

class dqnPolicy(eGreedyPolicy):
    
    
    def __init__(self, net, create_expl_func, expl_params,
                 learning_rate=1e-4, momentum=0.5, 
                 use_learning_decay=True, lr_decay=0.9,
                 discount=0.999):
        
        
        super().__init__(net, create_expl_func, expl_params)

        self.target_net = net()
        
        self.target_net.load_state_dict(self.func.state_dict())
        self.target_net.eval()
        
        self.discount = discount
        self.use_learning_decay = use_learning_decay
        self.optimizer = torch.optim.RMSprop(self.func.parameters(),
                                                 lr=learning_rate,
                                                 momentum=momentum)
        
        if use_learning_decay:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                    lr_decay)
    
    
    def update_target(self):
        
        """
        Update the target network by directly copying the weights of the active
        net work directly
        """       
        
        self.target_net.load_state_dict(self.func.state_dict())
        self.target_net.eval()

    def train(self, batch):
        """
        Train the active network from a batch of experience. Log the loss and 
        bellman target
        
        batch -> (dict) dictionary containing torch.tensors of the state, action,
                reward, next_state and done
        """
        
        
        self.func.train()
        #Load in the various states:
        states      = batch['state']
        actions     = batch['action'].view(-1,1)
        rewards     = batch['reward'].view(-1,1)
        next_states = batch['next_state']
        done        = batch['done']

        #Calculate the state_action_value
        state_action_value = self.func(states).gather(1, actions)
        
        #Create the next_state_action_values:            
        rewards[~(done)] += self.discount * self.target_net(next_states[~(done)]).max(1)[0].view(-1,1).float()   
                  
             
        loss = F.smooth_l1_loss(state_action_value, rewards)
        
        with torch.no_grad():
            Logger.getInstance().add('loss', loss.item())
            Logger.getInstance().add('bellman_target', torch.mean(rewards).item())
        
        self.optimizer.zero_grad()
        loss.backward()
     
        for param in self.func.parameters():
            param.grad.data.clamp_(-1, 1)
           
        self.optimizer.step()
        
        if self.use_learning_decay:
            self.scheduler.step()        
        return