# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 19:26:28 2020

@author: William Bankes
"""
import torch
import random


class replayMemory():
    
    """
    Simple replay_memory class that stores up to max_memory observations from the
    environment. The sampling is currently done at random but a future project
    may want to introduce a more complex method of sampling
    
    min_memory -> (int) minimum memory before the memory is 'full'
    
    max_memory -> (int) maximum memory before samples popped
    """
    
    
    def __init__(self, min_memory, max_memory):
        self.min_memory = min_memory
        self.max_memory = max_memory
        
        self.memory = list()
        
    def __len__(self):
        return len(self.memory)
    
    def fill(self):
        if len(self.memory) > self.min_memory:
            return False
        else:
            return True
        
    def add(self, element):        
        
        self.memory.append(element)
        
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        
    def sample(self, batch_size):
        s = random.sample(self.memory, batch_size)
        
        output = {'state':torch.tensor([x[0] for x in s], dtype=torch.float)}
        output['action'] = torch.tensor([x[1] for x in s], dtype=torch.long)
        output['reward'] = torch.tensor([x[2] for x in s], dtype=torch.float)
        output['next_state'] = torch.tensor([x[3] for x in s], dtype=torch.float)
        output['done'] = torch.tensor([x[4] for x in s], dtype=torch.bool)
        
        return output