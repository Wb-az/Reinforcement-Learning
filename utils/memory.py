#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: aze_ace
"""

import random
import numpy as np

# code sourced from
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html



class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, state, action, next_state, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, next_state, reward, done)
        self.position = (self.position + 1) % self.capacity
    
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        state, action, next_state, reward, done = map(np.stack, zip(*batch))
        
        
        return state, action, next_state, reward, done
    
    def __len__(self):
        
        return len(self.memory)

    

  
