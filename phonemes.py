#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:01:38 2021

@author: aze_ace
"""

import numpy as np
import copy
import pandas as pd
from collections import namedtuple
import matplotlib.pyplot as plt
import math


spell = "book took drum luck hush brush who tool new jury blush true through sue ball suit knew " \
        "fool loose lose pull room good boot look wolf rug foot sugar put dune hook doom cook " \
        "June cushion one could shoe woods bookshelf blue during rural noodles hush bug woman " \
        "football full would do too soon hood food pool you threw Lou two supper plumber publish " \
        "cup come "

phonemes = "bʊk tʊk drʌm lʌk hʌʃ brʌʃ huː tuːl njuː ʤʊəri blʌʃ truː θruː sjuː bɔːl sjuːt njuː " \
           "fuːl luːs luːz pʊl ruːm gʊd buːt lʊk wʊlf rʌg fʊt ʃʊgə pʊt djuːn hʊk duːm kʊk ʤuːn " \
           "kʊʃən wʌn kʊd ʃuː wʊdz bʊkʃɛlf bluː djʊərɪŋ rʊərəl nuːdlz hʌʃ bʌg wʊmən fʊtbɔːl fʊl " \
           "wʊd duː tuː suːn hʊd fuːd puːl juː θruː luː tuː sʌpə plʌmə pʌblɪʃ kʌp kʌm "

spell_list = list(spell.split(" "))
phonemes_list = list(phonemes.split(" "))

phonetic_dict = dict(zip(phonemes_list, spell_list, ))

# noinspection NonAsciiCharacters
ʊ_sound = list()
# noinspection NonAsciiCharacters
uː_sound = list()
# noinspection NonAsciiCharacters
ʌ_sound = list()

for pho in phonemes_list:
    if 'ʊ' in pho:
        ʊ_sound.append(phonetic_dict[pho])
    elif 'uː' in pho:
        uː_sound.append(phonetic_dict[pho])
    elif 'ʌ' in pho:
        ʌ_sound.append(phonetic_dict[pho])



Action = namedtuple('Action', 'name index delta_x delta_y')


# noinspection NonAsciiCharacters
class Phonemes:

    def __init__(self, size, action, **env_inf):
        """
        param action: a napmedtuple with agent's actions
        param size: is a tuple with number of column and raws
        param env_inf: a dictionary containing informatio about the seed and task
        """
        
        self.size = size
        self.grid = np.zeros(size)
        self.up = action('up', 0, -1, 0)
        self.down = action('down', 1, 1, 0)
        self.left = action('left', 2, 0, -1)
        self.right = action('right', 3, 0, 1)
        self.grab = action('grab', 4, 0, 0)
        self.seed = env_inf['seed']
        self.task = env_inf['sound']
        self.bound = 2 * np.sum(self.size) - 4
        self.area = np.prod(self.size) - self.bound

        # Boundaries
        self.grid[0, :], self.grid[:, 0], self.grid[:, -1], self.grid[-1, :] = 7, 7, 7, 7

        # Words to place on the grid on ly a third of the available area
        self.num_words = self.area // 3

        assert self.num_words >= 3, 'Increase the size of the environment'

        self.obstacles = self.area //10

        # sum all objects in the environment, words, obstacles, goal and agent
        self.total_objects = self.num_words + self.obstacles + 2  # goal + agent
        self.total_agents = self.obstacles + 1  # learner agent

        # Ramdomly choosing words
        self.short_u = np.random.choice(ʊ_sound, self.num_words // 3)
        self.open_middle_a = np.random.choice(ʌ_sound, self.num_words // 3)
        self.long_u = np.random.choice(uː_sound, self.num_words // 3)
        
        self.agent_pos = None
        self.ʊ_pos = None
        self.ʌ_pos = None
        self.uː_pos = None
        self.obstacle_pos = None
        self.goal_pos = None
        self.time_step = 0
        self.time_limit = self.area + self.obstacles
        self.dict_map_display = {0: '_', 1: '*', 2: 'ʊ', 3: 'ʌ', 4: 'u:', 5: 'A', 6: 'G', 7: 'X'}

    def env_step(self, action, prints=True):
        """
        This metods returns the observations, reward and boolean done
        transitions to another position are checked
        if the agent grab in a position with a word then
        the item is remove from the list of words
        after the agent action the obstacles are move and the observation is updated
        """
        done = False
        
        (x, y) = self.agent_pos
        
        if prints: 
            print('Agent position: {} |  Agent action: {} | Goal: {}'.format(self.agent_pos, action,
                                                                        self.goal_pos))
        reward = -1
        self.time_step += 1

        #############################
        # Undertaking an action
        #############################

        if action == self.up.name:

            self.agent_pos = (x + self.up.delta_x, y)

            if self.agent_pos[0] < 1:
                self.agent_pos = (x, y)
                reward -= 10

            elif self.agent_pos in self.obstacle_pos:
                self.agent_pos = (x, y)
                reward -= 20

        elif action == self.down.name:
            self.agent_pos = (x + self.down.delta_x, y)

            if self.agent_pos[0] > self.size[0] - 2:
                self.agent_pos = (x, y)
                reward -= 10

            elif self.agent_pos in self.obstacle_pos:
                self.agent_pos = (x, y)
                reward -= 20

        elif action == self.left.name:

            self.agent_pos = (x, y + self.left.delta_y)
            if self.agent_pos[1] < 1:
                self.agent_pos = (x, y)
                reward -= 10

            elif self.agent_pos in self.obstacle_pos:
                self.agent_pos = (x, y)
                reward -=20

        elif action == self.right.name:
            self.agent_pos = (x, y + self.right.delta_y)

            if self.agent_pos[1] > self.size[1] - 2:
                self.agent_pos = (x, y)
                reward -= 10

            elif self.agent_pos in self.obstacle_pos:
                self.agent_pos = (x, y)
                reward -= 20
                
        elif action == self.grab.name and self.agent_pos in self.ʊ_pos:
            # update list of items left
            self.ʊ_pos.remove(self.agent_pos)
            self.agent_pos = (x, y)

            if self.task == 'short_u':
                reward += 100
            else:
                reward -= -100

        elif action == self.grab.name and self.agent_pos in self.ʌ_pos:
            # update list of items left
            self.ʌ_pos.remove(self.agent_pos)
            self.agent_pos = self.agent_pos

            if self.task == 'middle_open':
                reward += 100
            else:
                reward -= -100

        elif action == self.grab.name and self.agent_pos in self.uː_pos:
            # update list of items left
            self.uː_pos.remove(self.agent_pos)
            self.agent_pos = self.agent_pos
            if self.task == 'long_u':
                reward += 100
            else:
                reward -= 100

        elif action == self.grab.name and self.agent_pos not in (
                self.ʊ_pos + self.ʌ_pos + self.uː_pos):
            self.agent_pos = self.agent_pos
            reward -=100
        else:
            reward = -1

        #############################
        # Verifying terminal state
        #############################

        # Time limit reached
        w = self.num_words//3 - len(self.ʊ_pos)
        if self.time_step == self.time_limit: # and self.agent_pos != self.goal_pos:
            done = True
            
            if w < 1:
                reward -= self.area * 3
            
            elif self.agent_pos == self.goal_pos:
                reward += self.area * w
            
            else: 
                reward += self.area * w - self.area//3

            if prints:
                print('Episode done')
                print('Last reward: {}'.format(reward))
                print('Words with {} sound collected: {}'.format('ʊ', w))   

        elif self.agent_pos == self.goal_pos:# and self.time_step == self.time_limit:
            done = True
            
            if w < 1:
                reward -= self.area * 3 
             
            else:
                reward += self.area * w

            if prints:
                print('Episode done')
                print('Last reward: {}'.format(reward))
                print('Words with {} sound collected: {}'.format('ʊ', w))                                       
        else:
            obst_pos, obs_reward = self.move_obstacles(action)
            reward = reward - obs_reward
            
            if prints:
                print('Step reward: {} | Obstacles positions: {}'. format(reward, obst_pos))
            
        observation = self.observe()

        return observation, reward, done, self.time_step
    
    def move_obstacles(self, action):
        """
        This function moves randomly the obstacles in the grid and updates the list
        of their position self.obstacle_pos for displaying
        """

        obs_reward = 0
        
        for i in range(len(self.obstacle_pos)):

            new_pos = np.array(self.obstacle_pos[i])

            (x, y) = new_pos

            if action == 'up':
                new_pos = (x + self.up.delta_x, y)
                
            elif action == 'down':
                new_pos = (x + self.down.delta_x, y)
                
            elif action == 'left':
                new_pos = (x, y + self.left.delta_y)
                
            elif action == 'right':
                new_pos = (x, y + self.right.delta_y)
            
            else:  
                if self.agent_pos[0] != new_pos[0]:
                    new_pos = (self.agent_pos[0] - new_pos[0], y)   
                if self.agent_pos[1] != new_pos[1]:
                    new_pos = (x, self.agent_pos[1] - new_pos[1])
                  
            if new_pos[0] < 1 or new_pos[0] > self.size[0] - 2:
                new_pos = (x, y)

            elif new_pos[1] < 1 or new_pos[1] > self.size[0] - 2:
                new_pos = (x, y)

            elif new_pos in self.obstacle_pos:
                new_pos = (x, y)

            elif new_pos == self.agent_pos:
                new_pos = (x, y)
                obs_reward = 20
            else:
                obs_reward = 0
            self.obstacle_pos[i] = new_pos

        return self.obstacle_pos, obs_reward

    @staticmethod
    def position_to_index(position, size):
        """
        param position: x,y coordinates
        return: coordinates index
        """
        return np.ravel_multi_index(position, size)

    def observe(self):
        """
        Returns a dictionary of the current observation of the environment
        including distance to the goal, to the obsatcles and the words left
        in the environment. The agent cannot see a word or the goal if an obstacle is
        superimposed, but knows the location of the words.
        """
        o = dict()

        distance_to_obs = list()
        distance_to_task = list()

        # Distance to the obstacles
        for pos in self.obstacle_pos:
            distance_to_obs.append((np.array(pos) - np.array(self.agent_pos)))

        # Distance to ʊ words
        if self.task == 'short_u':
            for pos in self.ʊ_pos:
                distance_to_task.append((np.array(pos) - np.array(self.agent_pos)))
        elif self.task == 'middle_open':
            for pos in self.ʌ_pos:
                distance_to_task.append((np.array(pos) - np.array(self.agent_pos)))
        else:
            for pos in self.uː_pos:
                distance_to_task.append((np.array(pos) - np.array(self.agent_pos)))

        o['obstacles'] = distance_to_obs
        o['dist_goal'] = np.array(self.goal_pos) - np.array(self.agent_pos)
        o['ʊ_pos'] = distance_to_task
        o['ʊ_coords'] = self.ʊ_pos
        o['agent_pos'] = self.agent_pos
        o['pho_left'] = np.array((len(self.ʊ_pos), len(self.ʌ_pos), len(self.uː_pos)))
        o['ʌ_coords'] = self.ʌ_pos
        o['u:_coords'] = self.uː_pos
        

        ob_rep, env_ob, _ = self.display()
            
        # Agent surroundings
        o['neigh'] = env_ob[self.agent_pos[0] - 1:
                            self.agent_pos[0] + 2, self.agent_pos[1] - 1:
                            self.agent_pos[1] + 2]

        return o

    def display(self):
        """
        Displays the action of the agent and the location of the words, goal and obstacles
        :return: string of the evironment, an array with agent observation (3X3) and array of
        environment to render using sns.
        """

        envir_rend = self.grid.copy()

        envir_rend[self.goal_pos] = 6

        for pos in self.ʊ_pos:
            envir_rend[pos] = 2

        for pos in self.ʌ_pos:
            envir_rend[pos] = 3

        for pos in self.uː_pos:
            envir_rend[pos] = 4

        for obs in self.obstacle_pos:
            envir_rend[obs] = 1

        env_ob = envir_rend.copy()

        envir_rend[self.agent_pos] = 5

        rend_grid = ""

        for r in range(self.size[0]):

            line = ''

            for c in range(self.size[1]):
                string_rend = self.dict_map_display[envir_rend[r, c]]

                line += '{0:2}'.format(string_rend)

            rend_grid += line + '\n'

        return rend_grid, env_ob, envir_rend

    def reset(self):
        """
        Randomly places phonemes, obstacles, goal and agent
        :return: observation of the environment
        """

        self.time_step = 0

        coord = list()

        for r in range(1, self.size[0] - 1):
            for c in range(1, self.size[1] - 1):
                coord.append((r, c))

        if self.seed:
            
            rng = np.random.default_rng(1234)
            rng.shuffle(coord)
            
        else:
            np.random.shuffle(coord)

        self.ʊ_pos = list()
        self.uː_pos = list()
        self.ʌ_pos = list()
        self.obstacle_pos = list()

        phonemes = self.num_words // 3

        for phoneme in range(phonemes):
            self.ʊ_pos.append(coord.pop())
            self.uː_pos.append(coord.pop())
            self.ʌ_pos.append(coord.pop())

        for obs in range(self.obstacles):
            self.obstacle_pos.append(coord.pop())

        self.goal_pos = coord.pop()
        
        # Agent placed randomly 
        self.agent_pos = coord[np.random.choice(range(len(coord)))]
        
        coord.remove(self.agent_pos)
        
        self.obstacle_pos = sorted(self.obstacle_pos)

        observation = self.observe()

        return observation
