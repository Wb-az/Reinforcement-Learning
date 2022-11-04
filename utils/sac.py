#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 18:01:38 2021

@author: aze_ace
"""

import os
import torch
import numpy as np
import torch.optim as optim
from utils.memory import ReplayMemory
from utils.networks_architecture import QNN, PolicyNN

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SoftActorCritic:
    """
    Implement soft actor-critic - based on original implemetation of
    Soft Actor-Critic:  Off-Policy Maximum Entropy Deep Reinforcement Learning
    """

    def __init__(self, actor_kwargs):
        """
        :param actor_kwargs: a dictionary of arguments passed to the actor
            gamma (float):  discount factor for the q function
            input_dim (array): size of the observatio/state
            n_actions (array): the environment action space
            min (float): minimum value to clamp the log std
            max (float): minimum to clamp the log std
            lr_q (float): learning rate for the q and targets approximator
            lr_pol (float): learning rate for the policy
            batch_size (int) size of action samples
            capacity (integer): the capacity of the memorry
            memory (method): the buffer to store the transitions-experience
            tau (float): soft value for the target update
            env (gym environment): the environment object
            self.log_alpha: log of the temperature to be learnt
            self.alpha: temperature to be learnt
            self.target_entropy: target entropy of the temperature product of the env action space
            action_space: the agent action space
            path: a string with the path tosave the models weigths
            loss_function: objective function to minimize
            target1: target value approximator
            target2: target value approximator
            q1: critic  approximator
            q2: critic approximator
            policy: policy approximator
            policy_optim: fuction to minimize the policy approximator loss
            q1_optim: fuction to minimize the q1 critic approximator loss
            q2_optim: fuction to minimize the q2 critic approximator loss
            alpha_optim: fuction to minimize the entropy
        """
        self.gamma = actor_kwargs['gamma']
        self.input_dim = actor_kwargs['state_dim']
        self.n_actions = actor_kwargs['actions_dim']
        self.min = actor_kwargs['min']
        self.max = actor_kwargs['max']
        self.lr_q = actor_kwargs['q_lr']
        self.lr_pol = actor_kwargs['lr']
        self.hidden = actor_kwargs['hidden_size']
        self.batch_size = actor_kwargs['batch_size']
        self.capacity = actor_kwargs['memory_size']
        self.memory = ReplayMemory(self.capacity)
        self.tau = actor_kwargs['tau']
        self.env = actor_kwargs['env']
        self.log_alpha = torch.zeros(1, requires_grad=True).to(device)
        self.alpha = self.log_alpha.detach().exp()
        self.target_entropy = -np.product(self.env.action_space.shape)
        self.action_space = actor_kwargs['env'].action_space
        self.path = actor_kwargs['path']
        self.loss_fn = torch.nn.MSELoss()

        # Initialize networks
        self.policy = PolicyNN(self.input_dim, self.n_actions,
                               self.hidden, env=self.env, min_log=self.min,
                               max_log=self.max).to(device)
        self.q1 = QNN(self.input_dim, self.n_actions, self.hidden).to(device)
        self.q2 = QNN(self.input_dim, self.n_actions, self.hidden).to(device)
        self.target1 = QNN(self.input_dim, self.n_actions, self.hidden).to(device)
        self.target2 = QNN(self.input_dim, self.n_actions, self.hidden).to(device)

        # Initialize network optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr_pol)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=self.lr_q)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=self.lr_q)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr_q)

        # copy value params to target param
        self.copy()

    def act(self, state):
        """
        :param state: a numpy array tepresenting the agent observatio
        :return: a numpy array, the next action to take
        """
        state = torch.from_numpy(state).unsqueeze(dim=0)
        state = state.to(device)
        _, action = self.policy.sample_action(state, reparam=False)
        return action.cpu().squeeze(0).numpy()

    def store_memory(self, state, action, next_state, reward, done):
        """
        Store the agent experience
        """
        self.memory.push(state, action, next_state, reward, done)

    def copy(self):
        """
        Copy model weights from value (source) to target
        """
        self.target1.load_state_dict(self.q1.state_dict())
        self.target2.load_state_dict(self.q2.state_dict())

    def polyak_update(self) -> object:
        """
        Soft target network update tau * value - (1-tau) * target
        """
        for tar_param, q_param, in zip(self.target1.parameters(),
                                       self.q1.parameters()):
            tar_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * tar_param.data)
        for tar_param, q_param, in zip(self.target2.parameters(),
                                       self.q2.parameters()):
            tar_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * tar_param.data)

    @staticmethod
    def train_step(net_loss, optimizer):
        """
        training step
        """
        optimizer.zero_grad()
        net_loss.backward()
        optimizer.step()

    def train_alpha(self, alpha_loss):
        """
        alpha training step
        """
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.detach().exp()

    def calc_q_targets(self, next_states, rewards, dones):
        """
        Compute q target and the next target value
        """
        with torch.no_grad():
            next_log_probs, next_actions = self.policy.sample_action(next_states)
            next_target_q1_preds = self.q1(next_states, next_actions).view(-1)
            next_target_q2_preds = self.q2(next_states, next_actions).view(-1)
            next_target_v = torch.min(next_target_q1_preds, next_target_q2_preds)
            next_target_v = (next_target_v - self.alpha * next_log_probs)

            q_targets = rewards.view(-1) + ~dones.view(-1) * self.gamma * next_target_v

            return q_targets

    def alpha_loss(self, log_probs):
        """
        Compute the value approximator loss
        """
        alpha_loss = - (self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.train_alpha(alpha_loss)
        return alpha_loss.item()

    def q_loss(self, states, actions, q_targets):
        """
        compute the critics loss
        """
        q1_pred = self.q1(states, actions).view(-1)
        q2_pred = self.q2(states, actions).view(-1)
        q1_loss = self.loss_fn(q1_pred, q_targets)
        q2_loss = self.loss_fn(q2_pred, q_targets)
        self.train_step(q1_loss, self.q1_optim)
        self.train_step(q2_loss, self.q2_optim)
        q_loss = q1_loss.item() + q2_loss.item()
        return q_loss

    def policy_loss(self, states):
        """
        Compute the policy loss
        """
        log_prob, rep_action = self.policy.sample_action(states)
        q1_preds = self.q1(states, rep_action).view(-1)
        q2_preds = self.q2(states, rep_action).view(-1)
        q_preds = torch.min(q1_preds, q2_preds)
        pol_loss = (self.alpha * log_prob - q_preds).mean()
        self.train_step(pol_loss, self.policy_optim)
        return pol_loss.item(), log_prob

    def update_parameters(self):
        """
        update the parameter using polyak soft update
        """
        states, actions, next_states, rewards, dones = self.memory_sample()
        q_targets = self.calc_q_targets(next_states, rewards, dones)

        # q loss
        q_loss = self.q_loss(states, actions, q_targets)

        # pol loss
        pol_loss, log_probs = self.policy_loss(states)

        # alpha loss
        alpha_loss = self.alpha_loss(log_probs)
        
        # Total loss
        total_loss = q_loss + pol_loss + alpha_loss

        # print(f'Loss Q: {q_loss}, Pol: {pol_loss}, Alpha: {alpha_loss}')

        # Soft update
        self.polyak_update()

        return total_loss

    def memory_sample(self):
        """
        Extract a sample of experience and convert the experience into tensors
        """

        state_batch, action_batch, next_state_batch, reward_batch, done = \
            self.memory.sample(self.batch_size)

        states = torch.tensor(state_batch).to(device)
        next_states = torch.tensor(next_state_batch).to(device)
        actions = torch.tensor(action_batch).to(device)
        rewards = torch.tensor(np.float32(reward_batch)).unsqueeze(1).to(device)
        dones = torch.tensor(done).unsqueeze(1).to(device)

        return states, actions, next_states, rewards, dones

    def save_check_point(self, episode):
        """Save model weights to path"""

        torch.save({'episode': episode, 'model': self.policy.state_dict(),
                    'optim': self.policy_optim.state_dict()}, os.path.join(self.path, 'policy.pth'))
        torch.save({'episode': episode, 'model': self.target1.state_dict()},
                   os.path.join(self.path, 'target1.pth'))
        torch.save({'episode': episode, 'model': self.target2.state_dict()},
                   os.path.join(self.path, 'target2.pth'))
        torch.save({'episode': episode, 'model': self.q1.state_dict(),
                    'optim': self.q1_optim.state_dict()}, os.path.join(self.path, 'q1.pth'))
        torch.save({'episode': episode, 'model': self.q2.state_dict(),
                    'optim': self.q2_optim.state_dict()}, os.path.join(self.path, 'q2.pth'))

    def load_model(self):
        """
        Load the approximator weights
        """
        checkpoint_pol = torch.load(os.path.join(self.path, 'policy.pth'), map_location=device)
        checkpoint_tar1 = torch.load(os.path.join(self.path, 'target2.pth'), map_location=device)
        checkpoint_tar2 = torch.load(os.path.join(self.path, 'target2.pth'), map_location=device)
        checkpoint_q1 = torch.load(os.path.join(self.path, 'q1.pth'), map_location=device)
        checkpoint_q2 = torch.load(os.path.join(self.path, 'q2.pth'), map_location=device)

        self.policy.load_state_dict(checkpoint_pol['model'], strict=False)
        self.target1.load_state_dict(checkpoint_tar1['model'], strict=False)
        self.target2.load_state_dict(checkpoint_tar2['model'], strict=False)
        self.q1.load_state_dict(checkpoint_q1['model'], strict=False)
        self.q2.load_state_dict(checkpoint_q2['model'], strict=False)
        print('...........Models loaded............')