#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal

np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# =============================================================================
# Critic Neural Network - Q
# =============================================================================

class QNN(nn.Module):

    def __init__(self, obs_size, num_actions, hidden):
        super(QNN, self).__init__()

        self.obs_size = obs_size
        self.hid_1 = hidden[0]
        self.hid_2 = hidden[1]
        self.hid_3 = hidden[2]
        self.num_actions = num_actions

        self.linear1 = nn.Linear(self.obs_size + self.num_actions, self.hid_1)
        self.linear2 = nn.Linear(self.hid_1, self.hid_2)
        self.linear3 = nn.Linear(self.hid_2, self.hid_3)
        self.output = nn.Linear(self.hid_3, 1)

        # init_hidden(self.linear1, self.linear2, self.linear3)
        nn.init.uniform_(self.output.weight, -0.003, 0.003)

    def forward(self, x, y):  # state and action

        x = torch.cat((x, y), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.output(x)


# =============================================================================
# Policy Network
# =============================================================================
class PolicyNN(nn.Module):

    def __init__(self, obs_size, num_actions, hidden, epsilon=1e-6, env=None, min_log=-20,
                 max_log=2):
        super(PolicyNN, self).__init__()

        self.obs_size = obs_size
        self.hid_1 = hidden[0]
        self.hid_2 = hidden[1]
        self.hid_3 = hidden[2]
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.min_log = min_log
        self.max_log = max_log
        self.dist = Normal
        self.action_space = env.action_space
        self.linear1 = nn.Linear(self.obs_size, self.hid_1)
        self.linear2 = nn.Linear(self.hid_1, self.hid_2)
        self.linear3 = nn.Linear(self.hid_2, self.hid_3)

        self.mean = nn.Linear(self.hid_3, self.num_actions)
        self.log_std = nn.Linear(self.hid_3, self.num_actions)

        # init_hidden(self.linear1, self.linear2, self.linear3)
        nn.init.uniform_(self.mean.weight, -0.003, 0.003)
        nn.init.uniform_(self.log_std.weight, -0.003, 0.003)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=self.min_log, max=self.max_log).exp()
        return mean, log_std

    def sample_action(self, obs, reparam=True):
        """
        Sampling from a normal distrbution
        """
        mean, std = self.forward(obs)
        action_pdist = self.dist(mean, std)  # for normal distribution

        # multivariate normal distribution
        if self.num_actions != 1:
            covar = torch.diag_embed(std)
            action_pdist = MultivariateNormal(mean, covariance_matrix=covar)

        # logprob action
        samples = action_pdist.rsample() if reparam else action_pdist.sample()
        mus = samples
        action = self.scale_action(torch.tanh(mus))
        if action.dim() == 1:
            action = action.unsqueeze(dim=-1)
        log_prob = (action_pdist.log_prob(mus) - torch.log(1 - action.pow(2) +
                    self.epsilon).sum(1))
        return log_prob, action

    def scale_action(self, action):
        """Scale continuous actions from tanh range"""

        low, high = torch.from_numpy(self.action_space.low), torch.from_numpy(
            self.action_space.high)
        low, high = low.to(device), high.to(device)

        return action * (high - low) / 2 + (low + high) / 2
