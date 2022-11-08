#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:23:19 2021

@author: aze_ace
"""


import os
import gym
import numpy as np
import torch
from utils.sac import SoftActorCritic
from utils.plotting import Stats, plot_episode_stats
from gym.wrappers.monitoring.video_recorder import VideoRecorder

np.random.seed(123)
torch.manual_seed(123)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
number_of_cpus = torch.multiprocessing.cpu_count()

print('Available cores', number_of_cpus)
# lunar_lander = gym.make('LunarLander-v2', continuous=True, render_mode='human')
# bipedal = gym.make('BipedalWalker-v3', render_mode='human')

lunar_lander = gym.make('LunarLander-v2', continuous=True, render_mode='rgb_array')
bipedal = gym.make('BipedalWalker-v3', render_mode='rgb_array')
envs = [lunar_lander, bipedal]

# Create directory for weights and videos
paths = list()
params_dirs = list()
for e in ['lunar', 'bipedal']:
    path = os.path.join(os.getcwd(), 'videos', e)
    path_par = os.path.join(os.getcwd(), 'params', e)
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_par, exist_ok=True)
    paths.append(path)
    params_dirs.append(path_par)

env = envs[1]
agent_params = {'gamma': 0.99, 'state_dim': env.observation_space.shape[0],
                'actions_dim': env.action_space.shape[0], 'min': -20, 'max': 2, 'q_lr': 0.0001,
                'lr': 0.0001, 'hidden_size': [256, 256, 256], 'batch_size': 256,
                'memory_size': 1000000, 'tau': 0.02, 'env': env, 'path': params_dirs[1]}

episodes = 600
max_steps = 1600
smooth = 10
evaluate = True
path_ = paths[1]
plots_names = ['lunar', 'bipedal']
plot_name =plots_names[1]


def warmup(transitions=10000, sac_agent=None, environment=None):
    """
    :param transitions: number of experiences to store in memory before training
    :param sac_agent: soft actor-critic algorithm
    :param environment: gym environment to train the agent policy
    :return: print a completion statement
    """
    obs, _ = environment.reset()
    for transition in range(transitions):

        w_action = environment.action_space.sample()
        # Next state after taking an action
        next_obs, w_reward, terminated, w_truncated, _ = environment.step(w_action)
        # store transition
        sac_agent.store_memory(next_obs, w_action, obs, w_reward, terminated)
        obs = next_obs
        if terminated or w_truncated:
            obs, _ = environment.reset()

    print('Warmup completed, total transitions stored: {}'.format(transitions))


def evaluate_sac(agent_policy, gym_env, eps=50, video_path=None):
    """
    :param agent_policy: trained agent algorithm
    :param gym_env: a gym environment
    :param eps: integer with the number of episodes to train for
    :param video_path: path to save videos
    """
    agent_policy.eval()

    for e in range(eps):

        state, _ = gym_env.reset()
        terminated = False
        ep_reward = 0
        steps_ = 0

        ep_path = video_path + '_ep_{}'.format(e + 1)
        video = VideoRecorder(gym_env, enabled=True, metadata={'step_id': e + 1},
                              base_path=ep_path)

        while not terminated:
            video.capture_frame()
            state = torch.from_numpy(state).unsqueeze(dim=0).to(device)
            _, actions = agent_policy.sample_action(state, reparam=False)
            actions = actions.cpu().squeeze().numpy()
            next_state, reward_, terminated, _, _ = gym_env.step(actions)
            ep_reward += reward_
            steps_ += 1
            state = next_state
        video.close()
        video.enabled = False
        gym_env.close()

        print('Episode: {} | Steps: {} | Reward: {:.2f} '.format(e + 1, steps_, ep_reward))


if __name__ == '__main__':

    # Build agent
    agent = SoftActorCritic(agent_params)

    # Logging stats
    rewards = np.zeros(episodes)
    loss_log = np.zeros(episodes)
    steps_log = np.zeros(episodes)
    returns_list = list()

    # Warmup
    warmup(sac_agent=agent, environment=env)

    best_return = env.reward_range[0]

    print('........Start training........')

    observation, _ = env.reset()
    for ep in range(episodes):

        returns = 0
        total_loss = 0
        steps = 0
        done = False

        while not done:

            action = agent.act(observation)
            next_observation, reward, done, truncated, _ = env.step(action)

            # store transition
            agent.store_memory(observation, action, next_observation, reward, done)
            returns += reward
            steps += 1

            loss = agent.update_parameters()
            total_loss += loss
            returns_list.append(reward)

            if steps == max_steps:
                break

            observation = next_observation

        observation, _ = env.reset(seed=123)
        loss_log[ep] += total_loss
        steps_log[ep] += steps
        rewards[ep] += returns

        avg_return = np.mean(rewards[-20:])
        if avg_return > best_return:
            best_return = avg_return
            agent.save_check_point(ep)

        print('Episode: {}/{} | Steps: {} | Reward: {:.2f} | Loss: {:.3f} | '
              'Cumulative steps: {}'.format(ep + 1, episodes, steps, rewards[ep], loss_log[ep] /
                                            steps, int(np.sum(steps_log))))

    print('Training completed, total steps: {}'.format(int(np.sum(steps_log))))
    stats = Stats(length_episodes=steps_log, reward_episodes=rewards, episode_loss=loss_log)

    plot_episode_stats(stats, agent.lr_pol, episodes=episodes, smoothing_window=smooth,
                       hideplot=False, name=plot_name)

    if evaluate:
        evaluate_sac(agent.policy, env, eps=10, video_path=path_)
