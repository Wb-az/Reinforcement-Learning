
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import namedtuple, defaultdict

seed = 123
rng = np.random.default_rng(seed)


def plot_episodes_stats(stats, episodes=None, smoothing_window=10, hideplot=False,
                       env_dim=None):
    """
    :param stats: a namedtuple containing the stats
    :param episodes: number of episodes run by the agent
    :param smoothing_window: intiger, number of observations per eavh window
    :param hideplot: boolean to display the plots
    :param env_dim: string with the environment dimensions
    :return: plots
    Note: This code was adapted from Microsoft, Introduction to Reinforcement Learning.
    """

    figs_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(figs_dir, exist_ok=True)
    size = (7, 4)

    # Plot the episode length over time
    fig1 = plt.figure(figsize=size)
    x = np.arange(1, episodes + 1)
    steps = pd.Series(stats.length_episodes).rolling(smoothing_window,
                                                     min_periods=smoothing_window).mean()
    plt.plot(x, steps, color='#0000B3')
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length")
    plt.savefig(os.path.join(figs_dir, 'ql_steps_{}_{}.png'.format(episodes, env_dim)))
    if hideplot:
        plt.close()
    else:
        plt.rcParams.update({'font.size': 10})
        plt.show()

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=size)
    rewards_smoothed = pd.Series(stats.reward_episodes).rolling(smoothing_window,
                                                                min_periods=smoothing_window).mean()
    plt.plot(x, rewards_smoothed, color='#0000B3')
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards per episode (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(os.path.join(figs_dir, 'ql_reward_{}_{}.png'.format(episodes, env_dim)))
    if hideplot:
        plt.close(fig2)
    else:
        plt.rcParams.update({'font.size': 10})
    plt.show(fig2)

    # Plot the episode mean reward per episode
    fig3 = plt.figure(figsize=size)
    mean_smoothed = pd.Series(stats.episode_mean_reward). \
        rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(x, mean_smoothed, color='#0000B3')
    plt.fill_between(x, mean_smoothed - stats.episode_std / 2,
                     mean_smoothed + stats.episode_std / 2,
                     color='#0000B3', alpha=0.2)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward per Episode and std (Smoothed over window size {})'.format(
        smoothing_window))
    plt.savefig(
        os.path.join(figs_dir, 'ql_average_{}_{}.png'.format(episodes, env_dim)))
    if hideplot:
        plt.close(fig3)
    else:
        plt.rcParams.update({'font.size': 10})
    plt.show(fig3)
    
    # Plot the words collected per episode
    fig4 = plt.figure(figsize=size)
    collection = pd.Series(stats.words_collected). \
        rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(x, collection, color='#0000B3')
    plt.xlabel("Episode")
    plt.ylabel("Number of words")
    plt.title("Word with phonetic ʊ sound collected (Smoothed over window size {})".format(
        smoothing_window))
    plt.savefig(
        os.path.join(figs_dir, 'ql_words_{}_{}.png'.format(episodes, env_dim)))
    if hideplot:
        plt.close(fig3)
    else:
        plt.rcParams.update({'font.size': 10})
    plt.show(fig3)

    return fig1, fig2, fig3


def rend_sns(env_array):
    """
    Convert a numpy array to a sns heat map
    :param env_array: an array representing the evironment/grid
    :return: a heat map with of array
    """
    
    fig,ax = plt.subplots(1, figsize=(6,4))

    # Colors for each of the unique items on the grid for the heatmap
    cmap = ['#ffffd9', '#202603', '#c2e699', '#7fcdbb', '#1d91c0', '#2ac01d', '#f1dc18',
            '#041f61']
    items = len(np.unique(env_array))
    sns.heatmap(env_array, linewidth=0.5, cmap=ListedColormap(cmap), ax=ax)
    colorbar = ax.collections[0].colorbar
    m = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks(
        [colorbar.vmin + 0.5 * m/ items + m * i / items for i in range(items)])
    colorbar.set_ticklabels(['empty', 'obstacle', 'ʊ', 'ʌ', 'u :', 'agent', 'goal', 'wall'])
    plt.show()

