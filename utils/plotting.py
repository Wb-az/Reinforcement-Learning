import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple

Stats = namedtuple('Stats', ['length_episodes', 'reward_episodes', 'episode_loss'])
plt.style.use('ggplot')

figs_dir = os.path.join(os.getcwd(), 'plots')
os.makedirs(figs_dir, exist_ok=True)


def plot_episode_stats(stats, lr, episodes=None, smoothing_window=None, hideplot=False,
                       name=None, plot_dir=figs_dir):
    """
    :param plot_dir:
    :param stats:  a namedtuple with rthe stats
    :param lr: float training learning rate
    :param episodes: an intiger with number of training episodes
    :param smoothing_window: an integer to average the reward every x episodes
    :param hideplot: a boolean to display or nor the displays the stats plots
    :param name: a string to save the environment
    :param plot_dir: a string to save the plots
    :return: plots of the stats
    """
    size = (8, 4)

    plots = os.path.join(plot_dir, name)
    os.makedirs(plots, exist_ok=True)
    # Plot the episode length over time
    y_ep = 'Episodes Length'
    show_plot(size, stats.length_episodes, smoothing_window=smoothing_window, y=y_ep, lr=lr,
              episodes=episodes, hideplot=hideplot, plot_dir=plots)

    # Plot the episode reward over time
    y_r = 'Sum of rewards per episode'
    show_plot(size, stats.reward_episodes, smoothing_window=smoothing_window, y=y_r, lr=lr,
              episodes=episodes, hideplot=hideplot, plot_dir=plots)

    # Plot loss over time
    y_loss = 'Loss'
    show_plot(size, stats.episode_loss, smoothing_window=smoothing_window, y=y_loss, lr=lr,
              episodes=episodes, hideplot=hideplot, plot_dir=plots)


def show_plot(size, log_metric, smoothing_window=1, y='steps', lr=None, episodes=None,
              hideplot=True, plot_dir=None):
    # fig = plt.figure(figsize=size)
    plt.figure(figsize=size)
    x = np.arange(1, episodes + 1)
    smoothed_log = pd.Series(log_metric).rolling(smoothing_window,
                                                 min_periods=smoothing_window).mean()
    std_log = pd.Series(log_metric).rolling(smoothing_window,
                                            min_periods=smoothing_window).std()
    plt.plot(x, smoothed_log, color='#0000B3')
    plt.fill_between(x, np.array(smoothed_log) - np.array(std_log) / 2,
                     smoothed_log + np.array(std_log) / 2,
                     color='#0000B3', alpha=0.2)
    plt.xlabel('Episodes')
    plt.ylabel(y)
    plt.savefig(
        os.path.join(plot_dir, '{}_{}_{}.png'.format(y.lower(), episodes, lr)))
    if hideplot:
        plt.close()
    else:
        plt.rcParams.update({'font.size': 10})
        plt.show()
