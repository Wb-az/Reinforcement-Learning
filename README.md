

# Reinforcement Learning and Deep Reinforcement Learning


**<h2><div>1. Introduction</h2></div>** 

This repository contains three reinforcement learning tasks. 
1. A customised phonetic environment and arbitrary policies
2. Build a Q-learning agent that trains on the phone environment
3. Advance implementation of Deep Reinforcement Learning Soft Actor-Critic to train gym 
   continuos environments: LunarLander-v2 and BipedalWalker-v3

The scope and results of each task is sumarised below.

**<h3><div>2. Task 1</h3></div>**

For this task a a custom phonetic environment was build. The agent's mission is to identify words with one of 
the phonetic sounds of the IPA English alphabet:  **ʊ**, **ʌ**, **uː** whilst avoiding hitting 
environment boundaries and movable obstacles. The agent was trained with three arbitrary policies: random, 
biased, and combination. This task illustrates the learning process and the impact of environment design, including the size of the grid,
reward, the number of phonemes in the grid and moving obstacles in the agent's performance 
(Figure 1)

**<h4><div> 2.1 Environment</div></h4>**

The phoneme environment is a configurable N x M array of integers representing objects.
All objects except the wall are placed randomly in the environment. Each object is represented as follows:

- 0 : empty cell
- 1 : moving obstacle
- 2 : 'ʊ' word
- 3 : 'ʌ' word
- 4 : 'u:' word
- 5 : Agent
- 6 : Goal
- 7 : Boundaries/walls

The words are randomly from the phoneme list. The grid can be adapted to collect the three 
sounds or any of their combinations by a minimal change in the rewards and policies functions. For a more advanced task each word with the same sound can be encoded with its own number. In this work the mission is to collect/learn the phonetic sound 'ʊ'.

- The available area to placed objects is  total grid area - the boundary area

a = M x N - 2 x (M + N) - 4

- The total number of words on the grid and moveable are given by floor division of the empty 
  cells (refer to noetbook in the associted documents for the full description)

  
- There is only one goal (G) and 1 learner (A)

**<h4><div> 2.2 Actions</div></h4>**

The actions available at each time step are:
- up
- down
- left 
- right
- grab 

After taking an action, the agent gets a reward and transitions to a new state. Then the environment sends a signal indicating whether the game is over or not. 

**<h4><div> 2.3 Observations</div></h4>**

The observation of the environment is a dictionary that containing
- relative coordinates to all words in the grid
- relative coordinates to the goal 
- relative coordinates to the obstacles
- a neighbourhood 3x3 array with the encoded values 
- a counter indicating the words left
- relative distance to the obstacles
- current location of the agent


**<h4><div> 2.4 Policies</div></h4>**
- Goal oriented "Biased policy" - only grabs sounds when at the same position of the sound to a 
  defined sound and 
  searches for the Goal.
- Random policy - takes actions at random if not sound at the same location.
- Combined policy - with  p = epsilon explores, otherwise follows the biased policy.


**<h4><div>2.5 Rewards</div></h4>**

- -1 per each time step
- -20 for hitting a moving obstacle 
- -10 for grabbing in an empty cell or hitting a wasll
- -10 for grabbing a word with 'ʊ' sound
- -20 for grabbing ʌ_pos and uː
- 100 if grabbing the correct sound

-  reaching the goal if all ʊ were collected  area  x phonemes collected
-  reaching the goal and ʊ left area x (total phonemes - phonemes conected)

#### Associated file
1. ```t1_phoneme_environment.ipynb``` - the Jupyter notebook of with class environment, policies, comparison and visualisation of the stats.


<!-- <figure>
	<img src="results/env_7x7.png" alt="Phonetic
Environment" height="230">(A) <img src="results/env_comparison_50_ep.png"
alt="Environment Comparison"
height="250">(B)
<figcaption  >  Figure 1. A Configurable phonetic environment size 7 x 7.  B.
Policies comparison at different environment configurations after 50 epochs
training.
</figcaption>
</figure> -->

#### Associated file
1. ```t1_phoneme_environment.ipynb``` - the Jupyter notebook of with class environment, policies, comparison and visualisation of the stats


### Task 2

In this task, the agent follows the Q-learning algorithm (off-policy algorithm) for the learning 
process. The reward 
per 
episode remarkedly improves in comparison with task 1. The effect of the enviromment size, epsilon and alphas on the learning process was also compared.


#### Associated files
1. ```phonemes.py``` – the class environment
2. ```plotting.py``` - a function to visualise the statistics from training
3. ```t2_qlearning.ipynb``` - the Jupyter notebook of the  q-learning implementation.



### Task 3 

The advanced algorithm, Soft Actor-Critic (SAC), combined policy and value-based methods in this task. The agent learns the Policy and the Value function. Two gym continuous environments experiments were used in this task:
- LunarLander-v2
- BipedalWalker-v3

#### Associated files (six)
1. ```utils``` – this folder contains four .py files:
-	```networks_architecture.py``` which contains policy, value function and critic approximators
-	```memory.py``` - a method to save the agent transitions
-	```sac.py``` - the implementation of the SAC algorithm
-	```plotting``` - a function to visualise the statistics from training

2. ```main.py``` - trains and evaluate the performance of the agent 
3. ```t3_sac_main.ipynb``` - the Jupyter notebook version of the main, designed to run on Google collab GPUs.
