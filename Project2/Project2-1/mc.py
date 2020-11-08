#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import defaultdict
#-------------------------------------------------------------------------
'''
    Monte-Carlo
    In this problem, you will implememnt an AI player for Blackjack.
    The main goal of this problem is to get familar with Monte-Carlo algorithm.
    You could test the correctness of your code 
    by typing 'nosetests -v mc_test.py' in the terminal.
    
    You don't have to follow the comments to write your code. They are provided
    as hints in case you need. 
'''
#-------------------------------------------------------------------------

def initial_policy(observation):
    """A policy that sticks if the player score is >= 20 and his otherwise
    
    Parameters:
    -----------
    observation:
    Returns:
    --------
    action: 0 or 1
        0: STICK
        1: HIT
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    # get parameters from observation
    score = observation[0]
    # action
    action = 0 if score >= 20 else 1
    ############################
    return action

def mc_prediction(policy, env, n_episodes, gamma = 1.0):
    """Given policy using sampling to calculate the value function 
        by using Monte Carlo first visit algorithm.
    
    Parameters:
    -----------
    policy: function
        A function that maps an obversation to action probabilities
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    Returns:
    --------
    V: defaultdict(float)
        A dictionary that maps from state to value
    """
    # initialize empty dictionaries
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> value
    V = defaultdict(float)
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    # loop each episode
    for i_episode in range(n_episodes):
        # initialize the episode
        observation = env.reset()
        # generate empty episode list
        episodes = []
        # loop until episode generation is done
        done = False
        while not done:
            # select an action
            action = policy(observation)
            # return a reward and new state
            next_observation, reward, done, _ = env.step(action)
            # append state, action, reward to episode
            episodes.append((observation, action, reward))
            # update state to new state
            observation = next_observation      
            
        observations = set([x[0] for x in episodes])
        # loop for each step of episode, t = T-1, T-2,...,0
        for i, observation in enumerate(observations):
            # compute G
            idx = episodes.index([episode for episode in episodes if episode[0] == observation][0])
            Q = sum([episode[2] * gamma ** i for episode in episodes[idx:]])
            # unless state_t appears in states
                # update return_count
            returns_count[observation] += 1.0    
                # update return_sum            
            returns_sum[observation] += Q
                # calculate average return for this state over all sampled episodes
            V[observation] = returns_sum[observation] / returns_count[observation]



    ############################
    
    return V

def epsilon_greedy(Q, state, nA, epsilon = 0.1):
    """Selects epsilon-greedy action for supplied state.
    
    Parameters:
    -----------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a. 
    state: int
        current state
    nA: int
        Number of actions in the environment
    epsilon: float
        The probability to select a random action, range between 0 and 1
    
    Returns:
    --------
    action: int
        action based current state
    Hints:
    ------
    With probability (1 − epsilon) choose the greedy action.
    With probability epsilon choose an action at random.
    """
    ############################
    # YOUR IMPLEMENTATION HERE #
    if np.random.random() < epsilon:
        action = np.random.randint(nA)        
    else:
        action = np.argmax(Q[state])
    ############################
    return action

def mc_control_epsilon_greedy(env, n_episodes, gamma = 1.0, epsilon = 0.1):
    """Monte Carlo control with exploring starts. 
        Find an optimal epsilon-greedy policy.
    
    Parameters:
    -----------
    env: function
        OpenAI gym environment
    n_episodes: int
        Number of episodes to sample
    gamma: float
        Gamma discount factor
    epsilon: float
        The probability to select a random action, range between 0 and 1
    Returns:
    --------
    Q: dict()
        A dictionary  that maps from state -> action-values,
        where Q[s][a] is the estimated action value corresponding to state s and action a.
    Hint:
    -----
    You could consider decaying epsilon, i.e. epsilon = epsilon-(0.1/n_episodes) during each episode
    and episode must > 0.    
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    # a nested dictionary that maps state -> (action -> action-value)
    # e.g. Q[state] = np.darrary(nA)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    for i_episode in range(n_episodes):
        # define decaying epsilon
        epsilon = 0.99*epsilon
        # initialize the episode
        observation = env.reset()
        # generate empty episode list
        episodes = []
        # loop until one episode generation is done
        done = False
        while not done:
            # get an action from epsilon greedy policy
            action = epsilon_greedy(Q, observation, env.action_space.n, epsilon)
            # return a reward and new state
            next_observation, reward, done, _ = env.step(action)
            # append state, action, reward to episode
            episodes.append((observation, action, reward))
            # update state to new state
            observation = next_observation       
        # loop for each step of episode, t = T-1, T-2, ...,0
        pairs = set([(episode[0], episode[1]) for episode in episodes])
        for (observation, action) in pairs:
            # compute G
            pair = (observation, action)
            idx = episodes.index([episode for episode in episodes if episode[0] == observation and episode[1] == action][0])
            V = sum([reward[2] * gamma ** i for i, reward in enumerate(episodes[idx:])])
                # update return_count
            returns_count[pair] += 1.
                # update return_sum
            returns_sum[pair] += V
                # calculate average return for this state over all sampled episodes
            Q[observation][action] = returns_sum[pair] / returns_count[pair]
    return Q
