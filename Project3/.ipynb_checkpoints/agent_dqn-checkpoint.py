#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""
from matplotlib import pyplot as plt

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.env = env
        self.args = args
        self.gamma = self.args.gamma
        self.batch_size = self.args.batch_size        
        self.memory_cap = self.args.memory_cap
        self.n_episode = self.args.n_episode
        self.lr = self.args.learning_rate     
        
        self.epsilon = self.args.epsilon
        self.epsilon_decay_window = self.args.epsilon_decay_window   
        self.epsilon_min = self.args.epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_window
        

        self.n_step = self.args.n_step
        self.f_update = self.args.f_update        
        self.load_model = self.args.load_model
        self.action_size = self.args.action_size        
#         self.algorithm = self.args.algorithm   
    
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print('using device ', torch.cuda.get_device_name(0))
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        self.Tensor = self.FloatTensor
        
        # Create the policy net and the target net
        self.policy_net = DQN()
        self.policy_net.to(self.device)
#         if self.algorithm == 'DDQN':
#             self.policy_net_2 = DQN()
#             self.policy_net_2.to(self.device)
        self.target_net = DQN()
        self.target_net.to(self.device)
        self.policy_net.train()        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)
        # buffer        
        self.memory = []
        
        ##
        self.mean_window = 100
        self.print_frequency = 100
        
        
        
        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            self.policy_net.load_state_dict(torch.load('model.pth', map_location=self.device))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.algorithm == 'DDQN':
                self.policy_net_2.load_state_dict(torch.load('model.pth', map_location=self.device))
            self.print_test = True

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass
    
    
    def make_action(self, observation, test=False):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if test:
            self.epsilon = self.epsilon_min*0.5
            observation = observation/255.
        else:
            self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_min)
        if random.random() > self.epsilon:
            observation = self.Tensor(observation.reshape((1,84,84,4))).transpose(1,3).transpose(2,3)
            state_action_value = self.policy_net(observation).data.cpu().numpy()
            action = np.argmax(state_action_value)
        else:
            action = random.randint(0, self.action_size-1)
        ###########################
        return action
    
    def push(self,state, action, reward, next_state, dead, done):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        if len(self.memory) >= self.memory_cap:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, dead, done))
        ###########################
        
        
    def replay_buffer(self):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.mini_batch = random.sample(self.memory, self.batch_size)       
        ###########################
        return 
        

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
#         self.steps_done = 0
        self.steps = []
        self.rewards = []
        self.mean_rewards = []
        self.best_reward = 0
        self.last_saved_reward = 0
        
#         # continue training from where it stopped
#         if self.load_model:
#             self.policy_net.load_state_dict(torch.load('model.pth', map_location=self.device))
#             self.target_net.load_state_dict(self.policy_net.state_dict())
#             self.epsilon = self.epsilon_min
        
        for episode in range(self.n_episode):
            # Initialize the environment and state
            state = self.env.reset()/255.
            self.last_life = 5
            total_reward = 0
            self.step = 0
            done = False
            
            while (not done) and self.step < 10000:
                # move to next state
                self.step += 1
#                 self.steps_done += 1
                action = self.make_action(state)
                next_state,reward,done,life = self.env.step(action)                
                # lives matter
                self.now_life = life['ale.lives']
                dead = self.now_life < self.last_life
                self.last_life = self.now_life
                next_state = next_state/255.
                # Store the transition in memory
                self.push(state, action, reward, next_state, dead, done)
                state = next_state
                total_reward += reward               
            
                if done:
                    self.rewards.append(total_reward)
                    self.mean_reward = np.mean(self.rewards[-self.mean_window:])
                    self.mean_rewards.append(self.mean_reward)
                    self.steps.append(self.step)
                    # print the process to terminal
                    if episode % self.print_frequency == 0:
                        print("="*50)
                        print('Current episode = ', episode)
                        self.print_and_plot()                        
                    # save the best model
                    if self.mean_reward > self.best_reward:
                        print('-------<Model updated with best reward = ', self.mean_reward,'>-------')
                        checkpoint_path = 'DQN_Module/model.pth'
                        torch.save(self.policy_net.state_dict(), checkpoint_path)
#                         self.last_saved_reward = self.mean_reward
                        self.best_reward = self.mean_reward
                    
                        
#             if total_reward > 0:
#                 #print
#                 string = 'Ep:' + str(episode) + ',\t total steps: '+str(self.steps_done)
#                 string += ',\t current steps: '+str(self.step) + ',\t total_reward: ' +str(total_reward)                
#                 print(string)

        ###########################
    
    
    
    def print_and_plot(self):        
#         print('Current steps = ', self.steps_done)
        print('Current epsilon = ', self.epsilon)                       
        print('Current mean reward = ', self.mean_reward)
        print('Current memory allocated = ', len(self.memory))
        print('Best mean reward = ', self.best_reward)
        fig1 = plt.figure(1)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.plot(self.steps)
        fig1.savefig('DQN_Module/steps.png')

        fig2 = plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.mean_rewards)
        # Take episode averages and plot them too        
        fig2.savefig('DQN_Module/rewards.png')
        