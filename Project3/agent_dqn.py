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
import time
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
        # import arguments
        self.args = args
        self.env = env
        self.batch_size = self.args.batch_size
        self.gamma = self.args.gamma
        self.lr = self.args.learning_rate
        self.memory_cap = self.args.memory_cap
        self.n_episode = self.args.n_episode
        self.n_step = self.args.n_step
        self.update_f = self.args.update_f
        self.explore_step = self.args.explore_step        
        self.action_size = self.args.action_size
        self.algorithm = self.args.algorithm
        self.save_path = "dqn/"
        print('using algorithm ', self.algorithm)
        
        # whether continue training
        self.load_model = self.args.load_model        
        
        # unify tensor tpye according to device names
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print('using device ',  torch.cuda.get_device_name(0))
        self.FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.use_cuda else torch.LongTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_cuda else torch.ByteTensor
        self.Tensor = self.FloatTensor # default type

        # epsilon decay
        self.epsilon = 1.0
        self.epsilon_min = 0.025
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step

        # Create the policy net and the target net
        self.policy_net = DQN()
        self.policy_net.to(self.device)
        if self.algorithm == 'DDQN':
            self.policy_net_2 = DQN()
            self.policy_net_2.to(self.device)
        self.target_net = DQN()
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # replay buffer
        self.memory = []
        
        # optimizer
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.lr)
        if self.algorithm == 'DDQN':
            self.optimizer_2 = optim.Adam(params=self.policy_net_2.parameters(), lr=self.lr)
        
        # other
        self.f_skip = 4 # frame skip
        self.n_avg_reward = 100
        self.f_print = 100
        self.print_test = False
        
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
        state = self.env.reset()/255.
        self.last_life = 5
        self.step = 0
        done = False
        total_reward = 0
        ###########################
        return state, done, total_reward
    
    
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
            self.epsilon = self.epsilon_min
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
    
    def push(self, state, action, reward, next_state, dead, done):
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
        # initialize
        self.steps_done = 0
        self.steps = []
        self.rewards = []
        self.mean_rewards = []
        self.best_reward = 0
        self.last_saved_reward = 0

        start = time.time()
        logfile = open('dqn.log', 'w+')
        # continue training
        if self.load_model:
            self.policy_net.load_state_dict(torch.load(self.save_path+'model.pth', map_location=self.device))           
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = self.epsilon_min
            
        for episode in range(self.n_episode):
            state, done, total_reward = self.init_game_setting()            
            while (not done) and self.step < 10000:
                # move to next state
                self.step += 1
                self.steps_done += 1
                action = self.make_action(state)
                next_state,reward,done,life = self.env.step(action)
                # lives matter
                now_life = life['ale.lives']
                dead = (now_life < self.last_life)
                self.last_life = now_life
                next_state = next_state/255.
                # Store the transition in memory
                self.push(state, action, reward, next_state, dead, done)
                state = next_state
                total_reward += reward 
                
                if len(self.memory) >= self.n_step and self.steps_done % self.f_skip == 0:
                    if self.algorithm == 'DQN':
                        self.optimize_DQN()
                    elif self.algorithm == 'DDQN':
                        self.optimize_DDQN()                    
                if self.steps_done % self.update_f == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())                

            self.rewards.append(total_reward)
            self.mean_reward = np.mean(self.rewards[-self.n_avg_reward:])
            self.mean_rewards.append(self.mean_reward)
            self.steps.append(self.step)
            # print progress in terminal
            progress = "Episode: " + str(episode) + ",\tCurrent mean reward: "+ "{:.2f}".format(self.mean_reward) + ',\tBest mean reward: ' + "{:.2f}".format(self.best_reward)
            progress += ",\tCurerent Reward: " + str(total_reward)  +",\tTime: " + time.strftime('%H:%M:%S', time.gmtime(time.time() - start))            
            print(progress)
            print(episode, self.mean_reward, self.best_reward, total_reward, time.time() - start, file=logfile)
            logfile.flush()
            if (episode+1) % self.f_print == 0:                
                self.plots()
            # save the best model
            if self.mean_reward > self.best_reward and self.steps_done > self.n_step:                
                checkpoint_path = self.save_path + 'model.pth'
                torch.save(self.policy_net.state_dict(), checkpoint_path)
                self.last_saved_reward = self.mean_reward
                self.best_reward = max(self.mean_reward, self.best_reward)
        ###########################

    def optimize_DQN(self):
        # sample
        self.replay_buffer()
        state, action, reward, next_state, dead, done = zip(*self.mini_batch)

        state = self.Tensor(np.float32(state)).permute(0,3,1,2).to(self.device)
        action = self.LongTensor(action).to(self.device)
        reward = self.Tensor(reward).to(self.device)
        next_state = self.Tensor(np.float32(next_state)).permute(0,3,1,2).to(self.device)
        dead = self.Tensor(dead).to(self.device)
        done = self.Tensor(done).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state).gather(1,action.unsqueeze(1)).squeeze(1)
        # Compute next Q, including the mask
        next_state_values = self.target_net(next_state).detach().max(1)[0]
        # Compute the expected Q value. stop update if done
        expected_state_action_values = reward + (next_state_values * self.gamma)*(1-done)
        # Compute Huber loss
        self.loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.data)
        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return

    def optimize_DDQN(self):
        # sample
        self.replay_buffer()
        state, action, reward, next_state, dead, done = zip(*self.mini_batch)

        # transfer 1*84*84*4 to 1*4*84*84, which is 0,3,1,2
        state = self.Tensor(np.float32(state)).permute(0,3,1,2).to(self.device)
        action = self.LongTensor(action).to(self.device)
        reward = self.Tensor(reward).to(self.device)
        next_state = self.Tensor(np.float32(next_state)).permute(0,3,1,2).to(self.device)
        dead = self.Tensor(dead).to(self.device)
        done = self.Tensor(done).to(self.device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state).gather(1,action.unsqueeze(1)).squeeze(1)
        state_action_values_2 = self.policy_net_2(state).gather(1,action.unsqueeze(1)).squeeze(1)
        # Compute next Q, including the mask
        next_state_values = self.target_net(next_state).detach().max(1)[0]
        next_state_values_2 = self.target_net(next_state).detach().max(1)[0]
        next_state_values = torch.min(next_state_values, next_state_values_2)
        # Compute the expected Q value. stop update if done
        expected_state_action_values = reward + (next_state_values * self.gamma)*(1-done)
        # Compute Huber loss
        self.loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.data)
        self.loss_2 = F.smooth_l1_loss(state_action_values_2, expected_state_action_values.data)
        # Optimize the model
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.optimizer_2.zero_grad()
        self.loss_2.backward()        
        self.optimizer_2.step()
        return

    def plots(self):
        fig1 = plt.figure(1)
        plt.clf()
        plt.title('Training_Steps_per_Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.plot(self.steps)
        fig1.savefig(self.save_path+'steps.png')

        fig2 = plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.rewards)
        
        if len(self.rewards) >= self.n_avg_reward:
            plt.plot(self.mean_rewards)
        fig2.savefig(self.save_path+'rewards.png')
        
        rewards = np.array(self.rewards)
        np.save(self.save_path+'rewards.npy', rewards)