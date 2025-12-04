import os
import random
import numpy as np
import gymnasium as gym

from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib.pyplot as plt 

### State Action Reward Store
class ReplayBuffer():
	def __init__(self, buffer_limit, DEVICE):
		self.buffer = deque(maxlen=buffer_limit)
		self.dev = DEVICE

	def put(self, transition):
		self.buffer.append(transition)

	def size(self):
		# print("SIZE is ", self.buffer)
		return len(self.buffer)

	#### RETURING State, Action Reward in Tensor	
	def sample(self, n):
		mini_batch = random.sample(self.buffer, n)
		s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

		for transition in mini_batch:
			s, a, r, s_prime, done = transition
			s_lst.append(s)
			a_lst.append(a)
			r_lst.append([r])
			s_prime_lst.append(s_prime)
			done_mask = 0.0 if done else 1.0
			done_mask_lst.append([done_mask])

		s_batch = torch.tensor(s_lst, dtype=torch.float).to(self.dev)
		a_batch = torch.tensor(a_lst, dtype=torch.float).to(self.dev)
		r_batch = torch.tensor(r_lst, dtype=torch.float).to(self.dev)
		s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float).to(self.dev)
		done_batch = torch.tensor(done_mask_lst, dtype=torch.float).to(self.dev)

		# r_batch = (r_batch - r_batch.mean()) / (r_batch.std() + 1e-7)

		return s_batch, a_batch, r_batch, s_prime_batch, done_batch


### Actor Value Neural Network
class PolicyNetwork(nn.Module):
	def __init__(self, state_dim, action_dim, actor_lr):
		super(PolicyNetwork, self).__init__()

		self.fc_1 = nn.Linear(state_dim, 64)	  ### Input Layer
		self.fc_2 = nn.Linear(64, 64)			  ### Hidden Layer	
		self.fc_mu = nn.Linear(64, action_dim)	  ### Output Layer Mu Func	
		self.fc_std = nn.Linear(64, action_dim)   ### Output Layer STD Func

		self.LOG_STD_MIN = -20
		self.LOG_STD_MAX = 2

		self.max_action = 2
		self.min_action = 0
		self.action_choices = [0, 1, 2]
		self.action_scale = (self.max_action - self.min_action) / 2.0
		self.action_bias = (self.max_action + self.min_action) / 2.0

		self.lr = actor_lr
		self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

	def forward(self, state):
		x = F.leaky_relu(self.fc_1(state))   	   ### Input Layer
		x = F.leaky_relu(self.fc_2(x))             ### Hidden Layer
		mu = self.fc_mu(x)						   ### MU Output Layer	
		log_std = self.fc_std(x)                   ### STD Output Layer
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
		return mu, log_std

	def sample(self, state):
		mean, log_std = self.forward(state)
		std = torch.exp(log_std)
		reparameter = Normal(mean, std)

		x_t = reparameter.rsample()
		y_t = torch.tanh(x_t)                      ### Tangent of Position ??

		### Choose integer number 0, 1, 2 close to the float number
		float_action = self.action_scale * y_t + self.action_bias
		print("ACTION TENSOR ", float_action)

		action = min(self.action_choices, key=lambda c: abs(c - float(float_action)))
		# action = torch.abs(
		# 	float_action - torch.tensor([0.,1.,2.], device=torch.device("cpu"))
		# ).argmin(dim=-1)

		log_prob = reparameter.log_prob(x_t)
		log_prob = log_prob - torch.sum(
			torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6), 
			dim=-1, 
			keepdim=True
		)

		return action, log_prob


### Agent
class ACSAgent:
	def __init__(self):
		self.state_dim      = 2      # [pos, vel]
		self.action_dim     = 1      # [0,1,2]
		self.lr_policy      = 0.001  # policy learning rate
		self.DEVICE   		= torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.PI  	  		= PolicyNetwork(self.state_dim, self.action_dim, self.lr_policy).to(self.DEVICE)

		self.buffer_limit   = 100000
		self.memory         = ReplayBuffer(self.buffer_limit, self.DEVICE)

		self.init_alpha     = 0.01
		self.lr_alpha       = 0.005
		self.log_alpha = torch.tensor(np.log(self.init_alpha)).to(self.DEVICE)
		self.log_alpha.requires_grad = True
		self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr_alpha)

		self.batch_size     = 200

	def choose_action(self, state):
		with torch.no_grad():
			action, log_prob = self.PI.sample(state.to(self.DEVICE))
		return action, log_prob

	def calc_target(self, mini_batch):
		print("State Action VALUE")
		s, a, r, s_prime, done = mini_batch
		with torch.no_grad():
			print("S PRIME", s_prime)
			a_prime, log_prob_prime = self.PI.sample(s_prime)
			prnt("A_prime", a_prime, log_prob_prime)
			
			entropy = - self.log_alpha.exp() * log_prob_prime

			q1_target, q2_target = self.Q1_target(s_prime, a_prime), self.Q2_target(s_prime, a_prime)
			q_target = torch.min(q1_target, q2_target)
			target = r + self.gamma * done * (q_target + entropy)
		return target

	def train_agent(self):
		print("Start TRAINING")
		mini_batch = self.memory.sample(self.batch_size)
		s_batch, a_batch, r_batch, s_prime_batch, done_batch = mini_batch
		
		td_target = self.calc_target(mini_batch)

		#### Q1 train ####
		q1_loss = F.smooth_l1_loss(self.Q1(s_batch, a_batch), td_target)
		self.Q1.optimizer.zero_grad() 			### Find Slope of Zero for Q2
		q1_loss.mean().backward()
		self.Q1.optimizer.step()
		#### Q1 train ####

		#### Q2 train ####
		q2_loss = F.smooth_l1_loss(self.Q2(s_batch, a_batch), td_target)
		self.Q2.optimizer.zero_grad()    		### Find Slope of Zero for Q2
		q2_loss.mean().backward()
		self.Q2.optimizer.step()
		#### Q2 train ####

		#### policy pi train ####
		a, log_prob = self.PI.sample(s_batch)
		entropy = -self.log_alpha.exp() * log_prob

		q1, q2 = self.Q1(s_batch, a), self.Q2(s_batch, a)
		q = torch.min(q1, q2)

		pi_loss = -(q + entropy)  # for gradient ascent
		self.PI.optimizer.zero_grad()

		pi_loss.mean().backward()
		self.PI.optimizer.step()
		#### policy train ####

		#### alpha train ####
		self.log_alpha_optimizer.zero_grad()
		alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
		alpha_loss.backward()
		self.log_alpha_optimizer.step()
		#### alpha train ####

		#### Q1, Q2 soft-update ####
		for param_target, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
			param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
		for param_target, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
			param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
		#### Q1, Q2 soft-update ####


if __name__ == '__main__':
	epocs = 3

	timestamp = "04122025"
	model_dir = "models/" + timestamp
	if not os.path.isdir(model_dir): os.mkdir(model_dir)

	env = gym.make("MountainCar-v0")
	agent = ACSAgent()
	score_list = []

	for epoc in range(epocs):
		state, info = env.reset()
		step_count = 0
		score, done = 0.0, False

		avg_reward_list = []

		while step_count < 10000:
			step_count += 1

			action, log_prob = agent.choose_action(
				torch.FloatTensor(state)
			)
			
			state_prime, reward, done, truncated, info  = env.step(action)
			if step_count % 100 == 0:
				print(
					"Epoc", epoc,
					"Step", step_count, 
					"Taking action ", action, 
					"receive reward ", reward
				)

			agent.memory.put((
				state, action, reward, state_prime, done
			))

			score += reward
			avg_reward = score/step_count
			avg_reward_list.append(avg_reward)

			state = state_prime

			if agent.memory.size() > 1000:
				agent.train_agent()



		score_list.append(score)

		plt.figure(figsize=(8, 5))
		plt.plot(avg_reward_list)
		plt.xlabel("Experiment Run", fontsize=12)
		plt.ylabel("Average Reward", fontsize=12)
		plt.title("Pendulum Average Reward", fontsize=14)
		plt.savefig("average_reward_epoc_{}.png".format(epoc), dpi=100)
