import torch ### Cost higher Energy and Burn Laptop ?

import numpy as np
import gymnasium as gym

import random

class DQN(torch.nn.Module):
	def __init__(self, state_size=8, action_size=4, hidden_size=64):
		### Three Layer Neural Network mapping State to Hidden to Action
		### Approximate Curves via Lines 

		super(DQN, self).__init__()
		self.layer1 = torch.nn.Linear(state_size, hidden_size)
		self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
		self.layer3 = torch.nn.Linear(hidden_size, action_size)

	def forward(self, state):
		#### Using RELU to map Continuous Number to 0 1 
		x = torch.relu(self.layer1(state))
		x = torch.relu(self.layer2(x))
		return self.layer3(x)

class DQNAgent:
	def __init__(self, state_size=8, action_size=4):
		self.device = torch.device("cpu")
		self.action_size = action_size
		self.state_size = state_size
		self.hidden_size = 64

		self.q_network = DQN(state_size, action_size, self.hidden_size).to(self.device)


	### Reinforcing Improving Steps 
	def step():
		print("Making Planning Step in Simulation")

	### Optima Action using Optimal Policy
	def act(self, state, eps=0.):
		### Choose Optima Action 
		if random.random() > eps:
			state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
			self.q_network.eval()

			with torch.no_grad():
				action_values = self.q_network(state)

			### ReTrain Agent
			# self.q_network.train() 

			### Select Action with Highest Values
			optima_action = np.argmax(action_values.cpu().data.numpy())
			return optima_action

		### Choose Random Action
		else:
			random_action = random.choice(np.arange(self.action_size))
			return random_action



	### Updating Optimal Policy Model
	def update_model(self):
		print("Updating Policy Model")


	### Updating State Action Value Estimation
	def update_target_network(self):
		print("Updating State Action Value Estimation")


env = gym.make("LunarLander-v3", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

score = 0
state, info = env.reset(seed=40)

agent = DQNAgent(state_size, action_size)

while True:
	### Random Action
	# action = env.action_space.sample()

	### Optima Agent Action
	action = agent.act(state, 0)


	state, reward, terminated, truncated, _ = env.step(action)
	print("Taking Action ", action, "Reward ", reward)
	done = terminated or truncated

	score += reward

	if done:
		break



