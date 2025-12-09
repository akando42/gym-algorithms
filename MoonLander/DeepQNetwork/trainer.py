import torch ### Cost higher Energy and Burn Laptop ?

import numpy as np
import gymnasium as gym

import random
from collections import deque

##############################
### DeepQNetwork Algorithm ###
##############################

# Initialize Q_online, Q_target = Q_online
# Initialize replay buffer D

# for each episode:
#     s = env.reset()

#     while not done:
#         a = epsilon_greedy(Q_online, s)
#         s', r, done = env.step(a)
#         store (s, a, r, s', done) in D
#         if len(D) > batch_size:
#             batch = sample(D)
#             y = compute_targets(batch)
#             update(Q_online, y)
#         every C steps:  Q_target ← Q_online
#         s = s'

class ReplayBuffer:
	def __init__(self, buffer_size=10000):
		self.buffer = deque(maxlen=buffer_size)

	def push(self, state, action, reward, next_state, done):
		self.buffer.append([
			state, action, reward, next_state, done
		])

	def sample(self, batch_size):
		print("Sampling State from Memory Store")
		states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
		return np.stack(states), actions, rewards, np.stack(next_states), dones

	def __len__(self):
		return len(self.buffer)


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

		self.memory = ReplayBuffer(10000)
		self.batch_size = 64


	### Reinforcing Improving Steps 
	def step(self, state, action, reward, next_state, done):
		print("Making Planning Step in Simulation")
		### Store Experiment in Memory
		self.memory.push(state, action, reward, next_state, done)

		### Perform Learning after Reach Experiment Threshold
		if len(self.memory) > self.batch_size:
			self.update_model()

	### Optima Action using Optimal Policy
	def act(self, state, eps=0.):
		### Choose Optima Action 
		if random.random() > eps:
			print("OPTIMA STEP")
			state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
			self.q_network.eval()

			with torch.no_grad():
				action_values = self.q_network(state)

			### ReTrain Agent ??
			self.q_network.train() 

			### Select Action with Highest Values
			optima_action = np.argmax(action_values.cpu().data.numpy())
			return optima_action

		### Choose Random Action
		else:
			print("RANDOM STEP")
			random_action = random.choice(np.arange(self.action_size))
			return random_action

	### Updating Optimal Policy Model
	def update_model(self):
		print("Updating Policy Model ", self.batch_size)

		### Sample 64 Experiments from Memory
		states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
		# print("SAMPLE LENGTH ", len(states))

		states = torch.from_numpy(states).float().to(self.device)
		next_states = torch.from_numpy(next_states).float().to(self.device)

		actions = torch.from_numpy(np.array(actions)).long().to(self.device)
		rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)

		dones = torch.from_numpy(np.array(dones).astype(np.uint8)).float().to(self.device)

		### Get Q-Value for Actions in the Policy Network
		q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

		### Get Maximum Q Value for Next State in Target Network
		next_q_values = self.target_network(next_states).max(1)[0].detach()

		### Compute Expected Q Values for Next 64 steps in future
		expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

		### Compute Differences between Expected States and Sampled States
		loss = torch.nn.MSELoss()(q_values, expected_q_values)

		### Zero All Gradients to Find Slope Zero Point
		self.optimizer.zero_grad()

		### Update Q Neural Network to Reduce Loss
		loss.backward()

		### Step the Optimizer
		self.optimizer.step()

	### Updating State Action Value Estimation
	def update_target_network(self):
		print("Updating State Action Value Estimation")


training_env = gym.make("LunarLander-v3")
state_size = training_env.observation_space.shape[0]
action_size = training_env.action_space.n

agent = DQNAgent(state_size, action_size)

### Training Neural Network
n_training_epocs = 3000

eps_start = 1.0  
eps_end = 0.01  ### Decaying Epsilon Rate
eps_decay = 0.995

scores = []
eps = eps_start

for epoc in range(1, n_training_epocs + 1):
	state, _ = training_env.reset()
	score = 0

	while True:
		print("Training in Epoc ", epoc)

		### Select Action using Current Policy
		action = agent.act(state, eps)
		next_state, reward, terminated, truncated, _ = training_env.step(action)
		done = terminated or truncated

		### Update Agent Policy Neural Network
		agent.step(state, action, reward, next_state, done)
		state = next_state
		score += reward

		if done:
			break

### Agent Playing using After Training Neural Network
action_env = gym.make("LunarLander-v3", render_mode="human")
state, info = action_env.reset(seed=40)
score = 0

while True:
	### Random Action
	# action = env.action_space.sample()

	### Optima Agent Action
	random_rate = 0
	action = agent.act(state, random_rate)

	state, reward, terminated, truncated, _ = action_env.step(action)	
	print("Taking Action ", action, "Reward ", reward)
	done = terminated or truncated

	score += reward

	if done:
		break



