import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt 

def group_index(state, groups):
	if isinstance(state, tuple):
		state = state[0]
	group_index = tuple(
		np.digitize(vel, groups[pos]) - 1 for pos, vel in enumerate(state)
	)
	return group_index

def q_table(env):
	alpha = 0.06       ### Learning Rate
	gamma = 0.99       ### Discount Rate
	epsilon = 0.3      ### Randomize Rate
	num_epocs = 500    ### Number of Experiment
	
	num_actions = env.action_space.n
	Q = np.zeros((
		num_groups[0],
		num_groups[1],
		num_actions
	))

	rewards = []
	num_actions_list = []

	for epoc in range(num_epocs):
		initial_state = env.reset()
		state = group_index(initial_state, state_groups)

		total_reward = 0
		num_actions_epoc = 0

		while True:
			if np.random.rand() < epsilon:
				action = env.action_space.sample()  ### ?
			else:
				action = np.argmax(Q[state])

			next_state, reward, done, _ = env.step(action)[:4]
			next_state = group_index(next_state, state_groups)

			### Q-VALUE update using the Q-Learning formula
			Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

			state = next_state
			total_reward += reward
			num_actions_epoc += 1

			if done:
				break

		rewards.append(total_reward)
		num_actions_list.append(num_actions_epoc)
		print(
            "EPISODE ", epoc, 
            "REWARD ", total_reward,
            "STEP ACTION counts ", num_actions_epoc
        )

	np.save("optimal_q_table.npy", Q)
	return rewards, num_actions_list

env = gym.make('MountainCar-v0', render_mode="rgb_array")
initial_state = env.reset()

num_groups = [21, 21]

state_groups = [
	np.linspace(
		env.observation_space.low[0],
		env.observation_space.high[0],
		num_groups[0]
	),
	np.linspace(
		env.observation_space.low[1],
		env.observation_space.high[1],
		num_groups[1]
	)
]

print("STATE GROUP ", state_groups)


rewards, num_actions_list = q_table(env)

plt.figure(figsize=(6, 4))
plt.plot(rewards,color='r', label='Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-learning in MountainCar-v0')
plt.savefig("MountainCar_QLearning_Algo.png", dpi=100)
plt.grid(True)

# print("OBSERVATION SPACE ", env.observation_space.low)
# print("OBSERVATION SPACE ", env.observation_space.high)
# print("ACTION SPACE", env.action_space)
# print("REWARD RANGE", env.action_space.n)

