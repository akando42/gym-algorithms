import numpy as np
import gymnasium as gym

def discretize_state(state, bins):
    """Discretize the continuous state into discrete bins."""
    if isinstance(state, tuple):
        state = state[0]
    discretized_state = tuple(np.digitize(s, bins[i]) - 1 for i, s in enumerate(state))
    return discretized_state

Q_table = np.load('optimal_q_table.npy')
max_steps=1000

# print(f"Q-table size: {Q_table.shape}")
# print("Q-table after training:")
# print(Q_table)

env = gym.make('MountainCar-v0', render_mode="human")

state_bins = [np.linspace(-1.2, 0.6, 20), np.linspace(-0.07, 0.07, 20)]
initial_state = env.reset()
initial_state = discretize_state(initial_state, state_bins)

total_reward = 0

state = initial_state

for step in range(max_steps):
	action = np.argmax(Q_table[state]) 
	next_state, reward, done, _ = env.step(action)[:4]
	state = discretize_state(next_state, state_bins)
	total_reward += reward
	if done:
		break
