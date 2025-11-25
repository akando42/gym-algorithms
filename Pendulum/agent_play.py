import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt 

env = gym.make("Pendulum-v1", render_mode = "human")
observations, info = env.reset(seed=0)

def metrics(obs):
	cos_theta = obs[0]
	sin_theta = obs[1]
	theta_dot = obs[2]
	theta = np.degrees(np.arctan2(sin_theta, cos_theta))
	# print("Angle:", theta, "Angle Velocity: ", theta_dot)
	return (theta, theta_dot)

metrics(observations)

total_reward = 0
reward_growth = []

for t in range(200):
	action = env.action_space.sample()
	next_obs, reward, terminated, truncated, info = env.step(action)
	total_reward += reward
	reward_growth.append(total_reward)
	
	(theta, theta_dot) = metrics(next_obs)
	print("Step ", t, theta, theta_dot, "Reward ", reward)

	if terminated or truncated:
		obs, info = env.reset()

print("Total Reward ", total_reward)
env.close()


### Plot Cummulative Reward Growth 
plt.figure(figsize=(8, 5))
plt.plot(reward_growth)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Cummulative Reward", fontsize=12)
plt.title("Pendulum Cummulative Reward Growth", fontsize=14)
plt.savefig("reward_growth.png", dpi=100)
plt.show()
