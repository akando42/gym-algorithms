import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

def metrics(obs):
	cos_theta = obs[0]
	sin_theta = obs[1]
	theta_dot = obs[2]
	theta = np.degrees(np.arctan2(sin_theta, cos_theta))
	# print("Angle:", theta, "Angle Velocity: ", theta_dot)
	return (theta, theta_dot)

### ACTOR-CRITIC AGENT ###

class ActorCritic:
	def __init__(self, 
			state_dim, 
			actor_alpha=0.01, 
			critic_alpha=0.1, 	
		):

		print("Agent ready")
		return

	def policy(self, obs):
		### Gaussian Random Action
		# action = np.random.uniform(-2.0, 2.0)

		### Max Left Action
		# action = -2

		### Max Right Action
		action = 2
		return [action] 


### TRAINING LOOP

env = gym.make("Pendulum-v1", render_mode= "human")
print("Action Space", env.action_space)
obs, info = env.reset(seed=0)

agent = ActorCritic(state_dim=3)

num_episodes = 100
rewards_per_episode = []

for episode in range(num_episodes):
	print("Training in episode ", episode)
	obs, info = env.reset() 
	total_reward = 0

	for step in range(200):
		action = agent.policy(obs)
		next_obs, reward, terminated, truncated, info = env.step(action)
		(theta, theta_dot) = metrics(next_obs)
		print(
			"Episode ", episode, 
			"Step ", step, 
			"Taking Action ", action,
			"Reward ", reward, 
			theta, theta_dot, 
		)
		done = terminated or truncated
		total_reward += reward
		obs = next_obs

		if done:
			break


env.close()
