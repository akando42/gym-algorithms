import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt 

from agent import ActorCriticSoftmaxAgent

agent_parameters = {
    "num_tilings": 32,
    "num_tiles": 8,
    "actor_step_size": 2**(-2),
    "critic_step_size": 2**1,
    "avg_reward_step_size": 2**(-6),
    "num_actions": 6,
    "iht_size": 4096
}

def metrics(obs):
	cos_theta = obs[0]
	sin_theta = obs[1]
	theta_dot = obs[2]
	theta = np.degrees(np.arctan2(sin_theta, cos_theta))
	# print("Angle:", theta, "Angle Velocity: ", theta_dot)
	return (theta, theta_dot)

current_env = gym.make("Pendulum-v1", render_mode = "human")
current_agent = ActorCriticSoftmaxAgent()

obs, info = current_env.reset(seed=0)
agent_last_state = metrics(obs)

agent_info = {
 	"num_tilings": agent_parameters['num_tilings'],
    "num_tiles": agent_parameters['num_tiles'],
    "actor_step_size": agent_parameters['actor_step_size'],
    "critic_step_size": agent_parameters['critic_step_size'],
    "avg_reward_step_size": agent_parameters['avg_reward_step_size'],
    "num_actions": agent_parameters["num_actions"],
    "iht_size": agent_parameters["iht_size"]
}

optimal_policy_weights = np.load(
	"experiments/optimal_policy_38_weights.npy"
)

# print(optimal_policy_weights)

current_agent.agent_init(agent_info)
actions = np.linspace(-2.0, 2.0, 10)
avg_reward = 0
avg_rewards = []
rewards = []

for t in range(200):
	optimal_action_index = current_agent.agent_optimal(
		agent_last_state,
		optimal_policy_weights
	)

	### OPTIMAL ACTION ACCORDING TO POLICY
	action = [actions[optimal_action_index]]

	### RANDOM ACTION
	# action = current_env.action_space.sample()

	next_obs, reward, terminated, truncated, info = current_env.step(action)
	
	agent_last_state = metrics(next_obs)
	rewards.append(reward)
	avg_reward = sum(rewards)/(t+1)
	avg_rewards.append(avg_reward)
	
	print("Agent Step ", t, "with avg reward", avg_reward)
	if terminated or truncated:
		obs, info = current_env.reset()

current_env.close()

### Plot Cummulative Reward Growth 
plt.figure(figsize=(8, 5))
plt.plot(avg_rewards)
plt.xlabel("Time Step", fontsize=12)
plt.ylabel("Average Reward", fontsize=12)
plt.title("Pendulum Average Reward Growth", fontsize=14)
plt.savefig("reward_growth.png", dpi=100)
plt.show()




