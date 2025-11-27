import os
import numpy as np
import gymnasium as gym
from agent import ActorCriticSoftmaxAgent

agent_parameters = {
    "num_tilings": 32,
    "num_tiles": 8,
    "actor_step_size": 2**(-2),
    "critic_step_size": 2**1,
    "avg_reward_step_size": 2**(-6),
    "num_actions": 10,
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
	"experiments/optimal_policy_4989_weights.npy"
)

print(optimal_policy_weights)

current_agent.agent_init(agent_info)
actions = np.linspace(-2.0, 2.0, 10)
best_reward = 0

for t in range(200):
	print("Agent Step ", t, "best reward ", best_reward)
	optimal_action_index = current_agent.agent_optimal(
		agent_last_state,
		optimal_policy_weights
	)

	action = [actions[optimal_action_index]]
	next_obs, reward, terminated, truncated, info = current_env.step(action)
	agent_last_state = metrics(next_obs)
	if reward > best_reward:
		best_reward = reward

	if terminated or truncated:
		obs, info = current_env.reset()

current_env.close()



