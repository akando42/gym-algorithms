import numpy as np
from tqdm import tqdm

from pendulum_env import PendulumEnvironment
from agent import ActorCriticSoftmaxAgent
from rl_glue import RLGlue

import os
import plot_script
import gymnasium as gym

import matplotlib.pyplot as plt 

def metrics(obs):
	cos_theta = obs[0]
	sin_theta = obs[1]
	theta_dot = obs[2]
	theta = np.degrees(np.arctan2(sin_theta, cos_theta))
	# print("Angle:", theta, "Angle Velocity: ", theta_dot)
	return (theta, theta_dot)

def run_experiment(
		current_env, 
		current_agent, 
		experiment_parameters,
		agent_parameters
	):

	obs, info = current_env.reset(seed=0)
	agent_info = {
	 	"num_tilings": agent_parameters['num_tilings'],
        "num_tiles": agent_parameters['num_tiles'],
        "actor_step_size": agent_parameters['actor_step_size'],
        "critic_step_size": agent_parameters['critic_step_size'],
        "avg_reward_step_size": agent_parameters['avg_reward_step_size'],
        "num_actions": agent_parameters["num_actions"],
        "iht_size": agent_parameters["iht_size"]
    }

	return_per_experiment = np.zeros(
		(experiment_parameters['num_runs'])
	)

	avg_reward_per_experiment = np.zeros(
		(experiment_parameters['num_runs'])
	)

	max_avg_reward = 0
	best_experiment_index = 0
	optimal_policy_weights = np.zeros((10, 4096))

	for i in tqdm(range(experiment_parameters['num_runs'])):		
		
		obs, info = current_env.reset() 
		total_reward = 0	
		average_reward = 0
		num_steps = 0

		#print(agent_parameters)

		### AGENT INIT
		agent_info["seed"] = i
		# print(current_agent)
		# print(agent_info)

		current_agent.agent_init(agent_info)
		actions = np.linspace(-2.0, 2.0, 10)

		agent_last_state = metrics(obs)
		agent_last_action = None
		agent_last_reward = None

		while num_steps < experiment_parameters['max_steps']:
			num_steps += 1
			action = None
			###
			
			### MAX LEFT
			# action = [1.9] 

			if num_steps == 1:
				### RANDOM START ACTION
				# action = current_env.action_space.sample()

				### ACTOR CRITIC SOFTMAX AGENT START
				agent_action_index = current_agent.agent_start(agent_last_state)
				action = [actions[agent_action_index]]

			else: 
				### ACTOR_CRITIC SOFTMAX AGENT STEP
				agent_action_index = current_agent.agent_step(
					agent_last_reward,
					agent_last_state
				)
				
				action = [actions[agent_action_index]]
				# print("AGENT START ACTION ",action)
				

			### IMPROVING ACTION SELECTION
			###
			# print(
			# 	"Taking Step ", num_steps, 
			# 	"with action ", action
			# )

			next_obs, reward, terminated, truncated, info = current_env.step(action)
			# if reward > -0.1:
			# 	print(
			# 		"EXPERIMENT ", i,
			# 		"STEP", num_steps, 
			# 		"REWARD ", reward
			# 	)

			total_reward += reward

			average_reward = current_agent.agent_message(
				"get avg reward"
			)

			#print("AVERAGE REWARD ", reward, average_reward)

			agent_last_state = metrics(next_obs)
			agent_last_reward = reward
			agent_last_action = action

			if terminated or truncated:

				obs, info = current_env.reset()

		return_per_experiment[i] = total_reward
		avg_reward_per_experiment[i] = average_reward
		actor_weights = current_agent.agent_message(
			"get actor weights"
		)

		# print("TOTAL REWARD ", total_reward)
		# print("AVERAGE REWARD ", average_reward)
		# print("ACTOR WEIGHT ", actor_weights.shape)

		if average_reward > max_avg_reward:
			max_avg_reward = average_reward
			optimal_policy_weights = actor_weights
			best_experiment_index = i

		# print("######## End Experiment ", i, "Reward ", total_reward)

	# print(return_per_experiments)

	current_env.close()

	if not os.path.exists('experiments'):
		os.makedirs('experiments')

	exper_data_file = "experiments/exper_data.npy"
	np.save(exper_data_file, return_per_experiment)

	avg_reward_data_file = "experiments/avg_reward_exp.npy"
	np.save(avg_reward_data_file, avg_reward_per_experiment)

	print(
		"BEST EXPERIMENT ", best_experiment_index, 
		"Avg Reward ", max_avg_reward
	)
	np.save(
		"experiments/optimal_policy_{}_weights.npy".format(
			best_experiment_index,
		),
		optimal_policy_weights
	)

	### Plot Total Reward Per Experiment
	# plt.figure(figsize=(8, 5))
	# plt.plot(return_per_experiment)
	# plt.xlabel("Experiment Run", fontsize=12)
	# plt.ylabel("Cummulative Reward", fontsize=12)
	# plt.title("Pendulum Cummulative Reward Growth", fontsize=14)
	# plt.savefig("reward_growth.png", dpi=100)
	# plt.show()

	### Plot Average Reward Per Experiment
	plt.figure(figsize=(8, 5))
	plt.plot(avg_reward_per_experiment)
	plt.xlabel("Experiment Run", fontsize=12)
	plt.ylabel("Average Reward", fontsize=12)
	plt.title("Pendulum Average Reward", fontsize=14)
	plt.savefig("average_reward.png", dpi=100)
	plt.show()


### ActorCritic Softmax TileCoding Agent
agent_parameters = {
    "num_tilings": 32,
    "num_tiles": 8,
    "actor_step_size": 2**(-2),
    "critic_step_size": 2**1,
    "avg_reward_step_size": 2**(-6),
    "num_actions": 10,
    "iht_size": 4096
}

# Environment parameters
environment_parameters = {}

experiment_parameters = {
	"max_steps": 100, 
	"num_runs": 5000
}

current_env = gym.make("Pendulum-v1")
current_agent = ActorCriticSoftmaxAgent()

run_experiment(
	current_env, 
	current_agent, 
	experiment_parameters, 
	agent_parameters
)

