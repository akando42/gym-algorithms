import numpy as  np
from base_agent import BaseAgent
from tile_coder import PendulumTileCoder
from softmax import compute_softmax_prob

def get_active(angle, ang_vel):
	space = np.zeros((360, 360))

	### 360 angle group
	### Generate 360 group from -pi to pi
	angles = np.linspace(-180, 180, 360, endpoint=False)
	angle_index = np.argmin(np.abs(angles - angle))

	### 360 angle velocity group
	### Generate 360 group from -2pi to 2pi
	angle_vels = np.linspace(-8, 8, 360, endpoint=False)
	angle_vel_index = np.argmin(np.abs(angle_vels - ang_vel))

	space[angle_index][angle_vel_index] = 1

	return space


class ActorCriticGaussianAgent(BaseAgent):
	def __init__(self):
		self.actor_step_size = None
		self.critic_step_size = None

		self.avg_reward = None

		self.actor_w = None
		self.critic_w = None
		self.actions = None

		self.softmax_prob = None
		self.prev_tiles = None
		self.last_action = None

	def agent_init(self):
		self.actor_step_size = agent_info.get("actor_step_size")
		self.critic_step_size = agent_info.get("critic_step_size")
		self.actions = list(range(agent_info.get("num_actions")))
		print("POSSIBLE ACTIONS: ", self.actions)

		hoang_space_size = agent_info.get("space_size")

		self.avg_reward = 0.0

		self.actor_w = np.zeros((
			len(self.actions), 
			hoang_space_size
		))

		self.critic_w = np.zeros({
			hoang_space_size
		})


	def agent_policy(self, active_states):

		### Recieve Active State Table 

		### Multiply Policy Weight by Active State table to find Policy Value

		### Select Best Policy

		### Return Action
		print("Agent Policy")


	def agent_start(self, state):
		angle, ang_vel = state

		active_states = get_active(angle, ang_vel)
		current_action = self.agent_policy(active_states)

		self.last_action = current_action
		self.prev_state = np.copy(active_states)

		return self.last_action

	def agent_step(self, reward, state):
		angle, ang_vel = state
		active_states = get_active(angle, ang_vel)

		## Calculate State Value Weights
		v_current = np.sum(self.critic_w[active_state])
		v_prev = np.sum(self.critic_w[self.prev_state])
		delta = reward - self.avg_reward + v_current - v_prev

		self.critic_w[self.prev_state] += self.critic_step_size * delta

		## Calculate Policy Weights
		if (v_current > v_prev):
			## Increase Policy Weight of current action
			self.actor_w[self.last_action][self.prev_state] += self.actor_step_size * delta
		else:
			self.actor_w[self.last_action][self.prev_state] += 0
			## Maintain Policy Weight of current action

		current_action = self.agent_policy(active_states)

		self.prev_state = active_state
		self.last_action = current_action

		return self.last_action

	
	def agent_message(self, message):
		if message == 'get avg reward':
			return self.avg_reward

		if message == "get actor weights":
			return self.actor_w

	def agent_optimal(self, state, optimal_policy_weights):
		angle, ang_vel = state
		active_tiles = self.tc.get_tiles(
			angle,
			ang_vel
		)

		softmax_prob = compute_softmax_prob(
			optimal_policy_weights,
			active_tiles
		)

		print("Softmax Probability ", softmax_prob)

		chosen_action = self.rand_generator.choice(
			self.actions, p=softmax_prob
		)

		return chosen_action


