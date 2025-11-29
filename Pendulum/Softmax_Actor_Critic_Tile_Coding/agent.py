import numpy as  np
from base_agent import BaseAgent
from tile_coder import PendulumTileCoder
from softmax import compute_softmax_prob

class ActorCriticSoftmaxAgent(BaseAgent): 
    def __init__(self):
        self.rand_generator = None

        self.actor_step_size = None
        self.critic_step_size = None
        self.avg_reward_step_size = None

        self.tc = None

        self.avg_reward = None
        self.critic_w = None
        self.actor_w = None

        self.actions = None

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
    
    def agent_init(self, agent_info={}):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            "iht_size": int
            "num_tilings": int,
            "num_tiles": int,
            "actor_step_size": float,
            "critic_step_size": float,
            "avg_reward_step_size": float,
            "num_actions": int,
            "seed": int
        }
        """
        # print("AGENT INFO", agent_info)

        # set random seed for each run
        self.rand_generator = np.random.RandomState(
            agent_info.get("seed")
        ) 

        iht_size = agent_info.get("iht_size")
        num_tilings = agent_info.get("num_tilings")
        num_tiles = agent_info.get("num_tiles")

        # initialize self.tc to the tile coder we created
        self.tc = PendulumTileCoder(
            iht_size=iht_size, 
            num_tilings=num_tilings, 
            num_tiles=num_tiles
        )

        # set step-size accordingly (we normally divide actor and critic step-size by num. tilings (p.217-218 of textbook))
        self.actor_step_size = agent_info.get("actor_step_size")/num_tilings
        self.critic_step_size = agent_info.get("critic_step_size")/num_tilings
        self.avg_reward_step_size = agent_info.get("avg_reward_step_size")

        self.actions = list(range(agent_info.get("num_actions")))
        print(
            "POSSIBLE ACTIONS ", 
            self.actions
        )

        # Set initial values of average reward, actor weights, and critic weights
        # We initialize actor weights to three times the iht_size. 
        # Recall this is because we need to have one set of weights for each of the three actions.
        self.avg_reward = 0.0
        self.actor_w = np.zeros(
            (len(self.actions), 
            iht_size)
        )
        self.critic_w = np.zeros(iht_size)

        self.softmax_prob = None
        self.prev_tiles = None
        self.last_action = None
    
    def agent_policy(self, active_tiles):
        """ policy of the agent
        Args:
            active_tiles (Numpy array): active tiles returned by tile coder
            
        Returns:
            The action selected according to the policy
        """
        
        #### SOFTMAX POLICY

        # compute softmax probability
        softmax_prob = compute_softmax_prob(
            self.actor_w, active_tiles
        )
        
        # Sample action from the softmax probability array
        # self.rand_generator.choice() selects an element from the array with the specified probability
        chosen_action = self.rand_generator.choice(
            self.actions, p=softmax_prob
        )
        
        # save softmax_prob as it will be useful later when updating the Actor
        self.softmax_prob = softmax_prob

        #### GAUSSIAN POLICY
        # print("ACTIVE TILE ", active_tiles)
        # print("WEIGHTS ", self.actor_w.shape)
        # mu = np.sum(self.actor_w[active_tiles])
        # mu = np.clip(mu, -2.0, 2.0)

        # action = mu + self.sigma * self.rand_generator.randn()
        # chosen_action = float(np.clip(action, -2.0, 2.0))

        # self.mu = mu 
        
        return chosen_action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        angle, ang_vel = state

        ### Use self.tc to get active_tiles using angle and ang_vel (2 lines)
        # set current_action by calling self.agent_policy with active_tiles
        # active_tiles = ?
        # current_action = ?

        # ----------------
        # your code here
        active_tiles = self.tc.get_tiles(angle, ang_vel)
        current_action = self.agent_policy(active_tiles)
        
        # ----------------

        self.last_action = current_action
        self.prev_tiles = np.copy(active_tiles)

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the environment's step based on 
                                where the agent ended up after the
                                last step.
        Returns:
            The action the agent is taking.
        """

        angle, ang_vel = state

        ### Use self.tc to get active_tiles using angle and ang_vel (1 line)
        # active_tiles = ?    
        # ----------------
        # your code here
        active_tiles = self.tc.get_tiles(angle, ang_vel)
        
        # ----------------

        ### Compute delta using Equation (1) (1 line)
        # delta = ?
        # ----------------
        # your code here
        v_current = np.sum(self.critic_w[active_tiles])
        v_prev = np.sum(self.critic_w[self.prev_tiles])
        delta = reward - self.avg_reward + v_current - v_prev
        
        # ----------------

        ### update average reward using Equation (2) (1 line)
        # self.avg_reward += ?
        # ----------------
        # your code here
        self.avg_reward += self.avg_reward_step_size * delta
        
        # ----------------

        # update critic weights using Equation (3) and (5) (1 line)
        # self.critic_w[self.prev_tiles] += ?
        # ----------------
        # your code here
        self.critic_w[self.prev_tiles] +=  self.critic_step_size * delta 
        
        # ----------------

        # update actor weights using Equation (4) and (6)
        # We use self.softmax_prob saved from the previous timestep
        # We leave it as an exercise to verify that the code below corresponds to the equation.

        ### SOFTMAX ACTOR WEIGHT UPDATES
        for a in self.actions:
            if a == self.last_action:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_w[a][self.prev_tiles] += self.actor_step_size * delta * (0 - self.softmax_prob[a])

        ### GAUSSIAN ACTOR WEIGHT UPDATES
        # grad_log_pi = (self.last_action - self.mu) / (self.sigma ** 2)
        # self.actor_w[a][self.prev_tiles] +=  self.actor_step_size * delta * grad_log_pi

        ### set current_action by calling self.agent_policy with active_tiles (1 line)
        # current_action = ? 
        # ----------------
        # your code here
        current_action = self.agent_policy(active_tiles)
        
        # ----------------

        self.prev_tiles = active_tiles
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

