import gymnasium as gym
import numpy as np

import torch

class DQN(torch.nn.Module):
    '''
    This class defines a deep Q-network (DQN), a type of artificial neural network used in reinforcement learning.
    The DQN is used to estimate the Q-values, which represent the expected return for each action in each state.
    
    Parameters
    ----------
    state_size: int, default=8
        The size of the state space.
    action_size: int, default=4
        The size of the action space.
    hidden_size: int, default=64
        The size of the hidden layers in the network.
    '''
    def __init__(self, state_size=8, action_size=4, hidden_size=64):
        '''
        Initialize a network with the following architecture:
            Input layer (state_size, hidden_size)
            Hidden layer 1 (hidden_size, hidden_size)
            Output layer (hidden_size, action_size)
        '''
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, state):
        '''
        Define the forward pass of the DQN. This function is called when the network is called to estimate Q-values.
        
        Parameters
        ----------
        state: torch.Tensor
            The state for which to estimate the Q-values.

        Returns
        -------
        torch.Tensor
            The estimated Q-values for each action in the input state.
        '''
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

### INIT ENV
env = gym.make('LunarLander-v3', render_mode="human")

### LOAD MODEL
state_size = 8
action_size = 4
hidden_size = 64

device =  torch.device("cpu")

model = DQN(state_size, action_size, hidden_size).to(device)
model.load_state_dict(
    torch.load("DQN_Agent.pt", map_location="cpu")
)

def play_DQN_episode(env):
    score = 0
    state, _ = env.reset(seed=42)
    
    while True:
        # eps=0 for predictions
        # action = agent.act(state, 0)
        s = torch.tensor(state, dtype=torch.float32)
        action_values = model.forward(s).detach().numpy()

        print("Action Values", action_values)
        action = np.argmax(action_values)
        print("Taking Action ", action)

        state, reward, terminated, truncated, _ = env.step(action) 
        done = terminated or truncated

        score += reward

        # End the episode if done
        if done:
            break 

    return score
 
score = play_DQN_episode(env)
print("Score obtained:", score)
