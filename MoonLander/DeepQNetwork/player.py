import gymnasium as gym
import numpy as np
import torch
import os

class DQN(torch.nn.Module):
    def __init__(self, state_size=8, action_size=4, hidden_size=64):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, state):
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

def play_DQN_episode(env, agent_neural_network):
    model = DQN(state_size, action_size, hidden_size).to(device)
    model.load_state_dict(
        torch.load(agent_neural_network, map_location="cpu")
    )
    score = 0
    state, _ = env.reset(seed=42)
    
    while True:
        s = torch.tensor(state, dtype=torch.float32)
        action_values = model.forward(s).detach().numpy()
        action = np.argmax(action_values)
    
    
        state, reward, terminated, truncated, _ = env.step(action) 
        done = terminated or truncated

        score += reward

        # End the episode if done
        if done:
            break 

    return score

agent_dir = "OptimaAgents"
agents = os.listdir(agent_dir)
scores = []
for agent in agents:
    model_path = agent_dir + "/" + agent
    score = play_DQN_episode(env, model_path)
    scores.append(score)
    print(agent, " scored ", score)

max_index = scores.index(max(scores))
max_agent = agents[max_index]

print(max_agent, "is the most OPTIMAL with score ", scores[max_index])
model_path = agent_dir + "/" + max_agent

for i in range(3):
    play_DQN_episode(env, model_path)
