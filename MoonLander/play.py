import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
state, info = env.reset()

print("ENV", env)

max_step = 200
for step in range(max_step):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    print("Step ", step)

    if terminated or truncated:
        print(f"Reach Final Step")
        break

env.close()