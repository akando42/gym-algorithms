import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")
state, info = env.reset()

max_step = 1000
for step in range(max_step):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    print("Step ", step)

    if terminated or truncated:
        print(f"Reach Final Step")
        break

env.close()