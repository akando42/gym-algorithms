import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")
obs, info = env.reset()

obs_space = env.observation_space

vel_range = [
    obs_space.low[0],
    obs_space.high[0]
]

pos_range = [
    obs_space.low[1], 
    obs_space.high[1]
]

print("Action space:", env.action_space)
print("Position ", pos_range)
print("Velocity ", vel_range)

# print(env.action_space)

max_step = 1000

for step in range(max_step):
    # action = env.action_space.sample()
    action = np.random.choice([0, 1, 2])
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state

    print(action, state, reward )

    # print("Step ", step)
    # print("State ", state)

    if terminated or truncated:
        print(f"Reach Final Step")
        break

env.close()