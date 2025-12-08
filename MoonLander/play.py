import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
min_param, max_param = env.observation_space.low, env.observation_space.high

x_range = np.linspace(min_param[0], max_param[0], 10, endpoint=True)
y_range = np.linspace(min_param[1], max_param[1], 10, endpoint=True)

x_velocity = np.linspace(min_param[2], max_param[2], 10, endpoint=True)
y_velocity = np.linspace(min_param[3], max_param[3], 10, endpoint=True)

angle = np.linspace(min_param[4], max_param[4])
angle_vel = np.linspace(min_param[5], max_param[5])

left_leg_contact = min_param[6], max_param[6]
right_leg_contact = min_param[7], max_param[7]

print("X Range ", x_range)
print("Y Range ", y_range)

print("X Velocity ", x_velocity)
print("Y Velocity ", y_velocity)

print("Angle ", angle)
print("Angle Vel", angle_vel)

print("Left Contact", left_leg_contact)
print("Right Contact", right_leg_contact)

actions = list(range(env.action_space.n))
print("ACTION SPACE ", actions)

state, info = env.reset()
total_reward = 0
max_step = 200

while True:
    # action = env.action_space.sample()
    action = 0
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state
    # print("Step ", step, reward)
    # print("State ", next_state)
    total_reward += reward

    if terminated or truncated:
        print(total_reward)
        print(f"Lander Crashed")
        break

env.close()