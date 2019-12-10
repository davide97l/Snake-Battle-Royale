import gym
import matplotlib.pyplot as plt
import numpy as np
import utils

env = gym.make('CartPole-v1')
env.reset()
utils.plot_environment(env)


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


# test the algorithm 500 times for at most 200 steps
totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

# test the algorithm for at most 200 steps and get all frames
frames = []
obs = env.reset()
for step in range(200):
    img = env.render(mode="rgb_array")
    frames.append(img)
    action = basic_policy(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break

# visualize cart pole animation
anim = utils.plot_animation(frames)
utils.save_animation(frames, 'cart_pole_naive.mp4', 30)
plt.show()
env.close()
