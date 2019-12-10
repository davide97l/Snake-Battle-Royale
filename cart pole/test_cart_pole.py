import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import utils


# run the model to play one episode, and return the frames
def render_policy_net(model, n_max_steps=500, seed=42):
    frames = []
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    return frames


env = gym.make('CartPole-v1')
env.reset()

filename = "cart_pole100.h5"
model = 0
if os.path.exists(filename):
    model = keras.models.load_model(filename)
frames = render_policy_net(model, n_max_steps=500)
# visualize cart pole animation
anim = utils.plot_animation(frames)
utils.plot_animation(frames)
utils.save_animation(frames, 'cart_pole.mp4', 30)
plt.show()
env.close()
