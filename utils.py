import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2


def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    return img


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=50):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch), blit=True,
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


def save_animation(frames, path, fs=30):
    size = frames[0].shape[1], frames[0].shape[0]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MP4V'), fs, size)
    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
