import matplotlib.pyplot as plt
import os
from snake.GAagent import get_stats_as_history


def GA_charts(population_name="standard_population"):
    max_fitness, max_avg_score, avg_fitness, avg_deaths, avg_score, max_score = get_stats_as_history(population_name)
    save_path = "weights/genetic_algorithm/" + str(population_name) + "/charts/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    title = "max fitness value over generations"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(max_fitness)
    ax.set(xlabel='generation', ylabel='max fitness value', title=title)
    ax.grid()
    fig.savefig(save_path + "/" + title + ".png")

    title = "average score over generations"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_score, label="Average score")
    ax.plot(max_avg_score, label="Max average score")
    ax.set(xlabel='generation', ylabel='score', title=title)
    ax.grid()
    plt.legend()
    fig.savefig(save_path + "/" + title + ".png")

    title = "average fitness value over generations"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_fitness)
    ax.set(xlabel='generation', ylabel='average fitness value', title=title)
    ax.grid()
    fig.savefig(save_path + "/" + title + ".png")

    title = "average deaths over generations"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(avg_deaths)
    ax.set(xlabel='generation', ylabel='average deaths', title=title)
    ax.grid()
    fig.savefig(save_path + "/" + title + ".png")

    title = "max score value over generations"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(max_score)
    ax.set(xlabel='generation', ylabel='max score value', title=title)
    ax.grid()
    fig.savefig(save_path + "/" + title + ".png")

    plt.show()


GA_charts()

