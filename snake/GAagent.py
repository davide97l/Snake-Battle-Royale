import random
from keras import backend as K
import os
from snake.Game import Game
from snake.Food import Food
from snake.Player import Player
from keras.models import Sequential
from keras.layers.core import Dense
import numpy as np


class GAAgent(object):

    def __init__(self, units=[12, 120, 120, 120, 4], population_name="standard_population", generation=100,
                 population=50, training=False):
        self.units = units
        self.population_name = population_name
        self.generation = generation
        self.population = population
        if not training:
            self.model = create_model_from_units(units, best_snake_weights(population_name, generation,
                                                                       (population, weights_size(units))))
        self.dim_state = units[0]
        self.name = "ga"

    def get_state(self, game, player, food):

        game_matrix = np.zeros(shape=(game.width+2, game.height+2))
        for p in game.player:
            for i, coord in enumerate(p.position):
                game_matrix[int(coord[1]/game.width), int(coord[0]/game.height)] = 1
        for food in game.food:
            game_matrix[int(food.y_food/game.width), int(food.x_food/game.height)] = 2
        for i in range(game.width+2):
            for j in range(game.height+2):
                if i == 0 or j == 0 or i == game.width+1 or j == game.height+1:
                    game_matrix[i, j] = 1
        head = player.position[-1]
        player_x, player_y = int(head[0]/game.width), int(head[1]/game.height)

        state = [
            player_x + 1 < game.width+2 and game_matrix[player_y, player_x+1] == 1,  # danger right
            player_x + -1 >= 0 and game_matrix[player_y, player_x-1] == 1,  # danger left
            player_y + -1 >= 0 and game_matrix[player_y-1, player_x] == 1,  # danger up
            player_y + 1 < game.height+2 and game_matrix[player_y+1, player_x] == 1,  # danger down
            player.direction == player.right,
            player.direction == player.left,
            player.direction == player.up,
            player.direction == player.down,
            food.x_food < player.x,  # food left
            food.x_food > player.x,  # food right
            food.y_food < player.y,  # food up
            food.y_food > player.y  # food down
            ]
        for i in range(len(state)):
            if state[i]:
                state[i] = 1
            else:
                state[i] = 0
        return np.asarray(state).reshape(1, self.dim_state)

    def get_weights(self):
        return self.model.get_weights()

    def set_model(self, model):
        self.model = model


# create a keras model given the layers' structure and weights matrix
def create_model_from_units(units, weights):
    if len(units) < 2:
        print("Error: A model has to have at least 2 layers (input and output layer)")
        return None
    model = Sequential()
    added_weights = 0
    layers = len(units)  # considering input layer and first hidden layer are created at the same time
    for i in range(1, layers):
        activation = 'relu'
        if i == layers-1:
            activation = 'softmax'
        if i == 1:
            model.add(Dense(output_dim=units[i], activation=activation, input_dim=units[0]))
        else:
            model.add(Dense(output_dim=units[i], activation=activation))
        weight = weights[added_weights:added_weights+units[i-1]*units[i]].reshape(units[i-1], units[i])
        added_weights += units[i-1]*units[i]
        model.layers[-1].set_weights((weight, np.zeros(units[i])))
    return model


# calculating the fitness value by playing a game with the given weights in snake
def cal_pop_fitness(new_population, units, population):
    fitness = []
    deaths = []
    avg_score = []
    max_scores = []
    for i in range(population):
        K.clear_session()
        weights = new_population[i]
        model = create_model_from_units(units, weights)
        fit, snake_deaths, snake_avg_score, record = run_game(model)
        snake_avg_score = round(snake_avg_score, 2)
        print('fitness value of snake ' + str(i) + ':  ' + str(fit) +
              '   Deaths: ' + str(snake_deaths) + '   Avg score: ' + str(snake_avg_score) + '   Record: ' + str(record))
        fitness.append(fit)
        deaths.append(snake_deaths)
        avg_score.append(snake_avg_score)
        max_scores.append(record)
    return np.array(fitness), np.array(deaths), np.array(avg_score), np.array(max_scores)


# selecting the best individuals in the current generation as parents for producing the offspring of the next generation
def select_mating_pool(pop, fitness, num_parents):
    temp_fitness = np.array(fitness, copy=True)
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(temp_fitness == np.max(temp_fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        temp_fitness[max_fitness_idx] = -99999999
    return parents


# creating children for next generation
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if random.uniform(0, 1) < 0.5:
                        offspring[k, j] = parents[parent1_idx, j]
                    else:
                        offspring[k, j] = parents[parent2_idx, j]
                break
    return offspring


# mutating the offsprings generated from crossover to maintain variation in the population
def mutation(offspring_crossover, mutations=1):
    for idx in range(offspring_crossover.shape[0]):
        for _ in range(mutations):
            i = random.randint(0, offspring_crossover.shape[1]-1)
            random_value = np.random.choice(np.arange(-1, 1, step=0.001), size=1, replace=False)
            offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value
    return offspring_crossover


# get the number of weights given a model's layer structure
def weights_size(units):
    s = 0
    for i in range(len(units)-1):
        s += units[i] * units[i+1]
    return s


# get the matrix weigths of the best snake of a specific generation and population
def best_snake_weights(population_name, generation, pop_size):
    path = "weights/genetic_algorithm/" + str(population_name) + "/stats.txt"
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    last = lines[-1]
    best_snake = last.split(" ")[-2]
    weights_path = "weights/genetic_algorithm/" + str(population_name) + "/generation_" + str(generation) + ".txt"
    population = np.loadtxt(weights_path)
    population.reshape(pop_size)
    weights = population[int(best_snake)]
    return weights


# use a model to play a game and return model fitness score
def run_game(snake):
    # parameters
    steps_per_game = 5000
    max_steps_per_food = 200

    steps = 0
    game = Game()
    game.player.append(Player(game, "green"))
    ga_agent = GAAgent(training=True)  # only used to get the state
    player = game.player[0]
    game.food.append(Food(game))
    food = game.food[0]
    game.game_speed = 0
    player.init_player(game)
    current_step = 0
    slow_penalty = 0
    while steps <= steps_per_game:
        state = ga_agent.get_state(game, player, food)
        prediction = snake.model.predict(state)
        move = np.argmax(prediction[0])
        player.do_move(move, game)
        current_step += 1
        if player.eaten:
            current_step = 0
        if current_step > max_steps_per_food:
            player.crash = True
            slow_penalty += 1
        if player.crash:
            player.init_player(game)
            current_step = 0
        steps += 1
    return player.deaths * (-150) + player.record * 5000 + slow_penalty * (-1000) + int(steps_per_game / (player.total_score + 1)) * (-100), \
           player.deaths, player.total_score / (player.deaths + 1), player.record


# create np weights matrix from keras model (hdf5 file) (exclude biases)
def matrix_weights_from_keras(path, pop_size):
    model = Sequential()
    model.add(Dense(output_dim=120, activation='relu', input_dim=12))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Dense(output_dim=120, activation='relu'))
    model.add(Dense(output_dim=4, activation='softmax'))
    model.load_weights(path)
    print(model.get_weights())
    keras_weights = []
    for i in range(4):
        weights = model.layers[i].get_weights()
        keras_weights.append(np.array(weights[0]).flatten())
    keras_weights = np.concatenate(np.array(keras_weights))
    keras_weights = np.tile(keras_weights, pop_size[0]).reshape(pop_size)
    return keras_weights


# return the history of the training stats as arrays
def get_stats_as_history(population_name):
    path = "weights/genetic_algorithm/" + str(population_name) + "/stats.txt"
    f = open(path, "r")
    lines = f.readlines()
    f.close()
    lines = lines[1::2]
    max_fitness = []
    max_avg_score = []
    avg_fitness = []
    avg_deaths = []
    avg_score = []
    max_score = []
    for line in lines:
        stats = line.split(" ")
        max_fitness.append(int(stats[0]))
        max_avg_score.append(float(stats[1]))
        avg_fitness.append(float(stats[2]))
        avg_deaths.append(float(stats[3]))
        avg_score.append(float(stats[4]))
        max_score.append(int(stats[5]))
    return max_fitness, max_avg_score, avg_fitness, avg_deaths, avg_score, max_score


# train model
if __name__ == "__main__":
    units = [12, 120, 120, 120, 4]  # no. of input units, no. of units in hidden layer n, no. of output units
    population = 50  # parameter: population
    num_weights = weights_size(units)  # weights of a single model (snake)
    pop_size = (population, num_weights)  # population size
    #  creating the initial weights
    new_population = np.random.choice(np.arange(-1, 1, step=0.01), size=pop_size, replace=True)
    num_generations = 100  # parameter: number of generations
    num_parents_mating = 12  # parameter: number of best parents selected for crossover
    mutations = 1  # parameter: number of weights to replace during mutation
    checkpoint = 5  # parameter: how many generations between saving weights
    population_name = "standard_population"  # parameter: name of the population
    current_gen = 60  # parameter: last finished generation

    # restore weights from previous generation
    restore_weights_from_txt = True
    if restore_weights_from_txt:
        path = "weights/genetic_algorithm/" + str(population_name) + "/generation_" + str(current_gen) + ".txt"
        new_population = np.loadtxt(path)

    # start training
    for generation in range(num_generations):
        # skip old generations
        if restore_weights_from_txt and generation <= current_gen:
            continue
        print('GENERATION ' + str(generation))
        # measuring the fitness of each snake in the population
        fitness, deaths, avg_score, max_scores = cal_pop_fitness(new_population, units, population)
        # print generation stats
        print('fittest snake in geneneration ' + str(generation) + ' : ', np.max(fitness))
        print('average fitness value in geneneration ' + str(generation) + ' : ', np.sum(fitness) / population)
        print('highest average score in geneneration ' + str(generation) + ' : ', np.max(avg_score))
        print('average deaths in geneneration ' + str(generation) + ' : ', np.sum(deaths) / population)
        print('average score in geneneration ' + str(generation) + ' : ', np.sum(avg_score) / population)
        print('max score in geneneration ' + str(generation) + ' : ', max_scores[np.argmax(max_scores)])

        # selecting the best parents in the population for mating
        parents = select_mating_pool(new_population, fitness, num_parents_mating)
        # generating next generation using crossover
        offspring_crossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))
        # adding some variations to the offspring using mutation.
        offspring_mutation = mutation(offspring_crossover, mutations)
        # creating the new population based on the parents and offspring
        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation

        # save generation stats
        dir_path = "weights/genetic_algorithm/" + str(population_name) + "/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path = "weights/genetic_algorithm/" + str(population_name) + "/stats.txt"
        f = open(path, "a+")
        f.write(str(generation) + "\n")
        f.write(str(np.max(fitness)) + " " + str(np.max(avg_score)) + " " + str(np.sum(fitness)/population) + " " +
                str(np.sum(deaths) / population) + " " + str(np.sum(avg_score) / population) + " " +
                str(max_scores[np.argmax(max_scores)]) + " " + str(np.argmax(fitness)) + " \n")
        f.close()

        # save weights matrix
        if generation % checkpoint == 0 or generation == num_generations-1:
            path = "weights/genetic_algorithm/" + str(population_name) + "/generation_" + str(generation) + ".txt"
            np.savetxt(path, new_population)
            print("weights saved")
