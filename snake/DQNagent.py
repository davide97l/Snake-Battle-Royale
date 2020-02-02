from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import pygame
from random import randint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from snake.Game import Game
from snake.Food import Food
from snake.Player import Player


class DQNAgent(object):

    def __init__(self, weights=False, dim_state=12, gamma=0.9, learning_rate=0.0005):
        self.dim_state = dim_state
        self.reward = 0
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self.network()
        if weights:
            self.model = self.network(weights)
        self.memory = []
        self.name = "dqn"

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

        #print(game_matrix)

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

    def set_reward(self, player):
        self.reward = 0
        if player.crash:
            self.reward = -10
            return self.reward
        if player.eaten:
            self.reward = 10
        return self.reward

    def network(self, weights=None):
        model = Sequential()
        model.add(Dense(output_dim=120, activation='relu', input_dim=self.dim_state))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=120, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(output_dim=4, activation='softmax'))  # [right, left, up, down]
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        if weights:
            model.load_weights(weights)
        return model

    # only for training
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # only for training
    def replay_mem(self, memory):
        if len(memory) > 1000:
            minibatch = random.sample(memory, 1000)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]).reshape((1, self.dim_state)))[0])
            target_f = self.model.predict(np.array([state]).reshape((1, self.dim_state)))
            target_f[0][np.argmax(action)] = target
            self.model.fit(np.array([state]).reshape((1, self.dim_state)), target_f, epochs=1, verbose=0)

    # only for training
    def train_online_net(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.dim_state)))[0])
        target_f = self.model.predict(state.reshape((1, self.dim_state)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, self.dim_state)), target_f, epochs=1, verbose=0)

    def get_weights(self):
        return self.model.get_weights()

    def set_model(self, model):
        self.model = model


# train model
if __name__ == "__main__":
    pygame.init()
    new_agent = False  # parameter: True if training a new agent
    training = False  # parameter: True if training an agent
    quick_test = True  # parameter: fast testing
    counter_games = 300  # parameter: number of iterations the agent has been trained
    max_games = 300  # parameter: max number of iterations to train the agent
    checkpoint = 100  # parameter: number of iterations between the which save the weights
    weights_name = 'weights/weights_snake_'  # parameter: name of the weights
    weights = weights_name + str(counter_games) + '.hdf5'
    agent = DQNAgent()
    if not new_agent:
        agent = DQNAgent(weights)
    game = Game(20, 20)
    game.player.append(Player(game, "green"))
    game.food.append(Food(game))
    score_plot = []
    counter_plot = []
    game.display_option = True
    game.game_speed = 30
    if training or quick_test:
        game.display_option = False
        game.game_speed = 0
    max_score = 0
    while counter_games <= max_games:

        game.player[0].init_player(game)
        step_counter = 0
        while not game.player[0].crash:
            # agent.epsilon is set to give randomness to actions
            agent.epsilon = 80 - counter_games
            if agent.epsilon < 1:
                agent.epsilon = 1
            if not training:
                agent.epsilon = 0

            # get old state
            state_old = agent.get_state(game, game.player[0], game.food[0])

            #print(state_old)
            #input("pause")

            # perform random actions based on agent.epsilon, or choose the action
            if randint(0, 200) < agent.epsilon:
                move = randint(0, 3)
            else:
                # predict action based on the old state
                prediction = agent.model.predict(state_old)
                move = np.argmax(prediction[0])

            action = game.player[0].move_as_array(move)

            # perform new move and get new state
            game.player[0].do_move(move, game)
            state_new = agent.get_state(game, game.player[0], game.food[0])

            # set reward for the new state
            reward = agent.set_reward(game.player[0])

            if training:
                # train short memory base on the new action and state
                agent.train_online_net(state_old, action, reward, state_new, game.player[0].crash)

            # store the new data into a long term memory
            agent.remember(state_old, action, reward, state_new, game.player[0].crash)
            if game.display_option:
                game.display()
                pygame.time.wait(game.game_speed)

            # prevents loops
            step_counter += 1
            if game.player[0].eaten:
                step_counter = 0
            if step_counter >= 1000:
                game.player[0].crash = True
            if game.player[0].score > max_score:
                max_score = game.player[0].score

        if training:
            agent.replay_mem(agent.memory)
        counter_games += 1
        print('Game', counter_games, '      Score:', game.player[0].score)
        score_plot.append(game.player[0].score)
        counter_plot.append(counter_games)
        if training and counter_games % checkpoint == 0:
            weights = weights_name + str(counter_games) + '.hdf5'
            agent.model.save_weights(weights)
    if training:
        sns.set(color_codes=True)
        ax = sns.regplot(np.array([counter_plot])[0], np.array([score_plot])[0], color="b", x_jitter=.1,
                         line_kws={'color': 'green'})
        ax.set(xlabel='games', ylabel='score')
        plt.show()
    print("Average: " + str(game.player[0].total_score / game.player[0].deaths))  # 27.527638190954775
    print("Max: " + str(max_score))  # 52

