from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np


class DQNAgent(object):

    def __init__(self, weights=False):
        self.dim_state = 12
        self.reward = 0
        self.gamma = 0.9
        self.learning_rate = 0.0005
        self.model = self.network()
        if weights:
            self.model = self.network(weights)
        self.epsilon = 0
        self.memory = []

    def get_state(self, game, player, food):

        game_matrix = np.zeros(shape=(22, 22))
        for player in game.player:
            for i, coord in enumerate(player.position):
                game_matrix[int(coord[1]/20), int(coord[0]/20)] = 1
        for food in game.food:
            game_matrix[int(food.y_food/20), int(food.x_food/20)] = 2
        for i in range(22):
            for j in range(22):
                if i == 0 or j == 0 or i == 21 or j == 21:
                    game_matrix[i, j] = 1
        head = player.position[-1]
        player_x, player_y = int(head[0]/20), int(head[1]/20)

        #print(game_matrix)

        state = [
            player_x + 1 < 22 and game_matrix[player_y, player_x+1] == 1,  # danger right
            player_x + -1 >= 0 and game_matrix[player_y, player_x-1] == 1,  # danger left
            player_y + -1 >= 0 and game_matrix[player_y-1, player_x] == 1,  # danger up
            player_y + 1 < 22 and game_matrix[player_y+1, player_x] == 1,  # danger down
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
        # state: [12] boolean array

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

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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

    def train_online_net(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1, self.dim_state)))[0])
        target_f = self.model.predict(state.reshape((1, self.dim_state)))
        target_f[0][np.argmax(action)] = target
        self.model.fit(state.reshape((1, self.dim_state)), target_f, epochs=1, verbose=0)
