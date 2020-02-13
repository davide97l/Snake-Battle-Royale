import numpy as np
import pygame


class Player(object):

    def __init__(self, game, color="green", ai="heuristic", depth=10):
        self.color = color
        if self.color == "green":
            self.image = pygame.image.load('img/snakeBody1.png')
            x = 0.3 * game.game_width
            y = 0.3 * game.game_height
        if self.color == "blue":
            self.image = pygame.image.load('img/snakeBody2.png')
            x = 0.3 * game.game_width
            y = 0.7 * game.game_height
        if self.color == "red":
            self.image = pygame.image.load('img/snakeBody3.png')
            x = 0.7 * game.game_width
            y = 0.3 * game.game_height
        if self.color == "purple":
            self.image = pygame.image.load('img/snakeBody4.png')
            x = 0.7 * game.game_width
            y = 0.7 * game.game_height
        self.x = x - x % 20
        self.y = y - y % 20
        self.position = []  # coordinates of all the parts of the snake
        self.position.append([self.x, self.y])  # append the head
        self.food = 1  # length
        self.eaten = False
        self.right = 0
        self.left = 1
        self.up = 2
        self.down = 3
        self.direction = self.right
        self.step_size = 20  # pixels per step
        self.crash = False
        self.score = 0
        self.record = 0
        self.deaths = 0
        self.total_score = 0  # accumulated score
        self.agent = None
        self.ai = ai  # options: heuristic, deep_search
        if ai == "deepSearch":
            self.depth = depth  # search range for deep_search move

    def init_player(self, game):
        self.x, self.y = game.find_free_space()
        self.position = []
        self.position.append([self.x, self.y])
        self.food = 1
        self.eaten = False
        self.direction = self.right
        self.step_size = 20
        self.crash = False
        self.score = 0

    def update_position(self):
        if self.position[-1][0] != self.x or self.position[-1][1] != self.y:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = self.x
            self.position[-1][1] = self.y

    def eat(self, game):
        for food in game.food:
            if self.x == food.x_food and self.y == food.y_food:
                food.food_coord(game)
                self.eaten = True
                self.score += 1
                self.total_score += 1

    def crushed(self, game, x=-1, y=-1):
        if x == -1 and y == -1:  # coordinates of the head
            x = self.x
            y = self.y
        if x < 20 or x > game.game_width - 40 \
                or y < 20 or y > game.game_height - 40:
            return True
        for player in game.player:
            if [x, y] in player.position:
                return True

    def do_move(self, move, game):
        if self.eaten:
            self.position.append([self.x, self.y])
            self.eaten = False
            self.food = self.food + 1

        move_array = [0, 0]

        if move == self.right:
            move_array = [self.step_size, 0]
        elif move == self.left:
            move_array = [-self.step_size, 0]
        elif move == self.up:
            move_array = [0, -self.step_size]
        elif move == self.down:
            move_array = [0, self.step_size]

        if move == self.right and self.direction != self.left:
            move_array = [self.step_size, 0]
            self.direction = self.right
        elif move == self.left and self.direction != self.right:
            move_array = [-self.step_size, 0]
            self.direction = self.left
        elif move == self.up and self.direction != self.down:
            move_array = [0, -self.step_size]
            self.direction = self.up
        elif move == self.down and self.direction != self.up:
            move_array = [0, self.step_size]
            self.direction = self.down
        self.x += move_array[0]
        self.y += move_array[1]

        if self.crushed(game):
            self.crash = True
            self.deaths += 1
            if self.score > self.record:
                self.record = self.score

        self.eat(game)
        self.update_position()

    def display_player(self, game):
        self.position[-1][0] = self.x
        self.position[-1][1] = self.y

        if not self.crash:
            for i in range(self.food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))
            pygame.display.update()
        else:
            game.end = True

    def heuristic_move(self, game):
        distance = []
        for food in game.food:
            distance.append(abs(self.x - food.x_food) + abs(self.y - food.y_food))
        food = game.food[np.argmin(distance)]

        reward = [-10000, -10000, -10000, -10000]

        def get_action_score(x, y):
            if self.crushed(game, x, y):
                return -1000
            return 1000 - (abs(x - food.x_food) + abs(y - food.y_food))

        if self.direction != self.left:
            move_array = [self.step_size, 0]
            reward[self.right] = get_action_score(self.x + move_array[0], self.y + move_array[1])
        if self.direction != self.right:
            move_array = [-self.step_size, 0]
            reward[self.left] = get_action_score(self.x + move_array[0], self.y + move_array[1])
        if self.direction != self.down:
            move_array = [0, -self.step_size]
            reward[self.up] = get_action_score(self.x + move_array[0], self.y + move_array[1])
        if self.direction != self.up:
            move_array = [0, self.step_size]
            reward[self.down] = get_action_score(self.x + move_array[0], self.y + move_array[1])
        return np.argmax(reward)

    def deep_search_move(self, state, game, player_x, player_y, depth=10, initial_depth=10):
        if depth == initial_depth:
            state[player_y, player_x] = 0
        if depth == 0:
            return 40 - self.distance_closest_food(game, player_x, player_y), -1
        if state[player_y, player_x] == 1:  # death
            return -1000 + (initial_depth - depth), -1
        add_reward = 0
        if state[player_y, player_x] == 2:  # food
            add_reward = 100 - (initial_depth - depth)

        if self.food > 20:
            if state[player_y, player_x+1] == 1 and state[player_y, player_x-1] == 1:
                add_reward -= 80
            if state[player_y+1, player_x] == 1 and state[player_y-1, player_x] == 1:
                add_reward -= 80

        reward = np.array([-1000, -1000, -1000, -1000])  # right, left, up, down

        old_state_value = state[player_y, player_x]
        state[player_y, player_x] = 1
        reward[self.right], _ = self.deep_search_move(state, game, player_x+1, player_y, depth-1, initial_depth)
        reward[self.left], _ = self.deep_search_move(state, game,  player_x-1, player_y, depth-1, initial_depth)
        reward[self.up], _ = self.deep_search_move(state, game,  player_x, player_y-1, depth-1, initial_depth)
        reward[self.down], _ = self.deep_search_move(state, game,  player_x, player_y+1, depth-1, initial_depth)
        state[player_y, player_x] = old_state_value

        return reward[np.argmax(reward)] + add_reward, np.argmax(reward)

    def network_move(self, game):
        distance = []
        for food in game.food:
            distance.append(abs(self.x - food.x_food) + abs(self.y - food.y_food))
        food = game.food[np.argmin(distance)]
        state = self.agent.get_state(game, self, food)
        prediction = self.agent.model.predict(state)
        move = np.argmax(prediction[0])
        return move

    def select_move(self, game):
        if self.agent and (self.agent.name == "ga" or self.agent.name == "dqn"):
            return self.network_move(game)
        if self.ai == "heuristic":
            return self.heuristic_move(game)
        player_x, player_y = game.get_player_coord(self)
        if self.ai == "deepSearch":
            return self.deep_search_move(game.get_matrix_state(), game, player_x, player_y,
                                     depth=self.depth, initial_depth=self.depth)[1]

    def move_as_array(self, move):
        if move == self.right:
            return [1, 0, 0, 0]
        elif move == self.left:
            return [0, 1, 0, 0]
        elif move == self.up:
            return [0, 0, 1, 0]
        elif move == self.down:
            return [0, 0, 0, 1]

    def set_agent(self, agent):
        self.agent = agent

    def distance_closest_food(self, game, x, y):
        distance = []
        for food in game.food:
            distance.append(abs(x - food.x_food/self.step_size) + abs(y - food.y_food/self.step_size))
        return distance[np.argmin(distance)]
