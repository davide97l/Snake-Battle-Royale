import pygame
from random import randint
from snake.DQNagent import DQNAgent
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import utils
import keyboard


class Game:

    def __init__(self, game_width=440, game_height=440, game_speed=30, display_option=True):
        pygame.display.set_caption('Snake')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")
        self.player = []
        self.food = []
        self.display_option = display_option
        self.game_speed = game_speed

    # return the coordinates of a location without snakes' parts or walls
    def find_free_space(self):
        x_rand = randint(20, self.game_width - 40)
        x = x_rand - x_rand % 20
        y_rand = randint(20, self.game_height - 40)
        y = y_rand - y_rand % 20
        for player in self.player:
            if [x, y] not in player.position:
                return x, y
            else:
                return self.find_free_space()


class Player(object):

    def __init__(self, game, color="green", weights=None):
        self.color = color
        if self.color == "green":
            self.image = pygame.image.load('img/snakeBody1.png')
            x = 0.3 * game.game_width
            y = 0.3 * game.game_height
        if self.color == "blue":
            self.image = pygame.image.load('img/snakeBody2.png')
            x = 0.3 * game.game_width
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
        self.weights = weights  # None = standard algorithm, else = RL algorithm
        self.agent = None
        if weights:
            self.agent = DQNAgent(weights)

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
        if x < 20 or x > game.game_width - 40\
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
            update_screen()
        else:
            game.end = True

    def euristic_move(self, game):
        distance = []
        for food in game.food:
            distance.append(abs(self.x - food.x_food) + abs(self.y - food.y_food))
        food = game.food[np.argmin(distance)]

        reward = [-10000, -10000, -10000, -10000]
        
        def get_action_score(x, y):
            if self.crushed(game, x, y):
                return -1000
            return 1000-(abs(x - food.x_food) + abs(y - food.y_food))

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

    def RL_move(self, game):
        distance = []
        for food in game.food:
            distance.append(abs(self.x - food.x_food) + abs(self.y - food.y_food))
        food = game.food[np.argmin(distance)]
        state = self.agent.get_state(game, self, food)
        prediction = self.agent.model.predict(state)
        move = np.argmax(prediction[0])
        return move

    def select_move(self, game):
        if not self.weights:
            return self.euristic_move(game)
        return self.RL_move(game)

    def move_as_array(self, move):
        if move == self.right:
            return [1, 0, 0, 0]
        elif move == self.left:
            return [0, 1, 0, 0]
        elif move == self.up:
            return [0, 0, 1, 0]
        elif move == self.down:
            return [0, 0, 0, 1]


class Food(object):

    def __init__(self, game):
        self.x_food, self.y_food = 0, 0
        self.food_coord(game)
        self.image = pygame.image.load('img/food2.png')

    def food_coord(self, game):
        self.x_food, self.y_food = game.find_free_space()

    def display_food(self, game):
        game.gameDisplay.blit(self.image, (self.x_food, self.y_food))
        update_screen()


def display_ui(game):
    myfont = pygame.font.SysFont('Segoe UI', 20)

    text_score1 = myfont.render('Green Snake Score: ' + str(game.player[0].score), True, (0, 0, 0))
    text_highest1 = myfont.render('Record: ' + str(game.player[0].record), True, (0, 0, 0))
    avg = game.player[0].total_score / (game.player[0].deaths + 1)
    text_avg1 = myfont.render('Avg: ' + str(round(avg)), True, (0, 0, 0))
    game.gameDisplay.blit(text_score1, (35, 440))
    game.gameDisplay.blit(text_highest1, (230, 440))
    game.gameDisplay.blit(text_avg1, (340, 440))

    if len(game.player) > 1:
        text_score2 = myfont.render('Blue Snake Score: ' + str(game.player[1].score), True, (0, 0, 0))
        text_highest2 = myfont.render('Record: ' + str(game.player[1].record), True, (0, 0, 0))
        avg = game.player[1].total_score / (game.player[1].deaths + 1)
        text_avg2 = myfont.render('Avg: ' + str(round(avg)), True, (0, 0, 0))
        game.gameDisplay.blit(text_score2, (35, 460))
        game.gameDisplay.blit(text_highest2, (230, 460))
        game.gameDisplay.blit(text_avg2, (340, 460))

    game.gameDisplay.blit(game.bg, (10, 10))


def display(game):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game)
    for player in game.player:
        player.display_player(game)
    for food in game.food:
        food.display_food(game)


def update_screen():
    pygame.display.update()


def run_snake():
    pygame.init()
    pygame.font.init()
    game = Game(440, 440)
    game.player.append(Player(game, "green"))
    game.player.append(Player(game, "blue", 'weights/weights_snake_300.hdf5'))
    game.food.append(Food(game))

    game.game_speed = 0  # parameter: game speed
    record = True  # parameter: True if recording the game
    frames = []

    if game.display_option:
        display(game)
    while not keyboard.is_pressed('s'):
        for i in range(len(game.player)):
            move = game.player[i].select_move(game)
            game.player[i].do_move(move, game)
            if game.player[i].crash:
                game.player[i].init_player(game)

        if game.display_option:
            display(game)
            pygame.time.wait(game.game_speed)
            if record:
                data = pygame.image.tostring(game.gameDisplay, 'RGBA')
                from PIL import Image
                img = Image.frombytes('RGBA', (440, 500), data)
                img = img.convert('RGB')
                frames.append(np.array(img))
    utils.save_animation(frames, 'videos/snake.mp4', 25)


def train_snake():
    pygame.init()
    new_agent = False  # parameter: True if training a new agent
    training = False  # parameter: True if training an agent
    quick_test = False  # parameter: fast testing
    counter_games = 300  # parameter: number of iterations the agent has been trained
    max_games = 500  # parameter: max number of iterations to train the agent
    checkpoint = 100  # parameter: number of iterations between the which save the weights
    weights_name = 'weights/weights_snake_'  # parameter: name of the weights
    weights = weights_name + str(counter_games) + '.hdf5'
    agent = DQNAgent()
    if not new_agent:
        agent = DQNAgent(weights)
    game = Game(440, 440)
    game.player.append(Player(game, "green"))
    game.food.append(Food(game))
    score_plot = []
    counter_plot = []
    if training or quick_test:
        game.display_option = False
        game.game_speed = 0
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
                display(game)
                pygame.time.wait(game.game_speed)

            # prevents loops
            step_counter += 1
            if step_counter >= 10000:
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
    print("Average: " + str(game.player[0].total_score / (game.player[0].deaths + 1)))  # 27.527638190954775
    print("Max: " + str(max_score))  # 52


#train_snake()
run_snake()
