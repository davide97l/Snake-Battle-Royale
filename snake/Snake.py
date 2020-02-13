import pygame
from snake.DQNagent import DQNAgent
from snake.GAagent import GAAgent
import numpy as np
import utils
import keyboard
from snake.Game import Game
from snake.Food import Food
from snake.Player import Player


def run_snake():
    pygame.init()
    pygame.font.init()
    game = Game(20, 20)

    snake_green = Player(game, "green", ai="heuristic")
    game.player.append(snake_green)
    snake_blu = Player(game, "blue")
    game.player.append(snake_blu)
    snake_red = Player(game, "red")
    game.player.append(snake_red)
    snake_purple = Player(game, "purple", ai="deepSearch", depth=8)
    game.player.append(snake_purple)

    game.food.append(Food(game))

    rl_agent = DQNAgent('weights/weights_snake_300.hdf5')
    snake_blu.set_agent(rl_agent)
    ga_agent = GAAgent(population_name="standard_population", generation=100)
    snake_red.set_agent(ga_agent)

    game.game_speed = 0  # parameter: game speed
    game.display_option = True  # parameter: show game
    record = True  # parameter: True if recording the game
    frames = []

    while not keyboard.is_pressed('s'):
        for i in range(len(game.player)):
            move = game.player[i].select_move(game)
            game.player[i].do_move(move, game)
            if game.player[i].crash:
                game.player[i].init_player(game)

        if game.display_option:
            game.display()
            pygame.time.wait(game.game_speed)
            if record:
                data = pygame.image.tostring(game.gameDisplay, 'RGBA')
                from PIL import Image
                img = Image.frombytes('RGBA', (game.game_width, game.game_height + 100), data)
                img = img.convert('RGB')
                frames.append(np.array(img))
    utils.save_animation(frames, 'videos/snake.mp4', 25)
    for i, snake in enumerate(game.player):
        print("Snake" + str(i+1) + " Max score: " + str(snake.record) +
              ", Avg Score: " + str(snake.total_score / snake.deaths) +
              ", Deaths: " + str(snake.deaths))


if __name__ == "__main__":
    run_snake()
