import pygame
from random import randint


class Game:

    def __init__(self, width=20, height=20, game_speed=30, display_option=True):
        self.game_width = width * 20 + 40
        self.game_height = height * 20 + 40
        self.width = width
        self.height = height
        if display_option:
            self.gameDisplay = pygame.display.set_mode((self.game_width, self.game_height + 80))
            self.bg = pygame.image.load("img/background.png")
            pygame.display.set_caption('Snake')
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

    def display_ui(self):
        myfont = pygame.font.SysFont('Segoe UI', 20)

        for i in range(len(self.player)):
            color = ""
            if self.player[i].color == "green":
                color = "Green"
            if self.player[i].color == "blue":
                color = "Blue"
            if self.player[i].color == "red":
                color = "Red"
            text_score = myfont.render(color + ' Snake Score: ' + str(self.player[i].score), True, (0, 0, 0))
            text_highest = myfont.render('Record: ' + str(self.player[i].record), True, (0, 0, 0))
            avg = self.player[i].total_score / (self.player[i].deaths + 1)
            text_avg = myfont.render('Avg: ' + str(round(avg)), True, (0, 0, 0))
            self.gameDisplay.blit(text_score, (35, 440 + i*20))
            self.gameDisplay.blit(text_highest, (230, 440 + i*20))
            self.gameDisplay.blit(text_avg, (340, 440 + i*20))

        self.gameDisplay.blit(self.bg, (10, 10))

    def display(self):
        if self.display_option:
            self.gameDisplay.fill((255, 255, 255))
            self.display_ui()
            for player in self.player:
                player.display_player(self)
            for food in self.food:
                food.display_food(self)
