import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d

import math
import random

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

humanWidth = 20
humanHeight = 100
humanLocation = Vec2d(200, 110 + humanHeight)

#text
pygame.font.init()
my_font = pygame.font.SysFont('Arial', 20)

class HumanDemo:
    def __init__(self):

        self.running = True
        self.drawing = True
        self.w, self.h = 600, 600
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        ### Init pymunk and create space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -9000.0)
        self.space.sleep_time_threshold = 0.3
        ### ground
        shape = pymunk.Segment(self.space.static_body, (-100, 100), (700, 100), 10.0)
        shape.friction = 1
        self.space.add(shape)

        
        points = [(-humanWidth, -humanHeight), (-humanWidth, humanHeight), (humanWidth, humanHeight), (humanWidth, -humanHeight)]
        mass = 1.0
        moment = pymunk.moment_for_poly(mass, points, (0, 0))
        body = pymunk.Body(mass, moment)
        body.position = humanLocation
        shape = pymunk.Poly(body, points)
        shape.friction = 1

        
        self.space.add(body, shape)
        self.human = body
        self.velocities = []
        self.angles = []


        ### draw options for drawing
        pymunk.pygame_util.positive_y_is_up = True
        pygame.display.set_caption("Fall Detection Simulator")
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # AI MODEL
        self.clf = joblib.load("SVM_MODEL.joblib")
    
    def predictFall(self):
        data = self.getSensorData()

        if (len(data) != 10):
            return 0

        if (int(self.clf.predict([data])[0])):
            return True

        return False

    def addLatestVelocityToStack(self, newVelocity):
        self.velocities.append(newVelocity)
        if (len(self.velocities) > 5):
            self.velocities.pop(0)

    def addLatestAngleToStack(self, newAngle):
        self.angles.append(newAngle)
        if (len(self.angles) > 5):
            self.angles.pop(0)

    def hasFallen(self):
        angle = self.human.angle * 57.3
        return ((angle > 88 and angle < 92) or (angle < -88 and angle > -92)) and self.velocities[-1] == 0

    def reset(self):
        self.human.position = humanLocation
        self.human.angle = 0
        self.human.velocity = (0, 0)

    def dumpDataToFile(self):
        all_zero = all(value < 1 for value in self.velocities)

        if (all_zero):
            return

        label =  '1' if self.hasFallen() else '0'
        velocitiesText = ' '.join([str(velocity) for velocity in self.velocities])
        anglesText = ' '.join([str(angle) for angle in self.angles])
        with open('data.txt', 'a') as f:
            f.write(label + ' A ' + velocitiesText + ' B ' + anglesText + "\n")

    def getSensorData(self):
        return self.velocities + self.angles

    def run(self):
        while self.running:
            self.loop()

    def loop(self):

        velocityX = round(self.human.velocity.x)
        velocityY = round(self.human.velocity.y)

        self.addLatestVelocityToStack((velocityX * velocityX + velocityY * velocityY) / 1000)
        self.addLatestAngleToStack(round(self.human.angle * 57.3))

        # Uncomment this if you want to collect data

        # if (self.hasFallen()):
        #     self.dumpDataToFile()
        #     self.reset()
        # elif random.random() < 0.01:
        #     # randomly dump data also
        #     self.dumpDataToFile()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r: #when r is pressed, reset the human
                    self.reset()
                elif not self.hasFallen():
                    if event.key == pygame.K_w:
                        impulse = (0, 1000)
                        point = (0, -humanHeight)
                        self.human.apply_impulse_at_local_point(impulse, point)
                    elif event.key == pygame.K_a:
                        impulse = (-100, 0)
                        point = (0, humanHeight)
                        self.human.apply_impulse_at_local_point(impulse, point)
                    elif event.key == pygame.K_d:
                        impulse = (100, 0)
                        point = (0, humanHeight)
                        self.human.apply_impulse_at_local_point(impulse, point)

        fps = 30.0
        dt = 1.0 / fps / 5
        self.space.step(dt)
        if self.drawing:
            self.draw()

        ### Tick clock and update fps in title
        self.clock.tick(fps)

    def draw(self):
        ### Clear the screen
        self.screen.fill(pygame.Color("white"))

        ### Draw space
        self.space.debug_draw(self.draw_options)

        # Print text
        self.screen.blit(my_font.render("Human velocity: " + str(round(self.velocities[-1])), True, (0, 0, 0)), (0,0))
        self.screen.blit(my_font.render("Human angle: " + str(round(self.angles[-1])), True, (0, 0, 0)), (0,30))
        self.screen.blit(my_font.render("Has fallen: " + str(self.hasFallen()), True, (0, 0, 0)), (0,60))
        self.screen.blit(my_font.render("Fall prediction: " + str(self.predictFall()), True, (0, 0, 0)), (0,90))

        ### All done, lets flip the display
        pygame.display.flip()


def main():
    demo = HumanDemo()
    demo.run()


if __name__ == "__main__":
    main()


