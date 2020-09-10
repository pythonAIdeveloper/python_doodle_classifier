"""
press t to train
press g to guess
press e to test
"""

import nn
import random
import time

# PREPARING DATA

t0 = time.time()
a = open("cats1000.bin", "rb")
b = open("rainbows1000.bin", "rb")
c = open("trains1000.bin", "rb")
cats_data = a.read()
rainbows_data = b.read()
trains_data = c.read()
a.close()
b.close()
c.close()

length = 784
totalData = 1000

class doodle():
    def __init__(self, data, index):
        self.training = []
        self.testing = []
        self.data = data
        self.index = index

cats = doodle(cats_data, 0)
rainbows = doodle(rainbows_data, 1)
trains = doodle(trains_data, 2)

threshold = 0.8 * totalData

def prepareData(category):
    for i in range(totalData):
        offset = i * length
        if i < threshold:
            for j in range(length):
                category.training.append(category.data[offset + j]/255)
        else:
            for k in range(length):
                category.testing.append(category.data[offset + k]/255)

# continuous values array
prepareData(cats)
prepareData(rainbows)
prepareData(trains)

def dataRefiner(data, index):
    a = []
    for i in range(int(len(data)/length)):
        c = []
        b = []
        for j in range(length):
            b.append(data[j + i*length])
        c.append(b)
        c.append([index])
        a.append(c)
    return a

p = dataRefiner(cats.training, 0)
q = dataRefiner(rainbows.training, 1)
r = dataRefiner(trains.training, 2)
s = dataRefiner(cats.testing, 0)
t = dataRefiner(rainbows.testing, 1)
u = dataRefiner(trains.testing, 2)

# alltogether
trainingMain = []
testingMain = []
for i in p:
    trainingMain.append(i)
for i in q:
    trainingMain.append(i)
for i in r:
    trainingMain.append(i)
for i in s:
    testingMain.append(i)
for i in t:
    testingMain.append(i)
for i in u:
    testingMain.append(i)

random.shuffle(trainingMain)
random.shuffle(testingMain)

# DATA PREPARED

t1 = time.time()
print(f"data prepared \ntime taken -> {t1-t0}\n")

# MAKING THE NEURAL NETWORK

doodleClassifier = nn.SingleLayerNewralNetwork(784, 256, 3)

t2 = time.time()
print(f"DoodleClassifier made \ntime taken -> {t2-t1}\n")

# trainDoodle
Epoch = 0
def trainDoodle():
    t1 = time.time()
    global Epoch
    Epoch += 1
    print("\ntraining doodleClassifier...")
    num = 0
    for i in trainingMain:
        print(f"train {num}")
        num += 1
        t = []
        if i[1] == [0]:
            t = [1, 0, 0]
        elif i[1] == [1]:
            t = [0, 1, 0]
        elif i[1] == [2]:
            t = [0, 0, 1]

        doodleClassifier.train(i[0], t)

    print(f"Epoch{Epoch} complete")
    t2 = time.time()
    print(f"time take -> {t2-t1}")

def testDoodle():

    t1 = time.time()

    print("\ntesting doodle classifier...")
    b = 0
    for i in testingMain:
        if i[1] == [0]:
            t = [1, 0, 0]
        elif i[1] == [1]:
            t = [0, 1, 0]
        else:
            t = [0, 0, 1]
        a = doodleClassifier.feedForward(i[0])
        biggest = a[0]
        g = a.index(max(a))

        c = [0, 0]

        c.insert(g, 1)

        if c == t:
            b += 1

    print(f"{b * 100 / len(testingMain)}%")
    t2 = time.time()
    print(f"time taken -> {t2 - t1}")

# NEURAL NETWORK MADE

"""
# testDoodle()
# t3 = time.time()
# print(f"time taken -> {t3-t2}\n")
#
# trainDoodle()
# t4 = time.time()
# print(f"time taken -> {t4-t3}\n")
#
# testDoodle()
# t5 = time.time()
# print(f"time taken -> {t5-t4}\n")
#
# trainDoodle()
# t6 = time.time()
# print(f"time taken -> {t6-t5}\n")
#
# testDoodle()
# t7 = time.time()
# print(f"time taken -> {t7-t6}\n")
"""

#drawing interface

print("press r to train\npress g to guess\npress e to test\ndraw using mouse\npress w to clear canvas\n")

# MAKING THE GUI

import pygame
from pygame.locals import *
pygame.init()

screen = pygame.display.set_mode((280, 280))
screen.fill((255, 255, 255))

pygame.display.update()
clock = pygame.time.Clock()

z = 0
while True:

    clock.tick(1000)
    x, y = pygame.mouse.get_pos()

    for event in pygame.event.get():

        if z == 1:
            pygame.draw.circle(screen, (0, 0, 0), (x, y), 7)
            pygame.display.update()

        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
            break

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_r:
                trainDoodle()

            elif event.key == pygame.K_w:
                pygame.draw.rect(screen, (255, 255, 255), (0, 0, 280, 280))

            elif event.key == pygame.K_g:
                t3 = time.time()
                Pixels = []

                for i in range(280):
                    for j in range(280):
                        Pixels.append(screen.get_at([i, j]))

                PixelsFinal = []

                for i in Pixels:
                    b = 1 - ((i[0] + i[1] + i[2]) / 3) / 255
                    PixelsFinal.append(b)

                PixelsFinal1 = []

                for i in range(int(len(PixelsFinal)/280)):
                    PixelsFinal1Maker = []
                    for j in range(280):
                          PixelsFinal1Maker.append(PixelsFinal[i * 280 + j])
                    PixelsFinal1.append(PixelsFinal1Maker)

                PixelsFinal2 = []

                for i in (PixelsFinal1):
                    a = []
                    for j in range(int(len(PixelsFinal1)/10)):
                        b = []
                        for k in range(10):
                            b.append(i[j * 10 + k])
                        a.append(b)
                    PixelsFinal2.append(a)

                PixelsFinal3 = []

                for i in range(len(PixelsFinal2)):
                    a6 = []
                    for j in range(len(PixelsFinal2[0])):
                        sum = 0
                        for k in range(len(PixelsFinal2[0][0])):
                            sum += PixelsFinal2[i][j][k]
                        sum /= len(PixelsFinal2[0][0])
                        a6.append(sum)
                    PixelsFinal3.append(a6)

                pixels28one = []

                for i in range(len(PixelsFinal3[0])):
                    for j in range(len(PixelsFinal3[0])):
                        g = 0
                        for k in range(10):
                            g += PixelsFinal3[i * 10 + k][j]
                        g /= 10
                        pixels28one.append(g)

                pixels28twop = []

                for i in range(int(len(pixels28one)/28)):
                    a = []
                    for j in range(28):
                        a.append(pixels28one[i*28+j])
                    pixels28twop.append(a)

                pixels28two = []

                for i in range(28):
                    for j in range(28):
                        pixels28two.append(pixels28twop[j][i])

                t = doodleClassifier.feedForward(pixels28two)
                c = t.index(max(t))

                """
                r = doodleClassifier.feedForward(pixels28one)
                b = r.index(max(r))
                if b == 0:
                    print("\nit is a cat")
                if b == 1:
                    print("\nit is a rainbow")
                if b == 2:
                    print("\nit is a train")
                """

                if c == 0:
                    print("\nit is a cat")
                if c == 1:
                    print("\nit is a rainbow")
                if c == 2:
                    print("\nit is a train")

                t4 = time.time()
                print(f"time taken->{t4-t3}")

            elif event.key == pygame.K_e:
                testDoodle()

        elif event.type == MOUSEBUTTONDOWN:
            z = 1
        elif event.type == MOUSEBUTTONUP:
            z = 0

# GUI MADE