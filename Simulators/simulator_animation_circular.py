import pygame
import os
from pygame.locals import *
from libs import *

if os.getcwd() != 'C:\\Users\\Sachin Konan\\Documents\\PythonPractice':
    os.chdir('C:\\Users\\Sachin Konan\\Documents\\PythonPractice')

def showSys(sys, screen, start_x, start_y):
    arrowmag = 2

    for i in range(0, len(sys)):
        string= str(sys[i].val)
        width, height = font.size(string)
        text = font.render(string, True, BLACK)
        x = start_x+ rect_size*i + space_size*i
        y = start_y
        pygame.draw.rect(screen, GREEN, [x, y, rect_size, rect_size])
        if(i != len(sys) - 1):
            pygame.draw.line(screen, GREEN, [x+rect_size, y + int(rect_size/2)], [x+rect_size+space_size,  y + int(rect_size/2)], 1)
            pygame.draw.polygon(screen, BLACK, [ [x+rect_size+space_size,  y + int(rect_size/2)], [x+rect_size+space_size - arrowmag*np.cos(45*np.pi/180), y + int(rect_size/2) - arrowmag*np.sin(45*np.pi/180)], [x+rect_size+space_size - arrowmag*np.cos(45*np.pi/180),  y + int(rect_size/2) +arrowmag*np.sin(45*np.pi/180)] ])
        else:
            pygame.draw.line(screen, GREEN, [x+rect_size, y + int(rect_size/2)], [x+rect_size+int(space_size/2),  y + int(rect_size/2)], 1)
            pygame.draw.line(screen, GREEN,  [x+rect_size + int(space_size/2), y + int(rect_size/2)], [x+rect_size+int(space_size/2),  y + rect_size + int(space_size/2)])
            pygame.draw.line(screen, GREEN,  [x+rect_size+int(space_size/2),  y + rect_size + int(space_size/2)], [start_x - int(space_size//2),  y + rect_size + int(space_size/2)])

            pygame.draw.line(screen, GREEN,  [start_x - int(space_size//2),  y + rect_size + int(space_size/2)], [start_x - int(space_size//2) , start_y + int(rect_size/2)])
            pygame.draw.line(screen, GREEN,  [start_x - int(space_size//2) , start_y + int(rect_size/2)], [start_x, start_y + int(rect_size/2)])
            pygame.draw.polygon(screen, BLACK, [ [start_x, start_y + int(rect_size/2)], [start_x - arrowmag*np.cos(45*np.pi/180), start_y + int(rect_size/2) - arrowmag*np.sin(45*np.pi/180)], [start_x - arrowmag*np.cos(45*np.pi/180),  start_y + int(rect_size/2) +arrowmag*np.sin(45*np.pi/180)] ])

        screen.blit(text, (x + int((rect_size - width)/2 ), y + int((rect_size - height)/2) ) )
        if(sys[i].nval != None and sys[i].nval != 0):
            string= str(sys[i].nval)
            width, height = font.size(string)
            text = font.render(string, True, BLACK)
            screen.blit(text, (x + int((rect_size - width)/2 ), y + int((space_size - height)/2) - space_size ) )


def showSysIndividual(sys, screen, start_x, start_y, titty):
    arrowmag = 2

    for i in [titty]:
        string= str(sys[i].val)
        width, height = font.size(string)
        text = font.render(string, True, BLACK)
        x = start_x+ rect_size*i + space_size*i
        y = start_y
        pygame.draw.rect(screen, GREEN, [x, y, rect_size, rect_size])
        if(i != len(sys) - 1):
            pygame.draw.line(screen, GREEN, [x+rect_size, y + int(rect_size/2)], [x+rect_size+space_size,  y + int(rect_size/2)], 1)
            pygame.draw.polygon(screen, BLACK, [ [x+rect_size+space_size,  y + int(rect_size/2)], [x+rect_size+space_size - arrowmag*np.cos(45*np.pi/180), y + int(rect_size/2) - arrowmag*np.sin(45*np.pi/180)], [x+rect_size+space_size - arrowmag*np.cos(45*np.pi/180),  y + int(rect_size/2) +arrowmag*np.sin(45*np.pi/180)] ])
        else:
            pygame.draw.line(screen, GREEN, [x+rect_size, y + int(rect_size/2)], [x+rect_size+int(space_size/2),  y + int(rect_size/2)], 1)
            pygame.draw.line(screen, GREEN,  [x+rect_size + int(space_size/2), y + int(rect_size/2)], [x+rect_size+int(space_size/2),  y + rect_size + int(space_size/2)])
            pygame.draw.line(screen, GREEN,  [x+rect_size+int(space_size/2),  y + rect_size + int(space_size/2)], [start_x - int(space_size//2),  y + rect_size + int(space_size/2)])

            pygame.draw.line(screen, GREEN,  [start_x - int(space_size//2),  y + rect_size + int(space_size/2)], [start_x - int(space_size//2) , start_y + int(rect_size/2)])
            pygame.draw.line(screen, GREEN,  [start_x - int(space_size//2) , start_y + int(rect_size/2)], [start_x, start_y + int(rect_size/2)])
            pygame.draw.polygon(screen, BLACK, [ [start_x, start_y + int(rect_size/2)], [start_x - arrowmag*np.cos(45*np.pi/180), start_y + int(rect_size/2) - arrowmag*np.sin(45*np.pi/180)], [start_x - arrowmag*np.cos(45*np.pi/180),  start_y + int(rect_size/2) +arrowmag*np.sin(45*np.pi/180)] ])

        screen.blit(text, (x + int((rect_size - width)/2 ), y + int((rect_size - height)/2) ) )
        if(sys[i].nval != None and sys[i].nval != 0):
            string= str(sys[i].nval)
            width, height = font.size(string)
            text = font.render(string, True, BLACK)
            screen.blit(text, (x + int((rect_size - width)/2 ), y + int((space_size - height)/2) - space_size ) )
        if(sys[i].bval != None ):
            bval = str(sys[i].bval)
            width, height = font.size(bval)
            wvaltext = font.render(bval, True, BLACK)
            screen.blit(wvaltext, ( x - space_size + int((rect_size - width)/2),  y + int((rect_size - height)/2) ))

def showAMatrix(sys, screen, newa, start_x, start_y):
    for i in range(0, len(newa)):
        for j in range(0, len(newa[0])):
            if(newa[i][j] != 0):
                x = start_x  + rect_size*j + space_size*j
                y = start_y - (i+1)*(space_size + rect_size)
                pygame.draw.rect(screen, RED, [x,y, rect_size, rect_size])
                string= str(newa[i][j])
                width, height = font.size(string)
                text = font.render(string, True, BLACK)
                screen.blit(text, (x + int((rect_size - width)/2 ), y + int((rect_size - height)/2) ) )

def showBMatrix(sys, screen, newb1, newb2, midpoint, start_x, start_y):
    for i in range(0, 2):
        if(i == 0):
            for j in range(0, len(newb1)):
                if(newb1[j][0] != 0):
                    string= str(newb1[j][0])
                    width, height = font.size(string)
                    text = font.render(string, True, BLACK)
                    x = start_x -space_size
                    y = start_y + rect_size + space_size + rect_size*j + space_size*j
                    pygame.draw.rect(screen, RED, [x, y, rect_size, rect_size])
                    screen.blit(text, (x + int((rect_size - width)/2 ), y + int((rect_size - height)/2) ) )
                    arrowmag = 3
                    if(j== 0):
                        pygame.draw.line(screen, BLACK, [x + int(rect_size/2), y], [x + int(rect_size/2), y - int(space_size/2)])
                        pygame.draw.polygon(screen, BLACK,  [[x + int(rect_size/2), y - int(space_size/2)], [x + int(rect_size/2) - arrowmag*np.sin(45*np.pi/180),y - int(space_size/2) + arrowmag*np.cos(45*np.pi/180)], [x + int(rect_size/2) - arrowmag*np.sin(-45*np.pi/180), y - int(space_size/2) + arrowmag*np.cos(-45*np.pi/180)] ])
        else:
            for j in range(0, len(newb2)):
                if(newb2[j][0] != 0):
                    string= str(newb2[j][0])
                    width, height = font.size(string)
                    text = font.render(string, True, BLACK)
                    x = start_x + rect_size*midpoint + space_size*midpoint - space_size
                    y = start_y + rect_size + space_size + rect_size*j + space_size*j
                    pygame.draw.rect(screen, RED, [x, y, rect_size, rect_size])
                    screen.blit(text, (x + int((rect_size - width)/2 ), y + int((rect_size - height)/2) ) )
                    if(j == 0):
                        pygame.draw.line(screen, BLACK, [x + int(rect_size/2), y], [x + int(rect_size/2), start_y + int(rect_size/2)])
                        pygame.draw.polygon(screen, BLACK,  [[x + int(rect_size/2), start_y + int(rect_size/2)], [x + int(rect_size/2) - arrowmag*np.sin(45*np.pi/180),start_y + int(rect_size/2) + arrowmag*np.cos(45*np.pi/180)], [x + int(rect_size/2) - arrowmag*np.sin(-45*np.pi/180), start_y + int(rect_size/2) + arrowmag*np.cos(-45*np.pi/180)] ])

import time
def backrunCircularSystolicAnimate(sys,A, B, verbose = True):
        Ar = len(A)
        Ac = len(A[0])
        Br = len(B)
        Bc = len(B[0])
        l = len(sys)
        midpoint = l//2

        matCheckerCircular(Ar, Ac, Br, Bc)
        sysCheckerCircular(Br, l)
        print(A)
        newa = np.zeros((Ar + midpoint - 1,Ac))
        for j in range(0, len(newa[0])):
            if(j < midpoint):
                subA = A[j,:]
            else:
                subA = np.concatenate((A[j, -midpoint:], A[j, 0:midpoint]), axis = 0 )
            newa[j%midpoint:j%midpoint + len(subA),j] = subA

        newb1 = B[0:midpoint,:]
        newb2 =B[midpoint:, :]

        print(newb1)
        print(newb2)
        output = np.zeros((l))
        for i in sys:
            i.set_target(Ac)

        pygame.init()
        size = (700, 700)
        screen = pygame.display.set_mode(size)
        screen.fill(WHITE)
        pygame.display.set_caption("Systolic Simulation")
        clock = pygame.time.Clock()

        start_y = int(size[0]/2) - int(rect_size/2)
        start_x = size[0]//2 - (l*rect_size + l*(space_size - 1) )//2

        done = True
        k = 0
        stahp = False
        output = np.zeros((Ac , Bc))
        lbuffer = None

        while(done):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = False
            recurseCircularApass(sys, A, k)
            recurseCircularBPass(sys, B, k)

            screen.fill(WHITE)
            text = font.render('ClockTicks: %s' %(k), True, RED)
            screen.blit(text, (0, 0))
            showAMatrix(sys, screen, newa, start_x, start_y)
            showBMatrix(sys, screen, newb1,newb2, midpoint, start_x, start_y)
            showSys(sys, screen, start_x, start_y)
            pygame.display.flip()
            clock.tick(0.8)

            screen.fill(WHITE)
            text = font.render('ClockTicks: %s' %(k), True, RED)
            screen.blit(text, (0, 0))
            showAMatrix(sys, screen, newa, start_x, start_y)
            showBMatrix(sys, screen, newb1,newb2, midpoint, start_x, start_y)
            stahp = True
            for y in range(len(sys) -1, -1, -1):
                i = sys[y]
                if(i.bval is None and i.e.passon is not None):
                    if(y != 0):
                        i.bval = i.e.passon
                    else:
                        i.bval = lbuffer

                showSysIndividual(sys, screen, start_x, start_y, y)

                if(i.bval is not None and i.nval is not None and not i.finished):
                    i.computeVal()
                    if(y != len(sys) - 1):
                        i.passon = i.bval
                        i.bval = None
                    else:
                        lbuffer = i.passon
                        i.passon = i.bval
                        i.bval = None
                if i.finished:
                    output[y][0] = i.val
                stahp = stahp and i.finished

            pygame.display.flip()
            clock.tick(0.8)

            if(newa.shape[0] >= 1):
                newa = np.delete(newa, 0,axis = 0)
            if(newb1.shape[0] >= 1):
                newb1 = np.delete(newb1,0, axis = 0)
            if(newb2.shape[0] >= 1):
                newb2 = np.delete(newb2,0, axis = 0)

            k+=1

        pygame.quit()


# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

pygame.font.init()
font = pygame.font.SysFont("sans", 10)

rect_size = 15
space_size = 15
mat_from_sys = space_size

import numpy as np

a = np.random.randint(low = 1, high = 9, size= (10,10))
b = np.random.randint(low = 1, high = 9, size = (10,1))

sys = initCircularSystolic(10)
backrunCircularSystolicAnimate(sys, a,b, False)
