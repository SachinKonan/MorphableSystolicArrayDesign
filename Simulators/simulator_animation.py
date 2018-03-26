import pygame
import os
from pygame.locals import *

if os.getcwd() != 'C:\\Users\\Sachin Konan\\Documents\\PythonPractice':
    os.chdir('C:\\Users\\Sachin Konan\\Documents\\PythonPractice')

class PE:
    def __init__(self, r, c):
        self.val = 0
        self.r = r
        self.c = c
        self.n = None
        self.w = None
        self.nval = None
        self.wval = None
        self.counter = 0
        self.finished = False
        self.target = 0

    def addWVal(self,a):
        self.wval = a

    def addNVal(self, a):
        self.nval = a

    def set_target(self, t):
        self.target = t

    def compute(self):
        self.val += self.nval*self.wval
        self.counter+=1
        self.globalchecker()

    def globalchecker(self):
        if(self.counter >= self.target):
            self.finished = True

    def getFinal(self):
        return self.val

    def addWNeighbor(self,p):
        self.w = p

    def addNeighbor(self,p):
        self.n = p

    def __str__(self):
        string = ''
        string +=  'PE(%s, %s) ' % (self.r,self.c)
        string += ' Val: %s ' %(self.val)
        string.ljust(35)
        newstring = ''
        newstring += 'Inputs: %s, %s' %(self.wval, self.nval) + '|'
        newstring.rjust(45)

        """
        if(self.n is None):
            string += 'No N '
        else:
            string += 'Nref(%s, %s)' % (self.n.r, self.n.c)

        if(self.w is None):
            string += 'No W '
        else:
            string += 'Wref(%s, %s)' % (self.w.r, self.w.c)"""
        return string+newstring

def initSystolic(rows, rcols):
    sys = [ [PE(i,j) for j in range (rcols)] for i in range(rows)]

    for i in range(len(sys)-1, -1, -1):
        for j in range(len(sys[0]) - 1, -1, -1):
            if(j-1 >= 0):
                sys[i][j].addWNeighbor(sys[i][j-1])
            if(i - 1 >= 0):
                sys[i][j].addNeighbor(sys[i-1][j])
    return sys

import numpy as np

def recurseApass(sys,A, sr, sc, it):
    colA = len(A[0]) -1
    output = []
    for i in range(0, sr):
        if(it < colA + sr and it - i >= 0 and it - i <= colA):
            sys[i][0].addWVal(A[i][it - i])

def recurseBpass(sys,B, sr, sc, it):
    rowB = len(B) - 1
    for i in range(0, sc):
        if(it < rowB + sc and it - i >= 0 and it - i <= rowB):
            sys[0][i].addNVal(B[it - i][i])

def autoPassinNextWaveofData(sys, A, B, i):
    sr = len(sys)
    sc = len(sys[0])
    recurseApass(sys, A, sr,sc, i)
    recurseBpass(sys, B, sr, sc, i)

def matChecker(Ar, Bc):
    if(Ar != Bc):
        raise Exception("Matrix Sizes A,B aren't compatible")

def sysChecker(Ar, Bc, sr, sc):
    if(Ar != sr or Bc != sc):
        raise Exception("Systolic array isn't sized properly for input matrices")

def showSys(sys, screen, start_x, start_y):
    global font
    for i in range(0, len(sys)):
        for j in range(0, len(sys[0])):
            string= str(sys[i][j].val)
            width, height = font.size(string)
            text = font.render(string, True, BLACK)
            x = start_x + j*rect_size + j*space_size
            y =  start_y + i*rect_size + i*space_size
            pygame.draw.rect(screen, GREEN, [x, y, rect_size, rect_size])
            pygame.draw.line(screen, GREEN, [x+rect_size, y + int(rect_size/2)], [x+rect_size+space_size,  y + int(rect_size/2)], 1)
            pygame.draw.line(screen, GREEN, [x+int(rect_size)/2, y + rect_size], [x+int(rect_size)/2,  y + rect_size + space_size], 1)
            arrowmag = 2
            pygame.draw.polygon(screen, BLACK, [ [x+rect_size+space_size,  y + int(rect_size/2)], [x+rect_size+space_size - arrowmag*np.cos(45*np.pi/180), y + int(rect_size/2) - arrowmag*np.sin(45*np.pi/180)], [x+rect_size+space_size - arrowmag*np.cos(45*np.pi/180),  y + int(rect_size/2) +arrowmag*np.sin(45*np.pi/180)] ])
            pygame.draw.polygon(screen, BLACK, [ [x+int(rect_size/2) ,  y + rect_size+space_size], [x+int(rect_size/2) + arrowmag*np.sin(45*np.pi/180), y + rect_size+space_size - arrowmag*np.cos(45*np.pi/180)], [x+int(rect_size/2) + arrowmag*np.sin(-45*np.pi/180),  y + rect_size+space_size - arrowmag*np.cos(-45*np.pi/180)] ])

            screen.blit(text, ( x + int((rect_size - width)/2),  y + int((rect_size - height)/2) ))
            if(sys[i][j].nval != None):
                nval = str(sys[i][j].nval)
                width, height = font.size(nval)
                nvaltext = font.render(nval, True, BLACK)
                screen.blit(nvaltext, ( x + int((rect_size - width)/2),  y - space_size + int((rect_size - height)/2) ))
            if(sys[i][j].wval != None):
                wval = str(sys[i][j].wval)
                width, height = font.size(wval)
                wvaltext = font.render(wval, True, BLACK)
                screen.blit(wvaltext, ( x - space_size + int((rect_size - width)/2),  y + int((rect_size - height)/2) ))

def showAMatrix(sys, screen, newa, start_x, start_y):
    global font
    for i in range(0, len(newa)):
        for j in range(0, len(newa[0])):
            if(newa[i][j] != 0):
                string= str(newa[i][j])
                width, height = font.size(string)
                text = font.render(string, True, BLACK)
                x = start_x -mat_from_sys - rect_size - j* rect_size - j*space_size
                y =  start_y + i*space_size + i*rect_size
                pygame.draw.rect(screen, RED, [x,y, rect_size, rect_size])
                screen.blit(text, ( x + int((rect_size - width)/2),  y + int((rect_size - height)/2) ))

def showBMatrix(sys, screen, newb, start_x, start_y):
    global font
    for i in range(0, len(newb)):
        for j in range(0, len(newb[0])):
            if(newb[i][j] != 0):
                string= str(newb[i][j])
                width, height = font.size(string)
                text = font.render(string, True, BLACK)
                x = start_x + j* rect_size + j*space_size
                y =  start_y - mat_from_sys -rect_size - i*space_size - i*rect_size
                pygame.draw.rect(screen, RED, [x,y, rect_size, rect_size])
                screen.blit(text, ( x + int((rect_size - width)/2),  y + int((rect_size - height)/2) ))
import time
def backrunSystolicAnimate(sys,A, B, verbose = True):
        Ar = len(A)
        Ac = len(A[0])
        Br = len(B)
        Bc = len(B[0])
        matChecker(Ac, Br)
        sysChecker(Ar, Bc, len(sys), len(sys[0]))

        newa = np.zeros((Ar, Ac + len(sys) - 1))
        for i in range(0, len(newa)):
            newa[i, i:i + len(A[0])] = A[i,:]
        print(newa)

        newb = np.zeros((Br + len(sys[0]) - 1, Bc))
        for i in range(0, Bc):
            newb[i:i + Ac,i] = B[:, i]
        print(newb)

        output = np.zeros((len(sys), len(sys[0])))
        for i in sys:
            for j in i:
                j.set_target(Ac)

        pygame.init()
        size = (700, 700)
        screen = pygame.display.set_mode(size)
        screen.fill(WHITE)
        pygame.display.set_caption("Systolic Simulation of %sx%s *%sx%s" %(Ar, Ac, Br, Bc))
        clock = pygame.time.Clock()

        start_x = int((size[0] - (len(sys[0])*rect_size + (len(sys[0]) - 1)*space_size))/2) + 100
        start_y = int((size[1] - (len(sys)*rect_size + (len(sys) - 1)*space_size))/2) + 100

        count = 0
        finish = False
        cutout = 0
        while(not finish):
            #passinNextWaveofData(sys, A, B, count)

            autoPassinNextWaveofData(sys, A,B, count)
            screen.fill(WHITE)
            text = font.render('ClockTicks: %s' %(count), True, RED)
            screen.blit(text, (0, 0))
            showSys(sys, screen, start_x, start_y)
            showAMatrix(sys, screen, newa, start_x, start_y)
            showBMatrix(sys, screen, newb, start_x, start_y)
            pygame.display.flip()
            clock.tick(0.8)

            if verbose:
                print('start')
                printSystolic(sys)
            all_check = True
            cutoutstart = time.time()
            for j in range(len(sys)-1, -1, -1):
                for k in range( len(sys[0]) - 1, -1, -1 ):
                    i = sys[j][k]
                    if(i.nval is not None and i.wval is not None and not i.finished):
                        i.compute()
                        i.nval = None
                        i.wval = None
                    if(i.n is not None and i.n.nval is not None):
                        i.addNVal(i.n.nval)
                    if(i.w is not None and i.w.wval is not None):
                        i.addWVal(i.w.wval)
                    if(i.finished):
                        output[i.r][i.c] = i.val

                    all_check = not(not all_check or not i.finished)
                    #all_check = False if not i.finished else all_check
            cutoutend = time.time()
            cutout += cutoutend - cutoutstart
            if(verbose):
                print('end')
                printSystolic(sys)
                print(all_check)

            screen.fill(WHITE)
            text = font.render('ClockTicks: %s' %(count), True, RED)
            screen.blit(text, (0, 0))
            showSys(sys, screen, start_x, start_y)
            showAMatrix(sys, screen, newa, start_x, start_y)
            showBMatrix(sys, screen, newb, start_x, start_y)
            pygame.display.flip()
            clock.tick(0.8)

            if(newa.shape[1] >= 1):
                newa = np.delete(newa, 0, axis=1)
            if(newb.shape[0] >= 1):
                newb = np.delete(newb, 0, axis=0)

            finish = all_check
            count +=1

        pygame.quit()
        if(verbose):
            print('Systolic Propogation Clock Ticks: %s' % (count + 1))
        return count, output, cutout

# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

pygame.font.init()
font = pygame.font.SysFont("sans", 9)

rect_size = 15
space_size = 15
mat_from_sys = space_size

import numpy as np

a = np.random.randint(low = 1, high = 9, size= (10,10))
b = np.random.randint(low = 1, high = 9, size = (10,1))
sys = initSystolic(10,1)
backrunSystolicAnimate(sys, a,b, False)
