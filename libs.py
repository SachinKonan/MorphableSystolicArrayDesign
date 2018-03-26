import matplotlib.pyplot as plt
"""constants derived from TPU paper"""
memory_time = 3.05*(10**-11)
clock_time = 1.42857*(10**-9) #this describes a 700 MHz System, each cycle takes this amount of time

class NormalPE:
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

def initNormalSystolic(rows, rcols):
    sys = [ [NormalPE(i,j) for j in range (rcols)] for i in range(rows)]

    for i in range(len(sys)-1, -1, -1):
        for j in range(len(sys[0]) - 1, -1, -1):
            if(j-1 >= 0):
                sys[i][j].addWNeighbor(sys[i][j-1])
            if(i - 1 >= 0):
                sys[i][j].addNeighbor(sys[i-1][j])
    return sys

def printSystolic(sys):
    for i in range(len(sys)):
        for j in range(len(sys[0])):
            print(sys[i][j], end=' ')
        print()

def recursegetDiagonals(d):
    if(all(v is None for v in d[-1]) ):
        return d[0:-1]
    else:
        refs = []
        for i in d[-1]:
            if(i.e == None and i.s == None):
                refs.append(None)
            elif(i.e == None and i.s != None):
                refs.append(i.s)
            elif(i.e != None and i.s == None):
                refs.append(i.e)
            else:
                refs.append(i.e)
                refs.append(i.s)
        d.append(list(set(refs)) )
        return recursegetDiagonals(d)

import numpy as np

def recurseNormalApass(sys,A, sr, sc, it):
    colA = len(A[0]) -1
    output = []
    for i in range(0, sr):
        if(it < colA + sr and it - i >= 0 and it - i <= colA):
            sys[i][0].addWVal(A[i][it - i])

def recurseNormalBpass(sys,B, sr, sc, it):
    rowB = len(B) - 1
    for i in range(0, sc):
        if(it < rowB + sc and it - i >= 0 and it - i <= rowB):
            sys[0][i].addNVal(B[it - i][i])

def autoNormalPassinNextWaveofData(sys, A, B, i):
    sr = len(sys)
    sc = len(sys[0])
    recurseNormalApass(sys, A, sr,sc, i)
    recurseNormalBpass(sys, B, sr, sc, i)

def matCheckerNormal(Ar, Bc):
    if(Ar != Bc):
        raise Exception("Matrix Sizes A,B aren't compatible")

def sysCheckerNormal(Ar, Bc, sr, sc):
    if(Ar != sr or Bc != sc):
        raise Exception("Systolic array isn't sized properly for input matrices")

import time
def backrunSystolicNormal(sys,A, B, verbose = True):
        start = time.time()
        Ar = len(A)
        Ac = len(A[0])
        Br = len(B)
        Bc = len(B[0])
        matCheckerNormal(Ac, Br)
        sysCheckerNormal(Ar, Bc, len(sys), len(sys[0]))

        output = np.zeros((len(sys), len(sys[0])))
        for i in sys:
            for j in i:
                j.set_target(Ac)

        count = 0
        finish = False
        cutout = 0
        end = time.time()

        while(not finish):
            #passinNextWaveofData(sys, A, B, count)
            autoNormalPassinNextWaveofData(sys, A,B, count)
            if verbose:
                print('start')
                printNormalSystolic(sys)
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
                printNormalSystolic(sys)
                print(all_check)
            finish = all_check
            count +=1
        if(verbose):
            print('Systolic Propogation Clock Ticks: %s' % (count))

        return count, output, cutout/count, end - start

def padArray(sr, sc, sub):
    result = np.zeros((sr, sc))
    result[:len(sub), :len(sub[0])] = sub
    return result

import math
def blockNormalSystolicMultiply(sys, A, B, verbose = True):
    sr = len(sys)
    sc = len(sys[0])

    Ar = len(A)
    Ac = len(A[0])

    Br = len(B)
    Bc = len(B[0])

    matCheckerNormal(Ac, Br)
    out = np.zeros((Ar,Bc))
    trackiterations = 0
    clocks = 0
    for_freq = 0
    #for matrix multiplication between layers, the multiplication is a matrix vs a vector
    #the reason that sr is repeated here, it is assumed that chunks of size (sr,sr) and (sr, sc)
    start_time = time.time()
    delete = 0
    for r in range(0, math.ceil(Ar/sr)):
        rstart = r*sr
        rend = rstart + sr
        if(rend > Ar):
            rend = Ar
        for c in range(0, math.ceil(Ac/sr)):
            cstart = c*sr
            cend = cstart + sr
            if(cend > Ac):
                cend = Ac
            #print(rstart, rend, cstart, cend)
            #mult = np.dot(padArray(sr, sr, A[rstart:rend, cstart:cend]), padArray(sr, sc, B[cstart:cend, 0].reshape(-1, 1)))
            clks, m,for_frequency, deleteTime = backrunSystolicNormal(sys, padArray(sr, sr, A[rstart:rend, cstart:cend]), padArray(sr, sc, B[cstart:cend, 0].reshape(-1, 1)), verbose = verbose )
            clocks = clks
            delete += deleteTime
            for_freq += for_frequency
            out[rstart:rend, 0:] = out[rstart:rend, 0:] + m[0:rend - rstart, :]
            sys = initNormalSystolic(sr, sc)
            trackiterations+=1
        #print('Finished section')
    total_time = time.time() - start_time
    if verbose:
        print('Total Number of Systolic Iterations: %s' %(trackiterations))
    return out, trackiterations, clocks, for_freq, total_time - delete


class CircularPE:
    def __init__(self, i):
        self.val = 0
        self.i = i
        self.e = None
        self.input = False
        self.val = 0
        self.nval = 0
        self.bval = None
        self.passon = None
        self.counter = 0
        self.finished = False
        self.target = 0

    def addENeighbor(self,x):
        self.e = x

    def addBVal(self, x):
        self.bval = x

    def addNVal(self,x):
        self.nval = x

    def computeVal(self):
        self.val += self.nval*self.bval
        self.counter+=1
        if(self.counter >= self.target):
            self.finished = True

    def set_target(self,x):
        self.target = x

    def addInputChannel(self):
        self.input = True

    def __str__(self):
        string = ''
        string += 'PE:%s, BV:%s, NV:%s, P:%s, V:%s||| ' % (self.i, self.bval, self.nval, self.passon, self.val)
        #string += 'P:%s,||| ' % (self.passon)
        return string

def initCircularSystolic(elements):
    sys = [CircularPE(i) for i in range(elements)]
    if(elements%2 != 0):
        raise Exception('Systolic Array Size Must be Divisible by Two for Circulation')
    for i in range(0, len(sys)):
        sys[i].addENeighbor(sys[i-1])
        if(i == len(sys)//2 or i == 0):
            sys[i].addInputChannel()
    return sys

def printCircularSystolic(sys):
    for i in range(len(sys)):
        print(sys[i], end='')
    print()

def recursegetDiagonals(d):
    if(all(v is None for v in d[-1]) ):
        return d[0:-1]
    else:
        refs = []
        for i in d[-1]:
            if(i.e == None and i.s == None):
                refs.append(None)
            elif(i.e == None and i.s != None):
                refs.append(i.s)
            elif(i.e != None and i.s == None):
                refs.append(i.e)
            else:
                refs.append(i.e)
                refs.append(i.s)
        d.append(list(set(refs)) )
        return recursegetDiagonals(d)

import numpy as np

def recurseCircularApass(sys, A, it):
    add = []
    midpoint = len(sys)//2
    for i in range(0, len(sys)):
        shifter = it - i%midpoint
        if(shifter >= 0 and shifter <= len(A) - 1):
            if(i < midpoint):
                sys[i].addNVal(A[i][shifter])
                #add.append(A[i][shifter])
            else:
                sys[i].addNVal(A[i][shifter - midpoint])
                #add.append(A[i][shifter - midpoint])
        else:
            sys[i].addNVal(0)
            #add.append(0)

def recurseCircularBPass(sys, B, it):
    midpoint = len(B)//2
    if(0 + it < midpoint or midpoint + it < len(B)):
        sys[0].addBVal(B[0+it][0])
        sys[len(sys)//2].addBVal(B[midpoint + it][0])

def primitive(sys, A, B):
    a = np.zeros((len(sys) + 1, len(sys)))


def autoCircularPassinNextWaveofData(sys, A, B, i):
    sr = len(sys)
    recurseApass(sys, A, sr,sc, i)
    recurseBpass(sys, B, sr, sc, i)

def matCheckerCircular(Ar, Ac, Br, Bc):
    if(Ac != Br):
        raise Exception("Matrix Sizes A,B aren't compatible")
    if(Ac%2 != 0 or Br%2 != 0):
        raise Exception("In order to use CircularSystolic, the matrices must be divisuble by Two")
    if(Ac != Ar):
        raise Exception("The matrix must be square for CircularSystolic to Work")

def sysCheckerCircular(Br, l):
    if(Br!= l):
        raise Exception("Systolic array isn't sized properly for input matrices")

import time
def backrunCircularSystolic(sys,A, B, verbose = True):
        start = time.time()
        Ar = len(A)
        Ac = len(A[0])
        Br = len(B)
        Bc = len(B[0])
        matCheckerCircular(Ar, Ac, Br, Bc)
        sysCheckerCircular(Br, len(sys))

        for i in sys:
            i.set_target(Ac)

        output = np.zeros((Ac , Bc))
        lbuffer = None
        stahp = False
        k = 0
        cutout = 0
        end = time.time()
        while(not stahp):
            recurseCircularApass(sys, A, k)
            recurseCircularBPass(sys, B, k)
            #print(input_result)
            #print(input_B_result)
            """for x in range(0, 2):
                if(input_B_result is not None):
                    sys[x*(len(sys)//2)].addBVal(input_B_result[x])"""

            """for j in range(0, len(sys)):
                if(input_result is not None):
                    sys[j].addNVal(input_result[j])"""

            stahp = True
            start_time = time.time()
            for y in range(len(sys) -1, -1, -1):
                i = sys[y]
                if(i.bval is None and i.e.passon is not None):
                    if(y != 0):
                        i.bval = i.e.passon
                    else:
                        i.bval = lbuffer

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
            delta = time.time() - start_time
            cutout += delta
            if(verbose):
                printCircularSystolic(sys)
            k+=1
        return output, k, cutout/k, end - start

def padArray(sr, sc, sub):
    result = np.zeros((sr, sc))
    result[:len(sub), :len(sub[0])] = sub
    return result

def blockCircularChecker(Ac, Br):
    if(Ac != Br):
        assert Exception('Matrices arent compatible for multiplication')

import math
def blockCircularSystolicMultiply(sys, A, B, verbose = True):
    sr = len(sys)

    Ar = len(A)
    Ac = len(A[0])

    Br = len(B)
    Bc = len(B[0])

    blockCircularChecker(Ac, Br)

    out = np.zeros((Ar,Bc))
    trackiterations = 0
    clocks = 0
    for_freq = 0
    #for matrix multiplication between layers, the multiplication is a matrix vs a vector
    #the reason that sr is repeated here, it is assumed that chunks of size (sr,sr) and (sr, sc)
    delete = 0
    start_time = time.time()
    for r in range(0, math.ceil(Ar/sr)):
        rstart = r*sr
        rend = rstart + sr
        if(rend > Ar):
            rend = Ar
        for c in range(0, math.ceil(Ac/sr)):
            cstart = c*sr
            cend = cstart + sr
            if(cend > Ac):
                cend = Ac
            #print(rstart, rend, cstart, cend)
            #mult = np.dot(padArray(sr, sr, A[rstart:rend, cstart:cend]), padArray(sr, sc, B[cstart:cend, 0].reshape(-1, 1)))
            m, k, for_frequency, cutout = backrunCircularSystolic(sys, padArray(sr, sr, A[rstart:rend, cstart:cend]), padArray(sr, 1, B[cstart:cend, 0].reshape(-1, 1)), verbose = verbose )
            for_freq += for_frequency
            clocks = k
            delete+=cutout
            out[rstart:rend, 0:] = out[rstart:rend, 0:] + m[0:rend - rstart, :]
            sys = initCircularSystolic(sr)
            trackiterations+=1
        #print('Finished section')
    end_time = time.time() - start_time
    if verbose:
        print('Total Number of Systolic Iterations: %s' %(trackiterations))
    return out, trackiterations, clocks, for_freq, end_time - delete


reps = lambda mat_size, r,c: math.ceil(r/mat_size)*math.ceil(c/mat_size)
memory = lambda mat_size, r,c: mat_size**2 + mat_size
clocks = lambda mat_size, r, c: 2*mat_size - 1

def find_optimum_shape_Neural_Net(layers, verbose = False):
    MAX = 512
    total = []
    for i in layers.keys():
        r,c = layers[i]['weights'].shape[0:2]
        print(r,c)
        min_shape, lat = find_optimum_shape_layer(r, c, MAX)
        total.append([min_shape, lat])
        if(verbose):
            print('Finished Sizing Layer%s'%(i))
    total = np.array(total)
    print(total)
    optimum_net_size = sum(np.multiply(total[:,0],total[:,1]))/sum(total[:,1])
    return int(optimum_net_size)

def find_optimum_shape_layer(r,c ,MAX_SIZE = 256):
    shapes = np.arange(1,MAX_SIZE)
    lats = []
    for i in shapes:
        lats.append(reps(i, r, c)*( memory_time*memory(i, r, c) + clock_time*clocks(i, r, c) ))
    return shapes[np.argmin(lats)], min(lats)