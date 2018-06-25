import matplotlib.pyplot as plt
"""constants derived from TPU paper"""
memory_time = 3.05*(10**-11)
clock_time = 1.42857*(10**-9) #this describes a 700 MHz System, each cycle takes this amount of time

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

import time
def backrunSystolic(sys,A, B, verbose = True):
        Ar = len(A)
        Ac = len(A[0])
        Br = len(B)
        Bc = len(B[0])
        matChecker(Ac, Br)
        sysChecker(Ar, Bc, len(sys), len(sys[0]))

        output = np.zeros((len(sys), len(sys[0])))
        for i in sys:
            for j in i:
                j.set_target(Ac)

        count = 0
        finish = False
        cutout = 0
        while(not finish):
            #passinNextWaveofData(sys, A, B, count)
            autoPassinNextWaveofData(sys, A,B, count)
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
            finish = all_check
            count +=1
        if(verbose):
            print('Systolic Propogation Clock Ticks: %s' % (count))
        return count, output, cutout

def padArray(sr, sc, sub):
    result = np.zeros((sr, sc))
    result[:len(sub), :len(sub[0])] = sub
    return result

import math
def blockSystolicMultiply(sys, A, B, verbose = True):
    sr = len(sys)
    sc = len(sys[0])

    Ar = len(A)
    Ac = len(A[0])

    Br = len(B)
    Bc = len(B[0])

    matChecker(Ac, Br)
    out = np.zeros((Ar,Bc))
    trackiterations = 0
    clocks = 0
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
            clks, m,deleteTime = backrunSystolic(sys, padArray(sr, sr, A[rstart:rend, cstart:cend]), padArray(sr, sc, B[cstart:cend, 0].reshape(-1, 1)), verbose = verbose )
            clocks = clks
            delete += deleteTime
            out[rstart:rend, 0:] = out[rstart:rend, 0:] + m[0:rend - rstart, :]
            sys = initSystolic(sr, sc)
            trackiterations+=1
        #print('Finished section')
    total_time = time.time() - start_time - delete
    if verbose:
        print('Total Number of Systolic Iterations: %s' %(trackiterations))
    return out, trackiterations, clocks, total_time

import math
def blockSystolicMultiply2D(sys, A, B, verbose = True):
    sr = len(sys)
    sc = len(sys[0])

    Ar = len(A)
    Ac = len(A[0])

    Br = len(B)
    Bc = len(B[0])

    matChecker(Ac, Br)
    out = np.zeros((Ar,Bc))
    trackiterations = 0
    clocks = 0
    print(math.ceil(Bc/sc))

    for r in range(0, math.ceil(Ar/sr)):
        rstart = r*sr
        rend = rstart + sr
        if(rend > Ar):
            rend = Ar
        for c in range(0, math.ceil(Ac/sc)):
            cstart = c*sc
            cend = cstart + sc
            if(cend > Ac):
                cend = Ac
            #print('A', padArray(sr,sc,A[rstart: rend, cstart: cend]))
            for bc in range(0, math.ceil(Bc/sc)):
                bcstart = bc*sc
                bcend = bcstart + sc
                if(bcend > Bc):
                    bcend = Bc
                #print('B', padArray(sc,sc,B[cstart:cend, bcstart: bcend]))
                if(verbose):
                    print('A', padArray(sr,sc,A[rstart: rend, cstart: cend]))
                    print('B', padArray(sc,sc,B[cstart:cend, bcstart: bcend]))
                clks, m,deleteTime = backrunSystolic(sys, padArray(sr,sc,A[rstart: rend, cstart: cend]), padArray(sc,sc,B[cstart:cend, bcstart: bcend]), verbose = verbose )
                if(verbose):
                    print('out',m)
                out[rstart:rend, bcstart:bcend] += m[0:rend-rstart,0:bcend - bcstart]
                sys = initSystolic(sr, sc)
                trackiterations+=1
    if verbose:
        print('Total Number of Systolic Iterations: %s' %(trackiterations))
    return out, trackiterations, clocks


def check_same(a,b):
    if(a.shape != b.shape):
        return False
    else:
        for i in range(0, len(a)):
            if(a[i][0] != b[i][0]):
                return False
            else:
                pass
        return True


a = np.array([
[1,2,3],
[1,2,3],
])
b = np.array([
[1,2,3],
[1,2,3],
[1,2,3]
])
sys = initSystolic(1, 1)
l,m,k = blockSystolicMultiply2D(sys, a, b, verbose = False)
print('final',l)
