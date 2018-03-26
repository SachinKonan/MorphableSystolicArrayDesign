import matplotlib.pyplot as plt
"""constants derived from TPU paper"""
memory_time = 3.05*(10**-11)
clock_time = 1.42857*(10**-9) #this describes a 700 MHz System, each cycle takes this amount of time

class PE:
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

def initSystolic(elements):
    sys = [PE(i) for i in range(elements)]
    if(elements%2 != 0):
        raise Exception('Systolic Array Size Must be Divisible by Two for Circulation')
    for i in range(0, len(sys)):
        sys[i].addENeighbor(sys[i-1])
        if(i == len(sys)//2 or i == 0):
            sys[i].addInputChannel()
    return sys

def printSystolic(sys):
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

def recurseApass(sys, A, it):
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

def recurseBPass(sys, B, it):
    midpoint = len(B)//2
    if(0 + it < midpoint or midpoint + it < len(B)):
        sys[0].addBVal(B[0+it][0])
        sys[len(sys)//2].addBVal(B[midpoint + it][0])

def primitive(sys, A, B):
    a = np.zeros((len(sys) + 1, len(sys)))


def autoPassinNextWaveofData(sys, A, B, i):
    sr = len(sys)
    sc = len(sys[0])
    recurseApass(sys, A, sr,sc, i)
    recurseBpass(sys, B, sr, sc, i)

def matChecker(Ar, Ac, Br, Bc):
    if(Ac != Br):
        raise Exception("Matrix Sizes A,B aren't compatible")
    if(Ac%2 != 0 or Br%2 != 0):
        raise Exception("In order to use CircularSystolic, the matrices must be divisuble by Two")
    if(Ac != Ar):
        raise Exception("The matrix must be square for CircularSystolic to Work")

def sysChecker(Br, l):
    if(Br!= l):
        raise Exception("Systolic array isn't sized properly for input matrices")

import time
def backrunSystolic(sys,A, B, verbose = True):
        Ar = len(A)
        Ac = len(A[0])
        Br = len(B)
        Bc = len(B[0])
        matChecker(Ar, Ac, Br, Bc)
        sysChecker(Br, len(sys))

        for i in sys:
            i.set_target(Ac)

        output = np.zeros((Ac , Bc))
        lbuffer = None
        stahp = False
        k = 0
        cutout = 0
        while(not stahp):
            recurseApass(sys, A, k)
            stime = time.time()
            recurseBPass(sys, B, k)
            etime = time.time()
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
                    if(y != len(sys) - 1):
                        i.computeVal()
                        i.passon = i.bval
                        i.bval = None
                    else:
                        i.computeVal()
                        lbuffer = i.passon
                        i.passon = i.bval
                        i.bval = None
                if i.finished:
                    output[y][0] = i.val
                stahp = stahp and i.finished
            delta = time.time() - start_time
            cutout += delta + etime - stime
            if(verbose):
                printSystolic(sys)
            k+=1
        return output, k, cutout

def padArray(sr, sc, sub):
    result = np.zeros((sr, sc))
    result[:len(sub), :len(sub[0])] = sub
    return result

def blockChecker(Ac, Br):
    if(Ac != Br):
        assert Exception('Matrices arent compatible for multiplication')

import math
def blockSystolicMultiply(sys, A, B, verbose = True):
    sr = len(sys)

    Ar = len(A)
    Ac = len(A[0])

    Br = len(B)
    Bc = len(B[0])

    blockChecker(Ac, Br)

    out = np.zeros((Ar,Bc))
    trackiterations = 0
    clocks = 0
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
            m, k, cutout = backrunSystolic(sys, padArray(sr, sr, A[rstart:rend, cstart:cend]), padArray(sr, 1, B[cstart:cend, 0].reshape(-1, 1)), verbose = verbose )
            clocks = k
            delete+=cutout
            out[rstart:rend, 0:] = out[rstart:rend, 0:] + m[0:rend - rstart, :]
            sys = initSystolic(sr)
            trackiterations+=1
        #print('Finished section')
    end_time = time.time() - start_time - delete
    if verbose:
        print('Total Number of Systolic Iterations: %s' %(trackiterations))
    return out, trackiterations, clocks, end_time

def padArray(sr, sc, sub):
    result = np.zeros((sr, sc))
    result[:len(sub), :len(sub[0])] = sub
    return result


A = np.random.rand(512,512)
B = np.random.rand(512, 1)
#result, iterations = backrunSystolic(sys, A, B, False)
#out, repititions, clks, time = blockSystolicMultiply(sys, A, B, False)

"""print('Numpy Result:')
print(np.dot(A,B).T)
print('Iterations from CircularSystolic: %s' %(repititions))
print(out.T)
print('Time for this:%s'%(time))"""

clocks = []
repetitions = []
memory = []
tim = []

shapes = np.arange(2, 258, 2)
for i in shapes:
    sys = initSystolic(i)
    outs, reps, cs, t = blockSystolicMultiply(sys, A, B, False)
    tim.append(t)
    memory.append(i**2)
    clocks.append(cs)
    repetitions.append(reps)
    print('Finished Systolic%s' %(i))

b = np.argmin(tim)
plt.plot(shapes[b], tim[b], marker = 'o', markersize = 3, color = 'red')
plt.plot(shapes,tim)
plt.title('Simulator Time')
plt.ylabel('Time(s)')
plt.xlabel('Circular Flow Systolic size')
plt.show()
print('Optimal Circular size time: %s' %(tim[b]))
print('Max size(256) time: %s' % (tim[-1]))
print('Real Circular Size to minimize loss: %s' %(b))
print('Real Circular Size speedup: %s' % (tim[-1]/tim[b]))

clocks = np.array(clocks)
repetitions = np.array(repetitions)
memory = np.array(memory)

total_loss = np.multiply(repetitions, np.add(memory_time*memory, clock_time*clocks))

best_size = np.argmin(total_loss)
speed_diff = total_loss[-1]/total_loss[best_size]

print('Optimal Size to minimize loss: %s' %(best_size))
print('Optimal Size speedup: %s' % (speed_diff))
plt.plot(shapes, total_loss, label = 'added')
plt.title("Theoretical Circular Systolic Time")
#plt.plot(shapes, memory, label = 'memory' )
#plt.plot(shapes, clocks, label = 'clocks')
#plt.plot(shapes, repetitions/(100**2), label = 'repetitions')
plt.legend()
plt.show()
