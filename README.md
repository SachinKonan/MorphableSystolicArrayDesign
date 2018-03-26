# MorphableSystolic

The systolic array is a widely used and relatively old parallel architecture for quickly multiplying matrices. In the climate of increasingly complex Neural Networks topologies, the need for efficient large-scale matrix multiplication is paramount. That is why an optimization method for resizing the structure of the systolic array for any given multiplication is proposed. This takes into account several latency-causers like repetition and memory sourcing. Additionally, a faster model for systolic multiplication, Circular-Flow Systolic, is presented, which moves information circularally rather than linearly for faster run-time.

## Getting Started

### Prerequisites

Python3
Numpy
Scipy
Matplotlib
Pygame(for animation)
```
pip install numpy
pip install scipy
pip install matplotlib
pip install pygame
```
## Running the tests

To run a matrix multiplication via either the Normal Systolic Array or the Circular Systolic Array, simply run:
```
from libs import *
import numpy as np

A = np.random.randint(5,2)
B = np.random.randint(2,5)

result, iterations, clks, for_freq, time = blockNormalSystolicMultiply(initNormalSystolic(x, 1), A, B, verbose = False)
```
the dimensions of the systolic array are variable, but run-time will be very large if set too low.

## Deployment

To visualize the Systolic Array's operation run either the NormalSystolic or CircularSystolic's animation files in the Simulator folder.


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Sachin Konan** 

