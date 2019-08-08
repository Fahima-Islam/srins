
import numpy as np
import math
from numpy import linalg as LA
from math import factorial



def scale(Y, minS,maxS):
    zeroTO1=(Y-np.min(Y))/(np.max(Y)-np.min(Y))
    scaling=(zeroTO1*(maxS-minS))+minS
    return scaling







