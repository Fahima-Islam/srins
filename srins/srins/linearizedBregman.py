
import numpy as np
import math
from numpy import linalg as LA
from math import factorial
from numpy.linalg import pinv
from matplotlib import pyplot as plt

def convolve_NS(sig,mask):
    """convolution of 1 D array with 2D array

                    Parameters
                    ----------
                    sig=1 D array
                        data to be deconvolved
                    mask :2 D array
                        Resolution Function
                    """

    con= np.dot(sig,mask) # when the resolution matrix is saved in row wise

    return(con)


def shrink(y,a):
    L1norm=LA.norm(y, ord=1)
    mu=np.full(y.shape[0], a)
    mody=y.copy()
    for i in xrange(y.shape[0]):
        if y[i]>mu[i]:
            mody[i]=y[i]-mu[i]
        if y[i]<-mu[i]:
            mody[i] = y[i]+mu[i]
        if (y[i]>=-mu[i]) & (y[i]<=mu[i]):
            mody[i] = 0
    return (mody)

def shrink(y, a):
    ay = np.abs(y)
    ay -= a
    ay[ay<0] =0
    return np.sign(y) * ay


def step(iniGuessU, iniGuessV, mask,sig,mask_mir,neu_N, delta_ER ):
    # sig1=sig/np.sum(sig)
    # iniGuessU=iniGuessU/np.sum(iniGuessU)
    # iniGuessV = iniGuessV / np.sum(iniGuessV)
    sigC=convolve_NS(iniGuessU,mask)
    # sigC =np.dot(mask,iniGuessU)
    relative_blur=sig-sigC
    # with np.errstate(divide='ignore'):
    #     relative_blur[np.isinf(relative_blur)] = -2
    deconvV = iniGuessV + convolve_NS(relative_blur, mask_mir)
    # deconvV = iniGuessV + np.dot(mask_mir, relative_blur)
    deconvU = delta_ER*shrink(deconvV, neu_N)
    # if (np.any(np.isnan(deconvU))==True):
    #     print (deconvU)
    # deconvU=deconvU/np.sum(deconvU)
    error=(LA.norm((deconvU-iniGuessU)))/iniGuessU.size
    diff = convolve_NS(deconvU, mask) - sig
    errorBL = np.sqrt(np.average(diff**2))
    # errorBL = LA.norm((convolve_NS(deconvV, mask) - sig), 1)
    iniGuessV=deconvV
    iniGuessU = deconvU
    return(iniGuessV,iniGuessU,error,errorBL)



def bregman_NS(sig, mask,iniGuessV, iniGuessU, neu_N, delta_ER, option,value, maxIter=100): #neu_inverseNoise, delta_energyResolution
    """decolvolution using the Linearized Bregman Iteration with non-stationary resolution functions

                   Parameters
                   ----------
                   sig=array
                       data to be deconvolved
                   mask :array
                       Resolution Function
                   iniGuessV :array
                       initial guess of the data
                   iniGuessU :array
                       initial guess of the data
                   option=string
                       'iteration' or 'error'
                   value: integer or float
                       number of iteration or the tolerance value for error
                   """
    mask_mir = mask.transpose()

    if option=='iteration':
        error=0
        it=value
        for i in xrange(value):
            iniGuessV, iniGuessU,error,errorBL=step(iniGuessU, iniGuessV, mask,sig,mask_mir,neu_N, delta_ER )

    if option=='errorModel':
        it=0
        while it<maxIter:
            iniGuessV, iniGuessU,error,errorBL=step(iniGuessU, iniGuessV, mask,sig,mask_mir,neu_N, delta_ER )
            it=it+1
            # if error<value:
            if errorBL <value:
                break
        # print('number of iteration: {}'.format(it))

    if option == 'error':
        it = 0
        while it<maxIter:
            iniGuessV, iniGuessU, error, errorBL = step(iniGuessU, iniGuessV, mask, sig, mask_mir, neu_N, delta_ER)
            it = it + 1
            #print("Iteration #%s" % it)
            if error<value:
                break
        print('number of iteration: {}'.format(it))

    return (iniGuessV,iniGuessU,error,it,errorBL)

