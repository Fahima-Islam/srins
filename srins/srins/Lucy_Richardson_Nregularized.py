
import numpy as np
import math
from numpy import linalg as LA
from math import factorial


def convolve_NS(sig,mask):
    """convolution of 1 D array with 2D array

                    Parameters
                    ----------
                    sig=1 D array
                        data to be deconvolved
                    mask :2 D array
                        Resolution Function
                    """

    con= np.dot(sig, mask)

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



def convolve_RelBlur(Iin,k):
    convp=np.dot(Iin,k)
    conv=convp[...,::-1]
    return (conv)


def deconvolve_NS(sig,mask,deconV,delta_ER, neu_N,option,value):
    """decolvolution using the Lucy-Richardson algorithm with non-stationary RF

                   Parameters
                   ----------
                   sig=array
                       data to be deconvolved
                   mask :array
                       Resolution Function
                   deconV :array
                       initial guess of the data
                   option=string
                       iteration
                   value: integer
                       number of iteration
                   """
    sig0=sig
    # mask_mir=mask[...,::-1]
    mask_mir = mask.transpose()
    deconv = deconV

    def main(deconv,mask,sig0,mask_mir):
        sigC=convolve_NS(deconv,mask)
        relative_blur=sig0/sigC
        with np.errstate(divide='ignore'):
            relative_blur[np.isinf(relative_blur)] = -2
        deconvP=deconv*convolve_NS(relative_blur, mask_mir)
        deconvU = delta_ER*shrink(deconvP, neu_N)
        error=LA.norm((deconvP-deconv))
        deconv=deconvU
        return(deconv,error)

    if option=='iteration':
        error=0
        it=value
        for i in xrange(value):
            deconv,error=main(deconv,mask,sig0,mask_mir)

    if option=='error':
        it=0
        while True:
            deconv,error=main(deconv,mask,sig0,mask_mir)
            it=it+1
            if error<value:
                break
        print('number of iteration: {}'.format(it))
    return (deconv,error,it)










