
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


def convolve_RelBlur(Iin,k):
    convp=np.dot(Iin,k)
    conv=convp[...,::-1]
    return (conv)


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

def deconvolve_L1_NS(sig,mask,deconV,rgP,option,value):
    """decolvolution using the Lucy-Richardson algorithm with L1 norm regularization for nonstationary RF

                    Parameters
                    ----------
                    sig=array
                        data to be deconvolved
                    mask :array
                        Resolution Function
                    deconV :array
                        initial guess of the data
                    eps: floats
                        smoothing parameter for TV gradient
                    rgp : float
                        regularization parameter
                    option=string
                        iteration
                    value: integer
                        number of iteration
                    """
    sig0=sig
    mask_mir=mask.transpose()
    deconv = deconV

    def main(deconv,mask,sig0,mask_mir,rgP):
            sigC=convolve_NS(deconv,mask)
            relative_blur=sig0/sigC
            with np.errstate(divide='ignore'):
                    relative_blur[np.isinf(relative_blur)] = -2

            dif1=np.ediff1d(deconv, to_begin=0)/np.sum(np.abs(np.ediff1d(deconv, to_begin=0)))
            dif2=np.ediff1d(dif1, to_begin=0)
            # norm=np.sqrt(np.sum(deconv**2))
            # mod_norm=np.sqrt(eps**2+norm**2)
            # div_rgp=rgP*mod_norm
            # one=np.full(sig0.shape, 1)
            # exone=convolve_NS(one, mask_mir)
            # rg=exone+(rgP*np.sign(deconv))
            deconvP=(deconv*convolve_NS(relative_blur,mask_mir))/(1-(rgP*dif2))
            error=LA.norm(deconvP-deconv)
            deconv=deconvP
            return(deconv,error)

    if option=='iteration':
            error=0
            for i in xrange(value):
                    deconv,error=main(deconv,mask,sig0,mask_mir,rgP)

    if option=='error':
            it=0
            while True:
                    deconv,error=main(deconv,mask,sig0,mask_mir,rgP)
                    it=it+1
                    if np.all(error<value):
                            break
            print('number of iteration: {}'.format(it))
    return(deconv)








