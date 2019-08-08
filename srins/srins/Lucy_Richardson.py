
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


def deconvolve_NS(sig,mask,deconV,option,value,maxIter=100):
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
    # sig0=sig/np.sum(sig)
    # mask_mir=mask[...,::-1]
    mask_mir = mask.transpose()
    deconv = deconV

    def main(deconv, mask, sig, mask_mir):
        sigC = convolve_NS(deconv, mask)
        relative_blur = sig / sigC
        with np.errstate(divide='ignore'):
            relative_blur[np.isinf(relative_blur)] = -2
            relative_blur[np.isnan(relative_blur)] = -2
        deconvP = deconv*convolve_NS(relative_blur, mask_mir)
        error = LA.norm((deconvP - deconv)) / deconv.size
        # print('error =', error, 'has nan =', np.isnan(deconvP).any())
        # errorBL = (LA.norm(convolve_NS(deconv, mask) - sig)) / deconv.size
        deconv = deconvP
        diff = convolve_NS(deconv, mask) - sig
        errorBL = np.sqrt(np.average(diff ** 2))
        return deconv, error, errorBL

    if option=='iteration':
        error=0
        it=value
        for i in xrange(value):
            deconv, error, errorBL = main(deconv,mask,sig,mask_mir)

    if option == 'errorModel':
        it = 0
        while it < maxIter:
            deconv, error, errorBL = main(deconv, mask, sig, mask_mir)
            it += 1
            # if error<value:
            if errorBL < value:
                break


    if option=='error':
        it=0
        while True:
            deconv, error, errorBL=main(deconv,mask,sig,mask_mir)
            it=it+1
            if error<value:
                break
        print('number of iteration: {}'.format(it))
    return (deconv,error, errorBL, it)










