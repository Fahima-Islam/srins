
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

    con= np.dot(mask,sig)

    return(con)


def convolve_RelBlur(Iin,k):
    convp=np.dot(Iin,k)
    conv=convp[...,::-1]
    return (conv)


def shrink(y,a):
    L1norm=LA.norm(y, ord=1)
    # print L1norm
    r=(y/L1norm)*np.max([L1norm-a,0])
    return r


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
    def main(iniGuessU, iniGuessV, mask,sig,mask_mir,neu_N, delta_ER ):
        sigC=convolve_NS(iniGuessU,mask)
        relative_blur=sig-sigC
        # with np.errstate(divide='ignore'):
        #     relative_blur[np.isinf(relative_blur)] = -2
        deconvV = iniGuessV + convolve_NS(relative_blur, mask_mir)
        deconvU = delta_ER*shrink(deconvV, 1/neu_N)
        error=LA.norm((deconvV-iniGuessV))
        errorBL = LA.norm(convolve_NS(deconvV, mask) - sig)
        iniGuessV=deconvV
        iniGuessU = deconvU
        return(iniGuessV,iniGuessU,error,errorBL)

    if option=='iteration':
        error=0
        it=value
        for i in xrange(value):
            iniGuessV, iniGuessU,error,errorBL=main(iniGuessU, iniGuessV, mask,sig,mask_mir,neu_N, delta_ER )

    if option=='errorModel':
        it=0
        while it<maxIter:
            iniGuessV, iniGuessU,error,errorBL=main(iniGuessU, iniGuessV, mask,sig,mask_mir,neu_N, delta_ER )
            it=it+1
            # if error<value:
            if errorBL >value:
                break
        
        print('number of iteration: {}'.format(it))

    if option == 'error':
        it = 0
        while it<maxIter:
            iniGuessV, iniGuessU, error, errorBL = main(iniGuessU, iniGuessV, mask, sig, mask_mir, neu_N, delta_ER)
            it = it + 1
            #print("Iteration #%s" % it)
            if error<value:
                break
       
        print('number of iteration: {}'.format(it))
        
    iniGuessV[iniGuessV<0]=0
    iniGuessU[iniGuessU<0]=0
    return (iniGuessV,iniGuessU,error,it,errorBL)










