
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



def deconvolve_L1_NS(sig,mask,deconV,eps,rgP,option,value,maxIter=100):
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

    def main(deconv,mask,sig0,mask_mir,eps,rgP):
            sigC=convolve_NS(deconv,mask)
            relative_blur=sig0/sigC
            with np.errstate(divide='ignore'):
                    relative_blur[np.isinf(relative_blur)] = -2
            norm=np.sqrt(deconv**2)
            mod_norm=np.sqrt(eps**2+norm**2)
            div_rgp=rgP*mod_norm
            deconvP=(deconv/(1-(div_rgp)))*convolve_NS(relative_blur,mask_mir)
            error=LA.norm(deconvP-deconv)/ deconv.size
            diff = convolve_NS(deconv, mask) - sig
            errorBL = np.sqrt(np.average(diff ** 2))
            # errorBL = (LA.norm(convolve_NS(deconv, mask) - sig)) / deconv.size
            deconv=deconvP
            return(deconv,error, errorBL)

    if option=='iteration':
            error=0
            for i in xrange(value):
                    deconv,error, errorBL=main(deconv,mask,sig0,mask_mir,eps,rgP)

    if option=='errorModel':
        it=0
        while it< maxIter:
            deconv, error, errorBL=main(deconv,mask,sig0,mask_mir,eps,rgP )
            it=it+1
            # if error<value:
            if errorBL <value:
                break


    if option=='error':
            it=0
            while True:
                    deconv,error, errorBL=main(deconv,mask,sig0,mask_mir,eps,rgP)
                    it=it+1
                    if np.all(error<value):
                            break
            print('number of iteration: {}'.format(it))
    return(deconv,error, errorBL,it)








