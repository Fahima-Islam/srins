
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



def deconvolve_TV_NS(sig,mask,deconV,eps,rgP,option,value):
    """decolvolution using the Lucy-Richardson algorithm with total variation minimization regularization for nonstationary RF

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
    mask_mir=mask[...,::-1]
    #m_tst=F.convolve(sig,mask,conv)
    deconv = deconV

    def main(deconv,mask,sig0,mask_mir,eps,rgP):
            sigC=convolve_NS(deconv,mask)
            relative_blur=sig0/sigC
            with np.errstate(divide='ignore'):
                    relative_blur[np.isinf(relative_blur)] = -2
            grad=np.gradient(deconv)
            norm=np.sqrt(grad**2)
            mod_norm=np.sqrt(eps**2+norm**2)
            division=(grad)/mod_norm
            division[np.isnan(division)] = 0.0
            with np.errstate(divide='ignore'):
                    division[np.isinf(division)] = -2
            divergence=np.gradient(division)
            div_rgp=rgP*divergence
            deconvP=(deconv/(1-(div_rgp)))*convolve_RelBlur(relative_blur,mask_mir)
            error=LA.norm(deconvP-deconv)
            deconv=deconvP
            return(deconv,error)

    if option=='iteration':
            error=0
            for i in xrange(value):
                    deconv,error=main(deconv,mask,sig0,mask_mir,eps,rgP)


    if option=='error':
            it=0
            while True:
                    deconv,error=main(deconv,mask,sig0,mask_mir,eps,rgP)
                    it=it+1
                    if np.all(error<value):
                            break
            print('number of iteration: {}'.format(it))
    return(deconv)












