
import numpy as np
import math
from numpy import linalg as LA
from math import factorial



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

def bregman_NS(sig1,sig2, mask1,mask2, iniGuessU1,iniGuessU2,iniGuessV1,iniGuessV2, neu_N1,neu_N2, delta_ER1,delta_ER2,  option,value, maxIter=100): #neu_inverseNoise, delta_energyResolution
    """decolvolution using the Linearized Bregman Iteration with non-stationary resolution functions for two different positions of the detectors

                   Parameters
                   ----------
                   sig1=1 D array
                        partial data to be deconvolved
                    sig2=1 D array
                        partial data to be deconvolved
                    mask1 :2 D array
                        Corresponding Resolution Function for sig1
                    mask2 :2 D array
                        Corresponding Resolution Function for sig2
                   iniGuessV1 :array
                       initial guess of the sig1
                   iniGuessV2 :array
                        initial guess of the sig2
                   iniGuessU1 :array
                       initial guess of the sig1
                   iniGuessU2 :array
                        initial guess of the sig2
                   option=string
                       'iteration' or 'error'
                   value: integer or float
                       number of iteration or the tolerance value for error
                   """

    def main(sig1, sig2, mask1, mask2, iniGuessU1,iniGuessU2, iniGuessV1,iniGuessV2, neu_N1, neu_N2, delta_ER1,delta_ER2  ):

        sigC1=np.dot(iniGuessU1, mask1)
        sigC2=np.dot(iniGuessU2, mask2,)

        relative_blur1=sig1-sigC1
        relative_blur2= sig2-sigC2

        deconvV1 = iniGuessV1+ np.dot(relative_blur1, mask1.T )
        deconvV2=iniGuessV2+np.dot(relative_blur2,mask2.T )

        deconvU1=delta_ER1*shrink(deconvV1, neu_N1)
        deconvU2=delta_ER2 * shrink(deconvV2, neu_N2)

        deconvV=deconvV1+deconvV2
        deconvU=deconvU1+deconvU2
        # deconvU = delta_ER*shrink(deconvV, 1/neu_N)

        error=LA.norm(((deconvU1+deconvU2)-(iniGuessU1+iniGuessU2)))
        errorBL = LA.norm((np.dot(deconvU1,mask1)-sig1)+(np.dot(deconvU2,mask2)-sig2))


        iniGuessV1=deconvV1
        iniGuessV2 = deconvV2

        iniGuessU1 = deconvU1
        iniGuessU2 = deconvU2
        return(deconvV,deconvU,iniGuessV1,iniGuessV2,iniGuessU1,iniGuessU2,error,errorBL)

    if option=='iteration':
        error=0
        it=value
        for i in xrange(value):
            deconvV, deconvU, iniGuessV1, iniGuessV2, iniGuessU1, iniGuessU2, error, errorBL=main(sig1, sig2, mask1, mask2, iniGuessU1,iniGuessU2, iniGuessV1,iniGuessV2,neu_N1,neu_N2 ,delta_ER1, delta_ER2 )

    if option=='errorModel':
        it=0
        while it<maxIter:
            deconvV, deconvU, iniGuessV1, iniGuessV2, iniGuessU1, iniGuessU2, error, errorBL=main(sig1, sig2, mask1, mask2, iniGuessU1,iniGuessU2, iniGuessV1,iniGuessV2,neu_N1,neu_N2 ,delta_ER1, delta_ER2 )
            it=it+1
            # if error<value:
            if errorBL >value:
                break
        # print('number of iteration: {}'.format(it))

    if option == 'error':
        it = 0
        while it<maxIter:
            deconvV, deconvU, iniGuessV1, iniGuessV2, iniGuessU1, iniGuessU2, error, errorBL = main(sig1, sig2, mask1, mask2, iniGuessU1,iniGuessU2, iniGuessV1,iniGuessV2,neu_N1,neu_N2 ,delta_ER1, delta_ER2 )
            it = it + 1
            #print("Iteration #%s" % it)
            if error<value:
                break
        print('number of iteration: {}'.format(it))

    return ( deconvU, error,it,errorBL)










