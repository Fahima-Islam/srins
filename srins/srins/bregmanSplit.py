
import numpy as np
import math
from numpy import linalg as LA
from math import factorial



def split_Bregman(sig, mask, initial_d, initial_b, muI, lamdaI, value, maxiter):
    """decolvolution using the split Bregman Iteration with non-stationary Resolution Function

                       Parameters
                       ----------
                       sig=array
                           data to be deconvolved
                       mask :array
                           Resolution Function
                       initial_d :array
                           Bragmen Parameter
                       initial_b :array
                           Bragmen Parameter
                       mu=float
                           noise controlling parameter
                        lamda=float
                           step size
                       ninner: integer
                           number of iteration for inner loop
                       nouter: integer
                           number of iteration for outer loop
                        max_cg: integer
                            number of iteration for conjugate gradient
                       """
    sigT=sig.transpose()
    mu=1/muI
    lamda=1/lamdaI
    maskT = mask.transpose()

    uk=np.dot(sig,maskT)

    dk_x=initial_d

    bk_x=initial_b
    fk = sig
    it = 0
    while it < maxiter:
        for jinner in xrange(1):
            ukp=uk
            ifkt=np.dot(sig,maskT)
            rhs=mu*ifkt+lamda*(dk_x-bk_x)

            ruk = np.dot(uk,mask)
            iukt = np.dot(ruk,maskT)
            r = rhs - mu * iukt -lamda *uk
            p = r
            rsold = np.dot(r.transpose(), r)

            for i in xrange(1):
                rp=np.dot(p,mask)
                irpt = np.dot(rp,maskT)
                Ap = mu * irpt + lamda *p

                alpha = rsold / np.dot(p.transpose(),Ap)
                uk = uk + alpha * p
                r = r - alpha * Ap
                rsnew = np.dot(r.transpose(),r)
                if rsnew < 1e-32:
                    break

                p = r + rsnew / rsold * p;
                rsold = rsnew

            sk_x = uk + bk_x
            dk_x = np.maximum(np.abs(sk_x)-1/lamda,0)*np.sign(sk_x)
            bk_x = sk_x-dk_x

        fk = fk + sigT - np.dot(uk,mask)
        diff = np.dot(uk, mask) - sig
        errorBL = np.sqrt(np.average(diff ** 2))
        it = it + 1
        if errorBL < value:
            break
    rec_tv = uk

    return (uk, it)











