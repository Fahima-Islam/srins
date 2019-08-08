
import numpy as np
import math
from numpy import linalg as LA
from math import factorial


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')



def convolve_all(Iin, RF, opt):
    Iin = Iin.copy()
    RF = RF.copy()
    maskZero = Iin[..., :] <= 0
    maxVal = np.max(Iin)
    Iin[maskZero] = maxVal + 1

    index = Iin.ndim
    try:
        shapeSig = (Iin.shape[0], Iin.shape[1])
        M = [np.meshgrid(Iin[i], Iin[i]) for i in xrange(shapeSig[0])]
        M = np.array(M)[:, 0, ...]

    except IndexError:
        shapeSig = (1, Iin.shape[0])
        M, T = np.meshgrid(Iin, Iin)

    try:
        shapeRF = (RF.shape[0], RF.shape[1])
    except IndexError:
        shapeRF = (1, RF.shape[0])

    if shapeSig[1] != shapeRF[0]:
        rp = np.tile(RF[shapeRF[0] - 1::], (((shapeSig[1]) - shapeRF[0]), 1))
        RF = np.vstack((RF, rp))

    try:
        shapeRF = (RF.shape[0], RF.shape[1])
    except IndexError:
        shapeRF = (1, RF.shape[0])

    il = np.tril(M[..., :(shapeRF[1] - 1), :])
    iu = np.triu(M)

    mask = il[..., :, ::] > 0
    mask_1 = iu[..., :, ::] > 0

    justified_mask = np.sort(mask[..., :, ::], index)
    justified_mask_1 = np.sort(mask_1[..., :, ::], index)
    justified_mask = justified_mask[..., :, ::]
    justified_mask_1 = justified_mask_1[..., :, ::-1]
    out = np.zeros_like(il[..., :, :])
    out_1 = np.zeros_like(iu[..., :, :])
    out[justified_mask] = il[..., :, :, ][mask]
    out_1[justified_mask_1] = iu[..., :, :][mask_1]
    mask_maxval = out == maxVal + 1
    out[:, :][mask_maxval] = 0

    mask1_maxval = out_1 == maxVal + 1
    out_1[:, :][mask1_maxval] = 0

    if index > 1:
        mod_input = np.hstack((out, out_1))
    else:
        mod_input = np.vstack((out, out_1))

    RFZero = RF[..., :] <= 0
    maxValRF = np.max(RF)
    RF[RFZero] = maxValRF + 1

    diags = [np.concatenate((RF[:, ::-1].diagonal(i), np.zeros(shapeSig[1] - len(RF[:, ::-1].diagonal(i)))), axis=0) for
             i in range(-shapeRF[0] + 1, shapeRF[1])]
    diags = np.array(diags[::-1])

    mask = diags[:(shapeRF[1] - 1), :] > 0
    justified_mask = np.sort(mask, 1)
    diags[:(shapeRF[1] - 1), :][justified_mask] = diags[:(shapeRF[1] - 1), :][mask]
    diags[:(shapeRF[1] - 1), :][~justified_mask] = 0

    maskD_maxval = diags == maxValRF + 1
    diags[:, :][maskD_maxval] = 0

    if index > 1:
        multi = np.vstack(mod_input * diags)
    else:
        multi = mod_input * diags

    convolve = np.sum(multi, axis=1)

    if index > 1:
        convolve = convolve.reshape(shapeSig[0], len(convolve) / shapeSig[0])

    if (shapeRF[1]) % 2 == 0:
        start = ((shapeRF[1]) / 2) - 1
    else:
        start = (shapeRF[1]) / 2

    same = convolve[..., start:start + shapeSig[1]]

    if opt == 'full':
        return (convolve)
    if opt == 'same':
        return (same)


def convolve(Iin, k, opt):
    # str_arr_I = raw_input('insert only the values of the first 1D array with the space between them:').split(' ')
    # Iin=0.0* np.ones(len(str_arr_I))
    # for i,j in zip (str_arr_I, xrange(len(str_arr_I))):
    # Iin[j]=i
    # str_arr_k = raw_input('insert only values of the second 1D array with the space between them:').split(' ')
    # k=0.0* np.ones(len(str_arr_k))
    # for i,j in zip (str_arr_k, xrange(len(str_arr_k))):
    # k[j]=i
    Iin = Iin.copy()
    maskZero = Iin <= 0
    maxVal = np.max(Iin)
    Iin[maskZero] = maxVal + 1

    kT = k[::-1]
    if len(Iin) < len(k):
        kT = Iin[::-1]
    kT0 = kT
    kT1 = kT
    if len(k) != len(Iin):
        kT_ = [0] * abs(len(k) - len(Iin))
        kT0 = np.concatenate((kT_, kT), axis=0)
        kT1 = np.concatenate((kT, kT_), axis=0)

    length_s = 0
    if len(Iin) > len(k):
        length_s = len(Iin)
    else:
        length_s = len(k)

    if len(Iin) < len(k):
        Iin = k

    M, T = np.meshgrid(Iin, Iin)
    il = np.tril(M[:(len(kT) - 1), :])
    iu = np.triu(M)
    mask = il > 0
    mask_1 = iu > 0
    justified_mask = np.sort(mask, 1)
    justified_mask_1 = np.sort(mask_1, 1)
    justified_mask = justified_mask[:, ::]
    justified_mask_1 = justified_mask_1[:, ::-1]
    out = np.zeros_like(il[:, :])
    out_1 = np.zeros_like(iu)
    out[justified_mask] = il[:, :][mask]
    out_1[justified_mask_1] = iu[mask_1]

    mask_maxval = out == maxVal + 1
    out[:, :][mask_maxval] = 0

    mask1_maxval = out_1 == maxVal + 1
    out_1[:, :][mask1_maxval] = 0

    Rr = np.concatenate((np.dot(out, kT0), np.dot(kT1, out_1)), axis=0)
    # print("for full mode {}".format(Rr) )
    if (len(kT)) % 2 == 0:
        start = ((len(kT)) / 2) - 1
    else:
        start = (len(kT)) / 2
    realSame = Rr[start:start + length_s]
    length_f = len(Rr)
    off = length_f - length_s
    off_eachend = off / 2
    f_idx = int(off_eachend)
    e_idx = f_idx + length_s
    Sr = Rr[f_idx:e_idx]
    # print("for same mode {}".format(Sr))
    if opt == 'full':
        return (Rr)
    if opt == 'same':
        return (Sr)
    if opt == 'sameR':
        return (realSame)


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


# def shrink(y,a):
#     if y>a:
#         r=y-a
#
#     if -a<y<a:
#         r=0
#
#     if y<a:
#         r=y+a
#
#     return r

def shrink(y,a):
    L1norm=LA.norm(y, ord=1)
    # print L1norm
    r=(y/L1norm)*np.max([L1norm-a,0])
    return r

def FWHM(Y,X):
    d = Y - (max(Y) / 2)
    indexes = np.where(d > 0)[0]
    return abs(X[indexes[-1]] - X[indexes[0]])

def scale(Y, minS,maxS):
    zeroTO1=(Y-np.min(Y))/(np.max(Y)-np.min(Y))
    scaling=(zeroTO1*(maxS-minS))+minS
    return scaling

def split_Bregman(sig, mask, initial_d, initial_b, mu, lamda, ninnner,nouter, max_cg):
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
    sigT=sig[np.newaxis].transpose()

    maskT = mask.transpose()

    uk=np.dot(maskT, sigT)

    dk_x=initial_d[np.newaxis].transpose()

    bk_x=initial_b[np.newaxis].transpose()
    fk = sigT
    for jouter in xrange (nouter):
        for jinner in xrange(ninnner):
            ukp=uk
            ifkt=np.dot(maskT, sigT)
            rhs=mu*ifkt+lamda*(dk_x-bk_x)

            ruk = np.dot(mask, uk)
            iukt = np.dot(maskT,ruk)
            r = rhs - mu * iukt -lamda *uk
            p = r
            rsold = np.dot(r.transpose(), r)

            for i in xrange(max_cg):
                rp=np.dot(mask,p)
                irpt = np.dot(maskT ,rp)
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

        fk = fk + sigT - np.dot(mask, uk)
    rec_tv = uk

    return (uk)



def bregman_NS(sig, mask,iniGuessV, iniGuessU, neu_N, delta_ER, option,value): #neu_inverseNoise, delta_energyResolution
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
        while True:
            iniGuessV, iniGuessU,error,errorBL=main(iniGuessU, iniGuessV, mask,sig,mask_mir,neu_N, delta_ER )
            it=it+1
            # if error<value:
            if errorBL >value:
                break
        print('number of iteration: {}'.format(it))

    if option == 'error':
        it = 0
        while True:
            iniGuessV, iniGuessU, error, errorBL = main(iniGuessU, iniGuessV, mask, sig, mask_mir, neu_N, delta_ER)
            it = it + 1
            if error<value:
                break
        print('number of iteration: {}'.format(it))

    return (iniGuessV,iniGuessU,error,it,errorBL)



def deconvolve_NS(sig,mask,deconV,option,value):
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
    mask_mir=mask[...,::-1]
    deconv = deconV

    def main(deconv,mask,sig0,mask_mir):
        sigC=convolve_NS(deconv,mask)
        relative_blur=sig0/sigC
        with np.errstate(divide='ignore'):
            relative_blur[np.isinf(relative_blur)] = -2
        deconvP=deconv*convolve_RelvBlur(relative_blur,mask_mir)
        error=LA.norm((deconvP-deconv))
        deconv=deconvP
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
            error=np.abs(deconvP-deconv)
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



def deconvolve_L1_NS(sig,mask,deconV,eps,rgP,option,value):
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
    mask_mir=mask[...,::-1]
    deconv = deconV

    def main(deconv,mask,sig0,mask_mir,eps,rgP):
            sigC=convolve_NS(deconv,mask)
            relative_blur=sig0/sigC
            with np.errstate(divide='ignore'):
                    relative_blur[np.isinf(relative_blur)] = -2
            norm=np.sqrt(deconv**2)
            mod_norm=np.sqrt(eps**2+norm**2)
            div_rgp=rgP*mod_norm
            deconvP=(deconv/(1-(div_rgp)))*convolve_RelBlur(relative_blur,mask_mir)
            error=np.abs(deconvP-deconv)
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








