#!/usr/bin/env python

from __future__ import (absolute_import, division, print_function)


import os, sys
import numpy as np
import matplotlib as mpl; mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np, histogram.hdf as hh, histogram as H
from scipy.interpolate import interp1d
import warnings
from numpy import linalg as LA


from dgsres import icg
import srins.powder.linearizedBregman as splb; reload(splb)
import srins.powder.conv_deconv as spcd


here = os.path.dirname(__file__ or '.')


def res_func(Eaxis, E0, interped_funcs, Ei, geom):
    "return resolution function y array for the given x axis (Eaxis). E is the energy transfer"
    params = dict()
    for name in interped_funcs.keys():
        value = interped_funcs[name](E0)
        params[name] = value
    y = icg.resolution(Eaxis, Ei=300., E0=E0, geom=geom, **params)
    y/=y.sum()
    return y


def test():
    # define E axis
    Eaxis = np.arange(-50, 240, 0.1)
    # load interpolated resolution functions
    import res_params_interped_funcs_Ei_300 as rpif
    # instrument geometry
    geom = icg.Geom(l1=11.6, l2=2.0, l3=3.)
    # fitting parameters plot
    variables = 't0 sigma a b R'.split()
    plt.figure(figsize=(6,8))
    for i,v in enumerate(variables):
        plt.subplot(3, 2, i+1)
        plt.plot(Eaxis, rpif.interped_funcs[v](Eaxis))
        plt.legend()
        continue
    plt.subplot(3, 2, 6)
    plt.plot(Eaxis, Eaxis, '-o', label='E')
    plt.legend()
    plt.tight_layout()
    plt.savefig("res-func-params-plot.png")
    # resolution funcs plot
    plt.figure(figsize=(9, 4))
    for E0 in np.arange(-50., 300., 15.):
        x = Eaxis
        y = res_func(x, E0, rpif.interped_funcs, Ei=300, geom=geom)
        plt.plot(x, y)
    plt.savefig("res-funcs-plot.png")
    # matrix
    res_matrix = np.zeros( (Eaxis.size, Eaxis.size) )
    # extend the E axis so that the normalization is correct
    extended_Eaxis = np.arange(-80, 280, 0.1) 
    startindex = np.where(np.isclose(extended_Eaxis, Eaxis[0]))[0][0]
    endindex = np.where(np.isclose(extended_Eaxis, Eaxis[-1], atol=0.0001))[0][0]
    from tqdm import tqdm
    # fill the matrix with resolution functions
    for i, E1 in tqdm(enumerate(Eaxis)):
        y1 = res_func(extended_Eaxis, E1, rpif.interped_funcs, Ei=300, geom=geom)
        res_matrix[i] = y1[startindex: endindex+1]
    # plot resmatrix
    plt.figure()
    plt.imshow(res_matrix)
    # plt.clim(0, 0.025)
    plt.colorbar()
    plt.savefig("resmat.png")
    # improve sanity of resolution matrix
    res_matrix[res_matrix<0] = 0
    # res_matrix[np.isnan(res_matrix)]
    #
    # DOS
    doshist = hh.load(os.path.join(here, '..', 'data', 'graphite-Ei_300-DOS.h5'))
    g = doshist.I
    E = doshist.E
    # interpolate using the predefined E axis
    g1 = np.interp(Eaxis, E, g)
    g1[Eaxis>217] = 0 # this is kind of arbitrary
    #
    # deconvolve
    from srins.powder.conv_deconv import convolve_NS as F
    RF_T = np.transpose(res_matrix)
    m = F(RF_T, res_matrix)
    maxdelta = 2. / LA.norm(m, ord=1)
    delta = maxdelta/1.2
    ini_uZ = np.zeros(Eaxis.shape[0])
    ini_vZ = np.zeros(Eaxis.shape[0])
    neu_N = 1 / 0.0005
    RV, RU, error, it, errorBL = splb.bregman_NS(g1, res_matrix, ini_vZ, ini_uZ, neu_N, delta, 'error', 0.05, maxIter=50)
    # check outputs
    plt.figure()
    plt.plot (Eaxis,RV)
    plt.legend()
    plt.savefig("DOS-SR.png")
    expected = np.loadtxt('expected/DOS-SR.txt')
    assert np.allclose(expected, np.array([Eaxis, RV]).T)
    return


if __name__ == '__main__': test()



