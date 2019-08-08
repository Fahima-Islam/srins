from __future__ import (absolute_import, division, print_function)

import os
import pytest
import numpy as np
from matplotlib import pyplot as plt
import numpy as np, histogram.hdf as hh, histogram as H
from scipy.interpolate import interp1d
import warnings
from pylab import genfromtxt;
from numpy import linalg as LA
from srins.powder.bregmanSplit import split_Bregman as S
from srins.powder.conv_deconv import convolve_NS


here = os.path.abspath(os.path.dirname(__file__) or '.')
data = os.path.join(here, '..', 'data')

def test_bregmanSplit():
    RF3_read = genfromtxt(os.path.join(data, "RF3_save.txt"))
    dos_300 = hh.load(os.path.join(data, 'graphite-Ei_300-DOS.h5'))
    DOS_300 = dos_300.I
    spacing = dos_300.E[1] - dos_300.E[0]
    energytobin = np.int(220 / spacing)
    DOS_300[energytobin::] = 0
    E_axis_DOS = dos_300.E
    E_axis_res = np.arange(-50, 240, 0.1)
    left = E_axis_res[E_axis_res < E_axis_DOS[0]]
    extended_E_axis_DOS = np.concatenate((left, E_axis_DOS))
    extended_g = np.concatenate((np.zeros(left.shape), dos_300.I))
    dos_intrp1 = interp1d(extended_E_axis_DOS, extended_g, kind='cubic')  # interpolation
    interpolated_dos = dos_intrp1(E_axis_res)
    E, g = E_axis_res, interpolated_dos
    RF_T = np.transpose(RF3_read)
    m = convolve_NS(RF_T, RF3_read)
    delta = 2. / LA.norm(m, ord=1)

    initial_d = np.zeros(g.shape[0])
    initial_b = initial_d
    mu = 100
    lamda = 1
    ninnner = 1
    nouter = 20
    max_cg = 500
    R0 = S(g, RF3_read, initial_d, initial_b, mu, lamda, ninnner, nouter, max_cg)

    plt.figure()
    plt.plot (E_axis_res,R0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    pytest.main([os.path.abspath(__file__)])
