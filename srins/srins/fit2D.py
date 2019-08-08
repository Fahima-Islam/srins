
# coding: utf-8

# # Fitting the 2D Resolution Function to a Gaussian-Ikeda Carpenter Function

#  Adding the path of the Ikeda Carpenter library

# import the Ikeda Carpenter Function

# In[1]:


import sys,os
sys.path.append('/home/fi0/dev/sandbox/SR-INS/srins/Jiao_Res/graphite-DOS/dgsres')
from dgsres import icg


# In[2]:


# this is a input parameter
import res_params_interped_funcs as rpif
reload(rpif)


# import the python libraries

# In[3]:


import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib notebook')
import histogram.hdf as hh, histogram as H
import lmfit
from pylab import genfromtxt;
from scipy.interpolate import interp1d
import math
from skimage import transform as tf
from skimage.transform import rotate
import roation as rot
from sklearn import preprocessing


# # resolution function from mcvine

# In[4]:


simdatadir = '/home/fi0/dev/sandbox/SR-INS/srins/Jiao_Res/graphite-DOS/Ei_300'


# In[15]:


Etransfer = 150.0
Qtransfer = 11.


# In[9]:


res_E_300=hh.load(os.path.join(simdatadir, str(Etransfer), "res-sim/iqe.h5" ))


# # grid creation with E and Q

# In[17]:


dE = res_E_300.E - Etransfer
dQ = res_E_300.Q - Qtransfer

print dE.shape
I=res_E_300.I.copy()
print I.shape
Egrid, Qgrid = np.meshgrid(dE, dQ)


# In[11]:


# dQ, dE


# # MAking the NAN values of RF zero

# In[ ]:


I[I!=I] = 0.0
print np.shape(I)
print I.shape[0]
print I.shape[1]


# In[19]:


plt.figure()
plt.pcolormesh(Qgrid,Egrid,I)
plt.show()


# # finding the index of the maximum intensity

# In[24]:


ind_dE_center = np.argmin(np.abs(dE))
ind_dQ_center = np.argmin(np.abs(dQ))


# # Finding the direction of the tails-PCA apllication

# #### Finding the index that have the intensities greater than zero

# In[107]:


#np.where to find the index


# In[28]:


ylA, xlA = np.where(I>0)
X = np.array([xlA, ylA]).T


# #### PCA Application

# In[29]:


from sklearn.decomposition import PCA
# pca = PCA(n_components=4)
pca = PCA()
pca.fit(X).transform(X)
# pca.fit(X)


# In[30]:


pca.components_


# In[31]:


angle=np.rad2deg(np.arccos(pca.components_[0,0]))
angle2=np.rad2deg(np.arccos(-pca.components_[0,0]))

print angle
print angle2


# In[32]:


if angle<angle2:
angleP=angle
else:
angleP=angle2


# In[33]:


angleP


# #### finding the line of the principal axis with coordinates

# In[35]:


dE_mxI = dE[ind_dE_center]
dQ_mxI = dQ[ind_dQ_center]


# In[36]:


yC=(spacing)*slope*(dE[xlA]-dE_mxI)+dQ_mxI # first principle axis
plt.figure()
plt.pcolormesh(dQ,dE,I.T)
plt.plot(yC, dE[xlA], label='pca1')
plt.clim(0, 1e-5)
plt.legend()
plt.show()


# # Rotation 

# In[37]:


print (angleP)
# Rotation_Bef=rot.rotate(I, angle2,idx_dQ_mxI,idx_dE_mxI, fill=0) ## with my code, dQ, dE
# E_Rotation_Bef=rot.rotate(I, angle2,idx_dE_mxI,idx_dQ_mxI, fill=0) ## with my code, dE, dQ
E_Rotation_Bef=rotate(I, angleP, resize=False, center=(ind_dE_center,ind_dQ_center)) ## with python code, dE, dQ


# In[39]:


plt.figure('original RF')
plt.pcolormesh(dQ, dE,I.T)
plt.show()

plt.figure('Rotation with pcolormesh')
plt.pcolormesh(dQ, dE,E_Rotation_Bef.T)
plt.show()

plt.figure('rotation')
plt.imshow(E_Rotation_Bef, label='rotation')
plt.show()
print dE[1]-dE[0]


