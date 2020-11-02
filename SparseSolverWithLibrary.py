#!/usr/bin/env python
# coding: utf-8

# In[65]:


from scipy.sparse.linalg import dsolve, spsolve, bicg, bicgstab, cg
from scipy.sparse.linalg import cgs, gmres, lgmres, lsmr
from scipy.sparse import csc_matrix
import numpy as np
import scipy.sparse as sp


# In[1]:


def vPrint(v, *args):  
    if v:
        print(*args)


# In[66]:


class LGMRES_with_Library:
    __instance = None
    library = []    

    @staticmethod 
    def getInstance():
        """ Static access method. """
        if LGMRES_with_Library.__instance == None:
            LGMRES_with_Library()
        return LGMRES_with_Library.__instance

    def __init__(self):
        """ Virtually private constructor. """
        self.threshDist = 0.01
        if LGMRES_with_Library.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            LGMRES_with_Library.__instance = self


# In[ ]:


class LSMR_with_Library:
    __instance = None
    library = []    

    @staticmethod 
    def getInstance():
        """ Static access method. """
        if LSMR_with_Library.__instance == None:
            LSMR_with_Library()
        return LSMR_with_Library.__instance

    def __init__(self):
        """ Virtually private constructor. """
        self.threshDist = 0.01
        if LSMR_with_Library.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            LSMR_with_Library.__instance = self


# In[ ]:


def computeProblemDistance(prob1, prob2):
    M1 = prob1['M']
    M2 = prob2['M']
    B1 = prob1['B']
    B2 = prob2['B']

    if (M1.shape != M2.shape):
#         print("    M1.shape != M2.shape")
        return 2
    elif np.sum(np.abs(B1-B2)) > 0:
#         print("    sum(abs(B1-B2))", np.sum(np.abs(B1-B2)))
        return 2
    else:
        return np.clip(np.mean(np.abs(M1.data - M2.data)), 0, 1)


# In[ ]:


def getSimilarProbSln(self, newProb):
    nCols = newProb['M'].shape[1]
    xZero = np.zeros_like(newProb['M'], shape = (nCols,))
    distances = [computeProblemDistance(newProb, oldProb) for oldProb in LGMRES_with_Library.library]
#     print("  distances:", distances)
    if len(distances) == 0:
        return None
    iMin = np.argmin(distances)
    minDist = distances[iMin]
    if minDist < self.threshDist:
        x0 = LGMRES_with_Library.library[iMin]['X']
        return x0
    else:
        return None
setattr(LGMRES_with_Library, "getSimilarProbSln", getSimilarProbSln)


# In[ ]:


def getSimilarProbSln(self, newProb):
    nCols = newProb['M'].shape[1]
    xZero = np.zeros_like(newProb['M'], shape = (nCols,))
    distances = [computeProblemDistance(newProb, oldProb) for oldProb in LSMR_with_Library.library]
#     print("  distances:", distances)
    if len(distances) == 0:
        return None
    iMin = np.argmin(distances)
    minDist = distances[iMin]
    if minDist < self.threshDist:
        x0 = LSMR_with_Library.library[iMin]['X']
        return x0
    else:
        return None
setattr(LSMR_with_Library, "getSimilarProbSln", getSimilarProbSln)


# In[ ]:


def my_lgmres(self, M, B, verbose = False):
    v = verbose
    vPrint(v,"LGMRES")
    newProb = {'M':M, 'B':B}
    X0 = self.getSimilarProbSln(newProb)
    if X0 is None:
        vPrint(v, "  Nothing in Library that is close.  Starting fresh")
        X, exitCode = lgmres(csc_matrix(newProb['M']), newProb['B'], atol=1e-6)
    else:
        vPrint(v, "  Starting from similar problem")
        X, exitCode = lgmres(csc_matrix(newProb['M']), newProb['B'], x0 = X0, atol=1e-6)
    if exitCode > 0 and X0 is not None:
        vPrint(v, "  Didn't converge with starting solution. Restarting fresh")
        X, exitCode = lgmres(csc_matrix(newProb['M']), newProb['B'], atol=1e-6)
    if exitCode == 0:
        vPrint(v, "  Solution was good.  Adding it to the library")
        LGMRES_with_Library.library.append({'M':M, 'B':B, 'X':X})
    else:
        vPrint(v, "  final exitCode:", exitCode)
    return (X, exitCode)
setattr(LGMRES_with_Library, "lgmres", my_lgmres)


# In[7]:


def my_lsmr(self, M, B, verbose = False):
    v = verbose
    vPrint(v,"LSMR")
    newProb = {'M':M, 'B':B}
    X0 = self.getSimilarProbSln(newProb)
    if X0 is None:
        vPrint(v,"  Nothing in Library that is close.  Starting fresh")
        solnPkg = lsmr(csc_matrix(newProb['M']), newProb['B'], atol=1e-6)
        X, exitCode = solnPkg[0:2]
    else:
        vPrint(v,"  Starting from similar problem")
        solnPkg = lsmr(csc_matrix(newProb['M']), newProb['B'], x0 = X0, atol=1e-6)
        X, exitCode = solnPkg[0:2] 
    if exitCode > 2 and X0 is not None:
        vPrint(v,"LSMR: Didn't converge with starting solution. Restarting fresh")
        solnPkg = lsmr(csc_matrix(newProb['M']), newProb['B'], atol=1e-6)
        X, exitCode = solnPkg[0:2]  
    if exitCode <= 2:
        vPrint(v,"  Solution was good.  Adding it to the library")
        LSMR_with_Library.library.append({'M':M, 'B':B, 'X':X})
    else:
        vPrint(v,"  final exitCode:", exitCode)
    return (X, exitCode)
setattr(LSMR_with_Library, "lsmr", my_lsmr)


# solver = LGMRES_with_Library.getInstance()

# """ lgmres(csc_matrix(self.M), self.B, x0=self.fullFields1D, tol=1e-6)""";

# B1 = np.array(sp.random(1000,1).todense()).flatten()
# B2 = B1.copy()
# B3 = B1.copy()

# M1 = sp.random(1000,1000)
# dM = 0.01*np.random.uniform(-1,1,size=M1.size)
# M2 = M1.copy()
# M2.data += dM
# dM = 0.1*np.random.uniform(-1,1,size=M1.size)
# M3 = M1.copy()
# M3.data += dM

# threshDist = 0.1

# solver.library
