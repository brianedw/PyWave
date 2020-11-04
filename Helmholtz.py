#!/usr/bin/env python
# coding: utf-8

# ## Imports

# Stock Imports

# In[ ]:


import math
import numpy as np
import scipy

from numba import njit

import numpy.linalg as dLA
import scipy.sparse.linalg as sLA
import scipy.sparse as sp
import scipy.sparse.linalg as sLA
from scipy.sparse.linalg import dsolve, spsolve, bicg, bicgstab, cg
from scipy.sparse.linalg import cgs, gmres, lgmres
from scipy.sparse import csc_matrix

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML


# Custom Imports

# In[ ]:


from colorize import *
from SparseSolverWithLibrary import LGMRES_with_Library


# In[ ]:


LGMRESLib = LGMRES_with_Library.getInstance()


# In[ ]:


pi = np.pi
j = 1j


# ## Notes

# Notes on coodinate systems:
#  - `(x, y)`: For display and specification.  Origin is at 0,0 in the lower left hand corner.
#  - `(i, j)`: Location in matrix. i is row, j is column.  (i, j) = (y, x)

# ## Function Library

# ### Physics

# In[ ]:


NORMAL_CODE = 0
ZERO_CODE = 1
DERZERO_CODE = 2


# In[ ]:


def pTypeToCode(str):
    if str == "normal":
        return NORMAL_CODE
    elif str == "zero":
        return ZERO_CODE
    elif str == "derZero":
        return DERZERO_CODE


# In[ ]:


class EMSim:
    pass


# In[ ]:


def __init__(self, shape=(10, 10), WL0=20, margin='auto'):
    """ EMSim is a simple 2D Finite Difference Frequency Domain Simulator 
    (FDFD).
    
    It was primarily designed as a teaching/research tool.  As such, it is
    all fully exposed with all methods and attributes being public.
    
    Args:
        shape (:tuple:`int`): The width (x) and height (y) in pixels of the 
            simulation sans any boundary that is added via `margin'.
        WL0 (real number): The length of a free space wavelength in pixels.
        margin (real number or 'auto'): How many pixels to surround the
            siumulation domain.  These pixels will be used to create an 
            absorbing boundary.  If 'auto' is used, `margin = 2*WL0`.

    Attributes:
        shape (:tuple:`int`): The width and height of the simulation sans
            any boundary that is added.
        shapeFull (:tuple:`int`): The width and height of the simulation
            including the margin.

        All three of the following access the same data block with views:
        nIndex (:np.array:(`complex`)):  The optical index of the simulation
            within the margin
        nIndexFull (:np.array:(`complex`)):  The optical index of the simulation
            including the margin.
        nIndexFull1D (:np.array:(`complex`)):  The optical index of the 
            simulation including the margin as a 1D vector as opposed to a 2D 
            array.
            
        All three of the following access the same data block with views:
        fields (:np.array:(`complex`)):  The result of the simulation within
            the margin
        fieldsFull (:np.array:(`complex`)):  The result of the simulation
            including the margin.
        fieldsFull1D (:np.array:(`complex`)):  The result of the simulation
            including the margin as a 1D vector as opposed to a 2D array.
        
        All three of the following access the same data block with views:
        sources (:np.array:(`complex`)):  The sources of the simulation within
            the margin
        sourcesFull (:np.array:(`complex`)):  The sources of the simulation
            including the margin.
        sourcesFull1D (:np.array:(`complex`)):  The sources of the simulation
            including the margin as a 1D vector as opposed to a 2D array.
        
        pType (:np.array:(`int`)): The category that the pixel falls into.
            [0: normal, 
             1: field == 0, 
             2: derivatives == 0] 
             within the margin
        catFull (:np.array:(`int`)): The category that the pixel falls into 
            including the margin.
        catFull1D (:np.array:(`int`)):  The category that the pixel falls into 
            including the margin as a 1D vector as opposed to a 2D array.

    """
    
    self.NORMAL_CODE = 0
    self.ZERO_CODE = 1
    self.DERZERO_CODE = 2
    
    if margin=='auto':
        self.margin = round(2*WL0)
    else:
        self.margin = margin
    self.shape = shape
    self.WL0 = WL0  # Size of wl0 in 'pixels'
    # self.margin = int(math.ceil(self.WL0))  # Size of PML in 'pixels'
    self.shapeFull = tuple(
        x + 2 * self.margin for x in self.shape)  # shape with PML
    # self.k0 = 2 * np.pi / WL0  # k0, free-space avenumber.
    
    m = self.margin
    shapeFull = self.shapeFull

    # The three vars below are all views of the same data block
    self.fieldFull = np.zeros(shapeFull, np.complex)
    self.field = self.fieldFull[m:-m, m:-m]
    self.fieldFull1D = self.fieldFull.ravel()

    # The three vars below are all views of the same data block
    self.sourceFull = np.zeros(shapeFull, np.complex)
    self.source = self.sourceFull[m:-m, m:-m]
    self.sourceFull1D = self.sourceFull.ravel()
    
    # The three vars below are all views of the same data block.
    # Normal: 0, field[x,y] == 0 : 1, d field[x,y]/du == 0 : 2
    self.pTypeFull = np.full(shapeFull, 0, np.uint8)  
    self.pType = self.pTypeFull[m:-m, m:-m]
    self.pTypeFull1D = self.pTypeFull.ravel()

    # The three vars below are all views of the same data block
    self.nIndexFull = np.full(shapeFull, 1., np.complex)  
    self.nIndex = self.nIndexFull[m:-m, m:-m]  
    self.nIndexFull1D = self.nIndexFull.ravel()

    
setattr(EMSim, "__init__", __init__)


# In[ ]:


def BuildSimBounds(self, c=.5, a=2):
    """
    Builds the margins of the simulation.
    
    The margins are built by padding the `nIndex` to `nIndexFull`
    and `cat` to `catFull`.  They are padded using the numpy.pad with the 'edge'
    option.  Whatever geometry is one the edge will copied into the margin.
    
    This should be called after all the changes to nIndex have been completed.
    
    Additionally, a loss term is added to the imaginary part of nIndex values 
    within the margin.  These losses are added according to the formula 
    `s(v) = c * v**a` where `v` is the relative distance into the margin.  The
    value `c` is maximum value to be added and `a` gives the growth rate.
    """
    m = self.margin
    WL0 = self.WL0
    (xMax, yMax) = self.shapeFull
    # Pad nIndex
    padded = np.pad(self.nIndex, ((m,m), (m,m)), 'edge')
    self.nIndexFull[:] = padded[:]
    def s(x):
        return((c) * (x)**a)
    sigmaX = [s((1/m)*max(m-i, 0, i-(xMax-m)+1)) for i in range(0,xMax)]
    sigmaY = [s((1/m)*max(m-i, 0, i-(yMax-m)+1)) for i in range(0,yMax)]
    edgeLoss = np.apply_along_axis(np.max, 0, np.meshgrid(sigmaY, sigmaX))
    self.nIndexFull[:] += 1j*edgeLoss
    # Pad pType
    padded = np.pad(self.pType, ((m,m), (m,m)), 'edge')
    self.pTypeFull[:] = padded[:]

setattr(EMSim, "BuildSimBounds", BuildSimBounds)


# In[ ]:


def solve(self):
    """
    Solves the M.X == B system for X, ie the fieldsFull1D.
    """
    X, code = lgmres(csc_matrix(self.M), self.B, atol=1e-6)
    self.fieldFull1D[:] = X
setattr(EMSim, "solve", solve)


# In[ ]:


def solveFromLast(self):
    """
    Solves the M.X == B system for X, ie the fieldsFull1D.
    
    Begins at the current state of fieldsFull1D.  This can be useful if only
    a small change had been made.
    """
    X, code = lgmres(csc_matrix(self.M), self.B, x0=self.fullFields1D, atol=1e-6)
    self.fieldFull1D[:] = X
setattr(EMSim, "solveFromLast", solveFromLast)


# In[ ]:


def solveFromLibrary(self):
    """
    Solves the M.X == B system for X, ie the fieldsFull1D.
    
    Searches the LGRMRESLib for the most similar problem that had been solved
    and uses this as the starting conditions.  This can be useful if only
    a small change had been made to some problem in the past.
    """
    X, code = LGMRESLib.lgmres(csc_matrix(self.M), self.B, verbose=False)
    self.fieldFull1D[:] = X
setattr(EMSim, "solveFromLibrary", solveFromLibrary)


# ### Indexing Functions

# Regarding Indexing:
# 
# A choice needed to be made with regards to how to store the data.  There were several options, all mired in historical convention.  When thinking about a point on a 2D plane, it is convention to consider it as (x, y) where x denotes travel to the right and y denotes travel upwards.  However, when one considers how a 2D matrix is written, the A[3,0] would be the 4 down from the top left corner.  Matrices, like images, are written from the upper left corner right and then down.  How do we choose to reconcile this?
# 
# In this work, it is less important how the matrices are stored and more important that the user interface is not confusing.  Therefore, we adopt the convention that A[0,0] refers to the origin is the in the lower-left corner and A[3,0] would the four values to the right of the origin. 

# In[ ]:


@njit
def indFull(xy, shapeFull):
    """ Converts from indexing of "full" arrays to "full1D" arrays.
    
    For instance, i1D = Ind((x,y), (xLength, yLength)) will convert an index 
    such that [A]Full[i1D] and [A]Full[x, y]] address the same value, where 
    [A] is `field`, `nIndex`, `source`, etc.

    It performs an identical function to 
    numpy.ravel_multi_index((y,x), (yMax, xMax), mode=('wrap', 'wrap'))

    Args:
        xy ( tuple:(int, int) ): The index in 2D, (x, y)
           ( tuple:( np.array(int), np.array(ints) ): The indices in 2D, ([xs], [ys])
        shapeFull (tuple: (int, int)): The second parameter.

    Returns:
        indices: The returned index/indices in 1D.

    """
    (xMax, yMax) = shapeFull
    (x, y) = xy
    (xWrap, yWrap) = (x % xMax, y % yMax)
    (i, j) = (yWrap, xWrap)
    inds = yWrap + (xWrap * yMax)
    
#     inds = np.ravel_multi_index(array, (yMax, xMax), mode=('wrap','wrap'))
    return inds


# In[ ]:


def ind(self, xy):
    """
    This returns the full1D index of a location for a point (x,y) in the 
    primary domain (ie no margins).
    xy = (3, 3)
    xy = ([1,2,3], [10, 12, 14])
    xy = ([1,2,3], 10)
    xy = (1, [10, 12, 14])
    """
    x,y = xy
    return indFull((x+self.margin, y+self.margin), self.shapeFull)
setattr(EMSim, "ind", ind)


# In[ ]:


def indRect(self, xyRange):
    """
    xyRange = ((xMin, xMax), (yMin, yMax))
    """
    ((xMin, xMax), (yMin, yMax)) = xyRange
    xLin = np.arange(xMin, xMax+1)
    yLin = np.arange(yMin, yMax+1)
    xArray, yArray = np.meshgrid(xLin, yLin)
    xFlat = xArray.flatten()
    yFlat = yArray.flatten()
    indFlat = self.ind((xFlat, yFlat))
    return indFlat
setattr(EMSim, "indRect", indRect)


# ### Optimized Equation Generator

# In[ ]:


def buildSparsePhysicsEqs(self):
    """
    Builds M and B of the matrix system M.X == B, where 
        X is the as of yet unknown fieldsFull1D
        B is essentially the sourcesFull1D
        and M is the relationship capturing the physics of the Helmholtz
        equations including the pixType optical index.
        
        
        
    """
    k0 = 2*pi/self.WL0
    n = self.fieldFull.size
    
    rslt = allEqGen(self.shapeFull, self.nIndexFull1D, self.pTypeFull1D, 
                    self.sourceFull1D, k0)
    (self.rList, self.cList, self.vList, self.bList) = rslt
    coo = sp.coo_matrix((self.vList, (self.rList, self.cList)), (n, n))
    self.M = coo
    self.B = np.array(self.bList)
setattr(EMSim, "buildSparsePhysicsEqs", buildSparsePhysicsEqs)


# In[ ]:


@njit
def allEqGen(shapeFull, nIndexFull1D, pTypeFull1D, sourceFull1D, k0):
    (xMax, yMax) = shapeFull
    nUnknowns = xMax*yMax
    rowCoeffs = []
    colCoeffs = []
    valCoeffs = []
    bCoeffs = []
    for x in range(xMax):
        for y in range(yMax):
            xy = (x, y)
            rslt = makeEzEq(xy, shapeFull, k0, nIndexFull1D, pTypeFull1D, 
                            sourceFull1D)
            (newRowArray, newColArray, newCoeffArray, newBArray) = rslt
            rowCoeffs.extend(newRowArray)
            colCoeffs.extend(newColArray)
            valCoeffs.extend(newCoeffArray)
            bCoeffs.extend(newBArray)
    return (rowCoeffs, colCoeffs, valCoeffs, bCoeffs)


# In[ ]:


@njit
def makeEzEq(xy, shapeFull, k0, nIndexFull1D, pTypeFull1D, sourceFull1D):
    """
    """
    (x, y) = xy
    delta = (1. + 0.j)
    zeroC = (0. + 0.j)
    
    # Calculate Indices
    xs = np.array([x, x-1, x+1,   x,   x])
    ys = np.array([y,   y,   y, y-1, y+1])
    inds = indFull((xs, ys), shapeFull)
    (i00, iN0, iP0, i0N, i0P) = inds
    eqNum = i00

    [ z00,  zN0,  zP0,  z0N,  z0P] = (pTypeFull1D[inds] == ZERO_CODE)
    [dz00, dzN0, dzP0, dz0N, dz0P] = (pTypeFull1D[inds] == DERZERO_CODE)
    
    s00 = sourceFull1D[i00]
    n00 = nIndexFull1D[i00]
    
    rowArray =   (eqNum, eqNum, eqNum, eqNum, eqNum)
    colArray =   (  i00,   iN0,   iP0,   i0N,   i0P)
    coeffArray = (zeroC, zeroC, zeroC, zeroC, zeroC)
    
    if z00 == 1: # is PEC
        coeffArray = (100+0j, zeroC, zeroC, zeroC, zeroC)
        bArray = (zeroC,)
    elif dz00 == 1: # is PMC
        coeffArray = (100+0j, zeroC, zeroC, zeroC, zeroC)
        bArray = (zeroC,)
    else: # is standard
        a = (4+0j) - (dzP0 + dzN0 + dz0P + dz0N)
        coeffArray = ((delta*n00*k0)**2 - a, 
                      1 - zN0|dzN0, 
                      1 - zP0|dzP0,
                      1 - z0N|dz0N,
                      1 - z0P|dz0P)
        bArray = (s00,)
    return (rowArray, colArray, coeffArray, bArray)


# ### Interface

# #### Source

# In[ ]:


def setSource(self, xy, val):
    """
    xy = (3, 3)
    xy = ([1,2,3], [10, 12, 14])
    xy = ([1,2,3], 10)
    xy = (1, [10, 12, 14])
    """
    (xs, ys) = xy
    ids = self.ind((xs,ys))
    self.sourceFull1D[ids] = val
setattr(EMSim, 'setSource', setSource)


# In[ ]:


def setSourceRect(self, xyRange, val):
    inds = self.indRect(xyRange)
    self.sourceFull1D[inds] = val
setattr(EMSim, 'setSourceRect', setSourceRect)


# In[ ]:


def setSourceLineX(self, xRange, y0, width, val):
    yRange = (round(y0-width/2), round(y0+width/2))
    inds = self.indRect((xRange, yRange))
    self.sourceFull1D[inds] = val
setattr(EMSim, 'setSourceLineX', setSourceLineX)


# In[ ]:


def setSourceLineY(self, yRange, x0, width, val):
    xRange = (round(y0-width/2), round(y0+width/2))
    inds = self.indRect((xRange, yRange))
    self.sourceFull1D[inds] = val
setattr(EMSim, 'setSourceLineY', setSourceLineY)


# In[ ]:


def zeroSources(self):
    self.sourceFull1D[:] = 0.+0.j
    # self.B*=0
setattr(EMSim, 'zeroSources', zeroSources)


# In[ ]:


# def applyExcitation(self, excitation):
#     (xs, ys, vals) = excitation
#     self.sources[:] = 0
#     for i in range(len(xs)):
#         self.sources[ys[i], xs[i]] = vals[i]
# setattr(EMSim, 'applyExcitation', applyExcitation)
    


# #### nIndex

# In[ ]:


def setNIndex(self, xy, n):
    (xs, ys) = xy
    ids = self.ind((xs,ys))
    self.nIndexFull1D[ids] = n
setattr(EMSim, 'setNIndex', setNIndex)


# In[ ]:


def setNIndexRect(self, xyRange, n):
    inds = self.indRect(xyRange)
    self.nIndexFull1D[inds] = n
setattr(EMSim, 'setNIndexRect', setNIndexRect)


# In[ ]:


def setNIndexLineX(self, xRange, y0, width, n):
    yRange = (round(y0-width/2), round(y0+width/2))
    inds = self.indRect((xRange, yRange))
    self.nIndexFull1D[inds] = n
setattr(EMSim, 'setNIndexLineX', setNIndexLineX)


# In[ ]:


def setNIndexLineY(self, yRange, x0, width, n):
    xRange = (round(y0-width/2), round(y0+width/2))
    inds = self.indRect((xRange, yRange))
    self.nIndexFull1D[inds] = n
setattr(EMSim, 'setNIndexLineY', setNIndexLineY)


# #### Boundary Conditions

# In[ ]:


def setPType(self, xy, pType):
    (xs, ys) = xy
    val = pTypeToCode(pType)
    ids = self.ind((xs,ys))
    self.pTypeFull1D[ids] = val
setattr(EMSim, 'setPType', setPType)


# In[ ]:


def setPTypeRect(self, xyRange, pType):
    ((xMin, xMax), (yMin, yMax)) = xyRange
    val = pTypeToCode(pType)
    ids = self.indRect(xyRange)
    self.pTypeFull1D[ids] = val
setattr(EMSim, 'setPTypeRect', setPTypeRect)


# In[ ]:


def setPTypeLineX(self, xRange, y0, width, pType):
    yRange = (round(y0-width/2), round(y0+width/2))
    val = pTypeToCode(pType)
    ids = self.indRect((xRange, yRange))
    self.pTypeFull1D[ids] = val
setattr(EMSim, 'setPTypeLineX', setPTypeLineX)


# In[ ]:


def setPTypeLineY(self, yRange, x0, width, pType):
    xRange = (round(x0-width/2), round(x0+width/2))
    val = pTypeToCode(pType)
    ids = self.indRect((xRange, yRange))
    self.pTypeFull1D[ids] = val
setattr(EMSim, 'setPTypeLineY', setPTypeLineY)


# ### Visualization

# In[ ]:


def add_margin(pil_img, m, color):
    width, height = pil_img.size
    new_width = width + 2*m
    new_height = height + 2*m
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (m, m))
    return result


# In[ ]:


def visualizeSim(self, zoom=3, maxNRange="auto", maxSourceRange="auto", domain="interest"):
    # PEC/PMC: 1,  Normal: 0
    if domain == "interest":
        nIndex = self.nIndex
        source = self.source
        pType = self.pType
    elif domain == "full":
        nIndex = self.nIndexFull
        source = self.sourceFull
        pType = self.pTypeFull
        
    PECIndex = (pType == ZERO_CODE).astype(np.uint8)
    PMCIndex = (pType == DERZERO_CODE).astype(np.uint8)

    if maxNRange=="auto":
        maxNR = np.max(np.abs(nIndex)) - 0.
    else:
        maxNR = maxNRange - 0.

    if maxSourceRange=="auto":
        maxSR = np.max(np.abs(source))
    else:
        maxSR = maxSourceRange

    chi = (nIndex - (1 + 1j*10**-6))
    rotScal = np.exp(1j*2*np.pi*(3/6))
    nColor3D = colorizeComplexArray(chi*rotScal, maxRad=maxNR, centerColor='black')

    PECColor3D = np.array([[(  0,  0,  255)]], dtype=np.uint8)
    PMCColor3D = np.array([[(255,  0,  0)]], dtype=np.uint8)

    choiceArray = (np.zeros_like(nIndex,dtype=np.uint8) + 
                   PECIndex + 2*PMCIndex)
    
    sourceMask = (source != 0).astype(np.uint8)
    sourceColor3D = colorizeComplexArray(source, maxRad=maxSR, centerColor='black')
    choiceArray = choiceArray*(1-sourceMask) + 3*sourceMask

    nMask = np.atleast_3d(choiceArray == 0)
    PECMask = np.atleast_3d(choiceArray == 1)
    PMCMask = np.atleast_3d(choiceArray == 2)
    sourceMask = np.atleast_3d(choiceArray == 3)
    pixArray = nMask*nColor3D + PECMask*PECColor3D + PMCMask*PMCColor3D + sourceMask*sourceColor3D
    
    image = Image.fromarray(np.swapaxes(pixArray, 0, 1)[::-1])
    (xSize, ySize) = pixArray.shape[0:2]
    imageBig = image.resize(
        (round(xSize * zoom), round(ySize * zoom)), Image.NEAREST)
    imageBigPadded = add_margin(imageBig, 10, (128,128,128))
    return imageBigPadded
setattr(EMSim, 'visualizeSim', visualizeSim)


# In[ ]:


# def visualizeSources(self, zoom=3):
#     maxRange = np.max(np.abs(self.sources))
#     pixArray = colorizeComplexArray(
#         self.sources[::-1], maxRad=maxRange, centerColor='black')
#     image = Image.fromarray(pixArray.T)
#     (ySize, xSize) = pixArray.shape[0:2]
#     imageBig = image.resize(
#         (xSize * zoom, ySize * zoom), Image.NEAREST)
#     return imageBig
# setattr(EMSim, 'visualizeSources', visualizeSources)


# In[ ]:


def visualizeField(self, zoom=3, maxRange="auto", domain="interest", func="real"):
    if maxRange=="auto":
        maxR = np.max(np.abs(self.field))
    else:
        maxR = np.abs(maxRange)
    if domain=="interest":
        field = self.field
    elif domain=="full":
        field = self.fieldFull
    else:
        raise Exception("visualizeField: 'domain' should be either 'full' or 'interest'")
    if func=="real":
        pixArray = colorizeComplexArray(
            field, maxRad=maxR, centerColor='black')
    elif func=="abs":
        pixArray = colorizeArray(field, 
                                 min_max=(0, maxR),
                                 colors=([0, 0, 0], [255, 0, 0],
                                         [255, 255, 255]),
                                 preFunc=lambda x: np.abs(x))
    else:
        raise Exception("visualizeField: 'func' should be either 'real' or 'abs'")
    image = Image.fromarray(np.swapaxes(pixArray, 0, 1)[::-1])
    (xSize, ySize) = pixArray.shape[0:2]
    imageBig = image.resize(
        (round(xSize * zoom), round(ySize * zoom)), Image.NEAREST)
    imageBigPadded = add_margin(imageBig, 10, (128,128,128))
    return imageBigPadded
setattr(EMSim, 'visualizeField', visualizeField)


# In[ ]:


def visualizeAll(self, zoom=3, domain="interest", func="real"):
    imageSim, imageField = (
        self.visualizeSim(zoom=zoom, domain=domain), 
        self.visualizeField(zoom=zoom, domain=domain, func=func))
    w1,h1 = imageSim.size
    w2,h2 = imageField.size
    hMax = max(h1,h2)
    wMax = max(w1,w2)
    new_im = Image.new('RGB', (2*wMax+6, hMax+4),(127, 127, 127))
    for i, (im,h,w) in enumerate(zip([imageSim, imageField], (h1,h2), (w1, w2))):
        new_im.paste(im, ((wMax-w)//2 + i*(wMax + 2) + 2, (hMax-h)//2 + 2))
    return new_im
setattr(EMSim, 'visualizeAll', visualizeAll)


# In[ ]:


def animateFields(self, maxRange="auto", domain="interest"):
    if domain=="interest":
        fields = self.field
        maxX, maxY = self.shape
    elif domain=="full":
        fields = self.fieldFull
        maxX, maxY = self.shapeFull
    
    fig = plt.figure()
    ax = plt.axes(xlim=(0-0.5, maxX-0.5), ylim=(0-.5, maxY-0.5))
    ax.axis('off')

    if maxRange=="auto":
        maxR = np.max(np.abs(fields))
    else:
        maxR = maxRange

    aC=np.swapaxes(fields, 0, 1)
    pixArray = colorizeComplexArray(aC, maxRad=maxR, centerColor='black')
    im = plt.imshow(pixArray,interpolation='none',)
    # initialization function: plot the background of each frame
    def init():
        pixArray = colorizeComplexArray(aC, maxRad=maxR, centerColor='black')
        
        im.set_data(pixArray)
        return im

    # animation function.  This is called sequentially
    def animate(i):
        phi = 2*pi*i/20
        pixArray = colorizeComplexArray(aC*np.exp(-1j*phi), maxRad=maxR, centerColor='black')
        im.set_data(pixArray)
        return im

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=20)
    return HTML(ani.to_jshtml())
setattr(EMSim, 'animateFields', animateFields)


# In[ ]:




