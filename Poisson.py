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


# In[ ]:


import bokeh
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.io import export_svgs
from bokeh.models import Arrow, NormalHead, OpenHead, VeeHead
from bokeh.layouts import gridplot, row

print("Bokeh Version:", bokeh.__version__)

output_notebook()
bokeh.io.curdoc().theme = 'dark_minimal'


# In[ ]:


from shapely.geometry import LineString, MultiPolygon, Polygon, MultiLineString, MultiPoint, Point


# Custom Imports

# In[ ]:


from colorize import *


# In[ ]:


pi = np.pi
j = 1j


# ## Notes

# Notes on coodinate systems:
# 
# We use `(x, y)` for both display and specification.  Origin is at 0,0 in the lower left hand corner.  For a given field `F[10,0]` would denote the point at `(x,y) = (10,0)`.  This allows intuitive specification, but runs counter to how one would typically write a matrix on a sheet of paper, with element `(i,j) = (0,0)` being in upper left hand corner, `i` denoting row and `j` denoting column.

# ## Function Library

# ### Physics

# In[ ]:


NORMAL_CODE = 0
PEC_CODE = 1
VOLT_CODE = 2


# In[ ]:


PEC_EPS = 1 - 1000j


# In[ ]:


def pTypeToCode(str):
    if str == "normal":
        return NORMAL_CODE
    elif str == "pec":
        return PEC_CODE
    elif str == 'volt':
        return VOLT_CODE


# In[ ]:


class ESim:
    pass


# In[ ]:


def __init__(self, shape=(10, 10), bounds='zero', margin='auto'):
    """ ESim is a simple 2D Finite Difference Frequency Domain Simulator 
    (FDFD) for quasi electro statics.
    
    It was primarily designed as a teaching tool for freshmen.
    
    Args:
        shape (:tuple:`int`): The width (x) and height (y) in pixels of the 
            simulation sans any boundary that is added via `margin'.
        bounds: One of the following:
            'abs' : Pads domain with 'margin' and then adds v=0 boundary.
            'per' : Periodic boundary.  Must match Left-Right or Bottom-Top.
            'pec' : Perfect electric conductor at some floating potential.
            'zero': Zero field boundary
        margin (:`int`): The size of an added boundary layer on all sides for use
                         with 'abs' boundary.

    Attributes:
        eps (:np.array:(`complex`)):  The permittivity field of the simulation.
        source (:np.array:(`complex`)):  The sources of the simulation within
        field (:np.array:(`complex`)):  The voltage field result of the simulation
        """
    
    self._NORMAL_CODE = 0
    self._PEC_CODE = 1
    self._VOLT_CODE = 2
    
    if margin=='auto':
        self._margin = max(shape)//4
    else:
        self._margin = margin
    margin = self._margin
    self._bounds = bounds
    self._shape = shape

    m = BoundaryCodeToPadding(bounds, margin)
    xSize, ySize = self._shape
    
    self._shapeFull = (m + xSize + m, m + ySize + m)
    shapeFull = self._shapeFull

    # The three vars below are all views of the same data block
    self._fieldFull = np.zeros(shapeFull, np.complex)
    self.field = self._fieldFull[m:(m+xSize), m:(m+ySize)]
    self._fieldFull1D = self._fieldFull.ravel()

    # The three vars below are all views of the same data block
    self._sourceFull = np.zeros(shapeFull, np.complex)
    self.source = self._sourceFull[m:(m+xSize), m:(m+ySize)]
    self._sourceFull1D = self._sourceFull.ravel()
    
    # The three vars below are all views of the same data block.
    # Normal: 0, field[x,y] == 0 : 1, d field[x,y]/du == 0 : 2
    self._pTypeFull = np.full(shapeFull, 0, np.uint8)  
    self._pType = self._pTypeFull[m:(m+xSize), m:(m+ySize)]
    self._pTypeFull1D = self._pTypeFull.ravel()

    # The three vars below are all views of the same data block
    self._epsFull = np.full(shapeFull, 1., np.complex)  
    self.eps = self._epsFull[m:(m+xSize), m:(m+ySize)]
    self._epsFull1D = self._epsFull.ravel()

    
setattr(ESim, "__init__", __init__)


# In[ ]:


def BoundaryCodeToPadding(code, margin):
    if code == 'abs':
        return margin
    elif code == 'per':
        return 0
    elif code == 'pec':
        return 1
    elif code == 'zero':
        return 1


# In[ ]:


def BuildSimBounds(self):
    """
    Builds the margins of the simulation.
    
    The margins are built by padding the `eps` to `epsFull`.
    Whatever geometry is one the edge will copied into the margin.
    
    It then adds an additional layer to terminate the simulation in 
    some way based on the bounds specified at instantiation.
    
    This should be called after all the changes to `eps` have been completed.
    """
    pecEps = 1-1000j
    margin = self._margin
    m = BoundaryCodeToPadding(self._bounds, self._margin)
    (xMax, yMax) = self._shapeFull
    # Pad nIndex.  Note that we first create the array, then copy the data into
    # the array to preserve the shared data between nIndex and nIndexFull.  Using
    # the 'edge' keyword propagates waveguides and similar into the PML region to
    # prevent weird reflections.
    padded = np.pad(self.eps, ((m,m), (m,m)), 'edge')
    self._epsFull[:] = padded[:]
    def s(x):
        return((c) * (x)**a)
    padded = np.pad(self._pType, ((m,m), (m,m)), 'edge')
    self._pTypeFull[:] = padded[:]
    bounds = self._bounds
    if bounds=='abs':
        ring0 = indRing(0, self._shapeFull)
        self._fieldFull1D[ring0] = 0j
        self._pTypeFull1D[ring0] = VOLT_CODE
    if bounds=='per':
        pass
    if bounds=='pec':
        ring0 = indRing(0, self._shapeFull)
        self._epsFull1D[ring0] = PEC_EPS
    if bounds=='zero':
        ring0 = indRing(0, self._shapeFull)
        self._fieldFull1D[ring0] = 0j
        self._pTypeFull1D[ring0] = VOLT_CODE

setattr(ESim, "BuildSimBounds", BuildSimBounds)


# In[ ]:


def solve(self):
    """
    Solves the M.X == B system for X, ie the fieldFull1D.
    """
    X, code = lgmres(csc_matrix(self._M), self._B, atol=1e-6)
    self._fieldFull1D[:] = X
setattr(ESim, "solve", solve)


# ### Indexing Functions

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
    m = BoundaryCodeToPadding(self._bounds, self._margin)
    return indFull((x+m, y+m), self._shapeFull)
setattr(ESim, "ind", ind)


# In[ ]:


def indRing(i, shapeFull):
    """
    Gives the full1D indices for an outer ring of the simulation.  For
    instance `indRing(0, shapeFull)` would be the indices at the outside
    edge of the simulation and `indRing(1, shapeFull)` would be the indices
    one pixel inside of those.
    
    This is useful handling the bounds of the simulation.
    """
    xMax, yMax = shapeFull
    l = []
    for x in range(i, xMax-i):
        cT = (x, i)
        cB = (x, yMax-i-1)
        l.append(indFull(cT, shapeFull))
        l.append(indFull(cB, shapeFull))
    for y in range(i+1, yMax-i-1):
        cL = (i, y)
        cR = (xMax-i-1, y)
        l.append(indFull(cL, shapeFull))
        l.append(indFull(cR, shapeFull))
    return np.array(l)


# ### Optimized Equation Generator

# The equation that is being solved is:
# $$-\nabla \cdot (\epsilon \nabla V) = \rho$$
# where we employ a discretization such that the voltage (electric potential) is a point at the center of each pixel with permittivity $\epsilon$.  We use this equation to relate nearest neighbors.  Let us consider the pixel in question to be at point $(0,0)$ with voltage $V_{(0,0)}$.
# 
# Let us begin by evaluating $D_x$ within pixel $(0,0)$ on the right side of the center point pointing toward $(1,0)$.  In order to lend some insight into this, let us consider $V_{(0,0)}$ and $V_{(1,0)}$ as the classic 1D problem of the parallel plate capacitor.  For the sake of simplicity of notation, let us denote $(0,0) \rightarrow a$ and $(1,0) \rightarrow b$.  Let us call the distance between the plates to be $d$.
# 
# It follows that we can perform a line integral from one plate to the other to obtain a voltage drop.
# $$(d/2)E_a + (d/2)E_b = -(V_a - V_b)$$
# We also know that normal $D$ is continuous so
# $$D_a = \epsilon_a E_a = \epsilon_b E_b$$
# 
# After performing some algebra, we can conclude that
# $$D_{a,\mathrm{right}} = \epsilon_a E_{a,\mathrm{right}} = \frac{2 \epsilon_a \epsilon_b (V_b - V_a)}{d(\epsilon_b + \epsilon_a)}$$
# 
# The concept behind $-\nabla \cdot (\epsilon \nabla V) = \rho$ is that the divergence of $D$ is equal to the charge contained therein.  Put another way:
# $$\nabla \cdot D_a = \rho_a \\
# D_{a,\mathrm{right}} + D_{a,\mathrm{up}} + D_{a,\mathrm{left}} + D_{a,\mathrm{down}} = \rho_a
# $$
# This equation can be evaluated pixel-by-pixel based on a given charge density field $\mathbf{\rho}$ and permittivity field $\mathbf{\epsilon}$ to render a linear system of equations for an unknown voltage field $\mathbf{V}$.  For fields that are $n\times m$, this will yield $n\times m$ equations with $n\times m$ unknowns.
# 
# Note that within this work we leave open the posibility that $\epsilon$ is complex and that the equation being solved is not for electrostatics, but rather electro-quasistatics.  Within the $e^{j \omega t}$ convetion, the $\epsilon = \epsilon_r - j\frac{\sigma}{\omega}$.  A good conductor could be described as $\epsilon_\mathrm{PEC} = 1 - 1000j$.

# The following functions have been organized so that they can be optimized using Numba.  Only the top level function makes use of OOP and all of the inner loop functions are purely numeric and use only basic Numpy and Python objects.

# In[ ]:


def buildSparsePhysicsEqs(self):
    """
    Builds M and B of the matrix system M.X == B, where 
        X is the as of yet unknown fieldsFull1D
        B is essentially the sourcesFull1D
        and M is the relationship capturing the physics of the Helmholtz
        equations including the pixType optical index.
    """
    n = self._fieldFull.size
    
    rslt = allEqGen(self._shapeFull, self._epsFull1D, self._pTypeFull1D, 
                    self._sourceFull1D, self._fieldFull1D)
    rList, cList, vList, bList = rslt
    coo = sp.coo_matrix((vList, (rList, cList)), (n, n))
    self._M = coo
    self._B = np.array(bList)
setattr(ESim, "buildSparsePhysicsEqs", buildSparsePhysicsEqs)


# In[ ]:


@njit
def allEqGen(shapeFull, epsFull1D, pTypeFull1D, sourceFull1D, fieldFull1D):
    """
    """
    (xMax, yMax) = shapeFull
    nUnknowns = xMax*yMax
    rowCoeffs = []
    colCoeffs = []
    valCoeffs = []
    bCoeffs = []
    for x in range(xMax):
        for y in range(yMax):
            xy = (x, y)
            rslt = makeEq(xy, shapeFull, epsFull1D, pTypeFull1D, 
                            sourceFull1D, fieldFull1D)
            (newRowArray, newColArray, newCoeffArray, newBArray) = rslt
            rowCoeffs.extend(newRowArray)
            colCoeffs.extend(newColArray)
            valCoeffs.extend(newCoeffArray)
            bCoeffs.extend(newBArray)
    return (rowCoeffs, colCoeffs, valCoeffs, bCoeffs)


# In[ ]:


@njit
def makeEq(xy, shapeFull, epsFull1D, pTypeFull1D, sourceFull1D, fieldFull1D):
    """
    """
    (x, y) = xy
    d = (1. + 0.j)
    zeroC = (0. + 0.j)
    
    # Calculate Indices
    xs = np.array([x, x-1, x+1,   x,   x])
    ys = np.array([y,   y,   y, y-1, y+1])
    inds = indFull((xs, ys), shapeFull)
    (i00, iN0, iP0, i0N, i0P) = inds
    eqNum = i00

    [ p00,  pN0,  pP0,  p0N,  p0P] = pTypeFull1D[inds]
    
    s00 = sourceFull1D[i00]
    f00 = fieldFull1D[i00]
    [ e00,  eN0,  eP0,  e0N,  e0P] = epsFull1D[inds]
    
    rowArray =   (eqNum, eqNum, eqNum, eqNum, eqNum)
    colArray =   (  i00,   iN0,   iP0,   i0N,   i0P)
    coeffArray = (zeroC, zeroC, zeroC, zeroC, zeroC)
    
    if p00 == NORMAL_CODE: # is PEC
        cN0 = 2*e00*eN0/(d*(e00+eN0))
        cP0 = 2*e00*eP0/(d*(e00+eP0))
        c0N = 2*e00*e0N/(d*(e00+e0N))
        c0P = 2*e00*e0P/(d*(e00+e0P))
        
        coeffArray = (-cN0 - cP0 - c0N - c0P, 
                      cN0,
                      cP0,
                      c0N,
                      c0P)
        bArray = (-s00,)
    elif p00 == VOLT_CODE: # is Voltage
        coeffArray = (1, 0, 0, 0, 0)
        bArray = (f00,)
    elif p00 == PEC_CODE: # is PEC
        coeffArray = (100+0j, zeroC, zeroC, zeroC, zeroC)
        bArray = (zeroC,)
    else:
        coeffArray = (1, 0, 0, 0, 0)
        bArray = (99,)

    return (rowArray, colArray, coeffArray, bArray)


# ### Interface

# #### Source

# In[ ]:


def setSource(self, xy, q):
    """
    Places charge `q` at location xy.  
    
    Note that `q` could be complex denoting phase.
    
    `xy` could be of the forms:
    xy = (3, 3)
    xy = ([1,2,3], [10, 12, 14])
    xy = ([1,2,3], 10)
    xy = (1, [10, 12, 14])
    """
    (xs, ys) = xy
    ids = self.ind((xs,ys))
    self._sourceFull1D[ids] = q
setattr(ESim, 'setSource', setSource)


# In[ ]:


def setSourcePolygon(self, mPoly, q):
    """
    Places charge `q` on all pixels within polygon(s).
    
    mPoly: a Shapely Polygon or MultiPolygon

    Note that `q` could be complex denoting phase.
    
    For instance
    ```
    poly = Shapely.geometry.Point(50,50).buffer(5)
    sim.setSourcePolygon(poly, -2)
    ```
    will place a ball with radius 5 at (x,y) = (50,50)
    wherein each pixel contains charge -2.
    """
    if isinstance(mPoly, Polygon):
        mPoly = MultiPolygon([mPoly])
    (nx, ny) = self._shape
    for x in range(nx):
        for y in range(ny):
            pt1 = Point(x,y)
            if mPoly.intersects(pt1):
                self.source[x,y] = q    
setattr(ESim, 'setSourcePolygon', setSourcePolygon)


# #### Epsilon

# In[ ]:


def setEps(self, xy, eps):
    """
    Sets the permittivity of a pixel at location xy = (x, y) to `eps`.
        
    Note that `eps` could be complex denoting phase conductivity.
    
    `xy` could be of the forms:
    xy = (3, 3)
    xy = ([1,2,3], [10, 12, 14])
    xy = ([1,2,3], 10)
    xy = (1, [10, 12, 14])
    """
    (xs, ys) = xy
    ids = self.ind((xs,ys))
    self._epsFull1D[ids] = eps
setattr(ESim, 'setEps', setEps)


# In[ ]:


def setEpsPolygon(self, mPoly, eps):
    """
    Sets the permittivity of all pixels within mPoly to `eps`.
    
    mPoly: a Shapely Polygon or MultiPolygon

    Note that `q` could be complex denoting phase.
    
    For instance
    ```
    poly = Shapely.geometry.Point(50,50).buffer(5)
    sim.setEpsPolygon(poly, 2.0)
    ```
    will place a ball with radius 5 at (x,y) = (50,50)
    wherein each pixel has permittivity of 2.0.
    """
    if isinstance(mPoly, Polygon):
        mPoly = MultiPolygon([mPoly])
    (nx, ny) = self._shape
    for x in range(nx):
        for y in range(ny):
            pt1 = Point(x,y)
            if mPoly.intersects(pt1):
                self.eps[x,y] = eps    
setattr(ESim, 'setEpsPolygon', setEpsPolygon)


# #### PEC

# In[ ]:


def setPEC(self, xy):
    """
    Sets the permittivity of a pixel at location xy = (x, y) to that of a very good conductor.
    
    `xy` could be of the forms:
    xy = (3, 3)
    xy = ([1,2,3], [10, 12, 14])
    xy = ([1,2,3], 10)
    xy = (1, [10, 12, 14])
    """
    (xs, ys) = xy
    ids = self.ind((xs,ys))
    self._epsFull1D[ids] = PEC_EPS
setattr(ESim, 'setPEC', setPEC)


# In[ ]:


def setPECPolygon(self, mPoly):
    """
    Sets the permittivity of all pixels within mPoly to `eps`.
    
    mPoly: a Shapely Polygon or MultiPolygon

    Note that `q` could be complex denoting phase.
    
    For instance
    ```
    poly = Shapely.geometry.Point(50,50).buffer(5)
    sim.setEpsPolygon(poly, 2.0)
    ```
    will place a ball with radius 5 at (x,y) = (50,50)
    wherein each pixel has permittivity of 2.0.
    """
    if isinstance(mPoly, Polygon):
        mPoly = MultiPolygon([mPoly])
    (nx, ny) = self._shape
    for x in range(nx):
        for y in range(ny):
            pt1 = Point(x,y)
            if mPoly.intersects(pt1):
                self.eps[x,y] = PEC_EPS    
setattr(ESim, 'setPECPolygon', setPECPolygon)


# #### Voltage

# In[ ]:


def setVoltage(self, xy, v):
    """
    Fixes the electric potential (voltage) of a pixel at location xy = (x, y) to `v`.
    
    Note that `v` could be complex, denoting phase.
    
    `xy` could be of the forms:
    xy = (3, 3)
    xy = ([1,2,3], [10, 12, 14])
    xy = ([1,2,3], 10)
    xy = (1, [10, 12, 14])
    """
    (xs, ys) = xy
    ids = self.ind((xs,ys))
    self._fieldFull1D[ids] = v
    self._pTypeFull1D[ids] = VOLT_CODE
setattr(ESim, 'setVoltage', setVoltage)


# In[ ]:


def setVoltagePolygon(self, mPoly, v):
    """
    Fixes the electric potential (voltage) of all pixels within mPoly to `v`.
    
    mPoly: a Shapely Polygon or MultiPolygon

    Note that `v` could be complex denoting phase.
    
    For instance
    ```
    poly = Shapely.geometry.Point(50,50).buffer(5)
    sim.setVoltagePolygon(poly, 5)
    ```
    will place a ball with radius 5 at (x,y) = (50,50)
    wherein each pixel will be fixed to 5V.
    """
    if isinstance(mPoly, Polygon):
        mPoly = MultiPolygon([mPoly])
    (nx, ny) = self._shape
    for x in range(nx):
        for y in range(ny):
            pt1 = Point(x,y)
            if mPoly.intersects(pt1):
                self.field[x,y] = v
                self._pType[x,y] = VOLT_CODE
setattr(ESim, 'setVoltagePolygon', setVoltagePolygon)


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


def visualizeFieldBokeh(self, imageWidth=800, maxRangeV="auto", func="real", purpose="display"):
    if maxRangeV=="auto":
        maxR = np.max(np.abs(self.field))
        if maxR == 0:
            maxR = 1
    else:
        maxR = np.abs(maxRangeV)

    m = BoundaryCodeToPadding(self._bounds, self._margin)
    
    field = self._fieldFull
    fieldT = np.swapaxes(field, 0, 1)[::-1]

    if func=="real":
        pixArray = colorizeComplexArray(
            fieldT+0.00001j, maxRad=maxR, centerColor='black')
    elif func=="abs":
        pixArray = colorizeArray(fieldT, 
                                 min_max=(0, maxR),
                                 colors=([0, 0, 0], [255, 0, 0],
                                         [255, 255, 255]),
                                 preFunc=lambda x: np.abs(x))
    else:
        raise Exception("visualizeField: 'func' should be either 'real' or 'abs'")
        
    alphaArray = np.full_like(pixArray[:,:,0], fill_value=255)
    imageDataWithAlpha = np.concatenate([pixArray, np.atleast_3d(alphaArray)], axis=2)[::-1]
    
    dw, dh = self._shape
    dwF, dhF = self._shapeFull
    pw = int(round(imageWidth + 57))
    ph = int(round(pw*dh/dw + 31))
    TOOLTIPS = [("(x,y)", "($x{1.}, $y{1.})")]
    p = figure(x_range=(-0.5, dw-0.5), y_range=(-.5, dh-0.5), plot_width=pw, plot_height=ph, tools="pan,wheel_zoom,reset,save", tooltips=TOOLTIPS)
    p.image_rgba(image=[imageDataWithAlpha], x=-m-0.5, y=-m-0.5, dw=dwF, dh=dhF)
    p.line(x=[0-0.5,dw-0.5,dw-0.5,-0.5,-0.5], y=[-0.5,-0.5,dh-0.5,dh-0.5,-0.5], color="red", line_width=1)
    if purpose == 'display':
        show(p)
    else:
        return p
setattr(ESim, 'visualizeFieldBokeh', visualizeFieldBokeh)


# In[ ]:


def visualizeSimBokeh(self, imageWidth=800, maxRangeEps="auto", maxRangeQ="auto", purpose="display"):
    # PEC/PMC: 1,  Normal: 0
    eps = np.swapaxes(self._epsFull, 0, 1)[::-1]
    source = np.swapaxes(self._sourceFull, 0, 1)[::-1]
    pType = np.swapaxes(self._pTypeFull, 0, 1)[::-1]
    
    m = BoundaryCodeToPadding(self._bounds, self._margin)
        
    PECIndex = (pType == PEC_CODE).astype(np.uint8)

    if maxRangeEps=="auto":
        maxNR = np.max(np.abs(eps)*(~(eps==PEC_EPS))) - 0.
        # maxNR = np.max(np.abs(eps)) - 0.
    else:
        maxNR = maxRangeEps - 0.

    if maxRangeQ=="auto":
        maxSR = np.max(np.abs(source))
        if maxSR == 0:
            maxSR = 1
    else:
        maxSR = maxRangeQ

    chi = (eps - (1 + 1j*10**-6))
    rotScal = np.exp(1j*2*np.pi*(3/6))
    epsColor3D = colorizeComplexArray(chi*rotScal, maxRad=maxNR, centerColor='black')

    PECColor3D = np.array([[(  0,  0,  255)]], dtype=np.uint8)

    choiceArray = (np.zeros_like(eps,dtype=np.uint8) + PECIndex + 2*0 + 3*0)
    
    sourceMask = (source != 0).astype(np.uint8)
    sourceColor3D = colorizeComplexArray(source, maxRad=maxSR, centerColor='black')
    choiceArray = choiceArray*(1-sourceMask) + 3*sourceMask

    epsMask = np.atleast_3d(choiceArray == 0)
    PECMask = np.atleast_3d(choiceArray == 1)
    sourceMask = np.atleast_3d(choiceArray == 3)
    pixArray = epsMask*epsColor3D + PECMask*PECColor3D + sourceMask*sourceColor3D
    
    alphaArray = np.full_like(pixArray[:,:,0], fill_value=255)
    imageDataWithAlpha = np.concatenate([pixArray, np.atleast_3d(alphaArray)], axis=2)[::-1]
    
    dw, dh = self._shape
    dwF, dhF = self._shapeFull
    pw = int(round(imageWidth + 57))
    ph = int(round(pw*dh/dw - 25))
    TOOLTIPS = [("(x,y)", "($x{1.}, $y{1.})")]
    p = figure(x_range=(-0.5, dw-0.5), y_range=(-.5, dh-0.5), plot_width=pw, plot_height=ph, tools="pan,wheel_zoom,reset,save", tooltips=TOOLTIPS)
    p.image_rgba(image=[imageDataWithAlpha], x=-m-0.5, y=-m-0.5, dw=dwF, dh=dhF)
    p.line(x=[0-0.5,dw-0.5,dw-0.5,-0.5,-0.5], y=[-0.5,-0.5,dh-0.5,dh-0.5,-0.5], color="gray", line_width=1, line_dash='dotted')
    if purpose == 'display':
        show(p)
    else:
        return p
setattr(ESim, 'visualizeSimBokeh', visualizeSimBokeh)


# In[ ]:


def visualizeAllBokeh(self, func="real", maxRangeV='auto', maxRangeQ='auto', maxRangeEps='auto'):
    p1 = self.visualizeSimBokeh(purpose='value', maxRangeQ=maxRangeQ, maxRangeEps=maxRangeEps)
    p2 = self.visualizeFieldBokeh(func=func, purpose='value', maxRangeV=maxRangeV)
    p2.x_range = p1.x_range
    p2.y_range = p1.y_range
    dw, dh = self._shape
    pAll = gridplot([[p1,p2]], plot_width=600 + 57, plot_height=int((dh/dw)*600+57))
    show(pAll)
setattr(ESim, 'visualizeAllBokeh', visualizeAllBokeh)


# In[ ]:


def animateFields(self, maxRangeV="auto", domain="interest"):
    if domain=="interest":
        fields = self._field
        maxX, maxY = self._shape
    elif domain=="full":
        fields = self._fieldFull
        maxX, maxY = self._shapeFull
    
    fig = plt.figure()
    ax = plt.axes(xlim=(0-0.5, maxX-0.5), ylim=(0-0.5, maxY-0.5))
    ax.axis('off')

    if maxRangeV=="auto":
        maxR = np.max(np.abs(fields))
    else:
        maxR = maxRangeV

    aC=np.swapaxes(fields, 0, 1)
    pixArray = colorizeComplexArray(aC, maxRad=maxR, centerColor='black')
    im = plt.imshow(pixArray,interpolation='none',)
    # initialization function: plot the background of each frame
    def init():
        pixArray = colorizeComplexArray(np.real(aC)+0.001j, maxRad=maxR, centerColor='black')
        
        im.set_data(pixArray)
        return im

    # animation function.  This is called sequentially
    def animate(i):
        phi = 2*pi*i/20
        pixArray = colorizeComplexArray(np.real(np.exp(-1j*phi)*aC)+0.001j, maxRad=maxR, centerColor='black')
        im.set_data(pixArray)
        return im

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=20)
    return HTML(ani.to_jshtml())
setattr(ESim, 'animateFields', animateFields)


# In[ ]:




