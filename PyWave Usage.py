#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', '"./Helmholtz.ipynb"')


# In[ ]:


from math import *
import numpy as np
from numpy import sin, cos, exp


# In[ ]:


J = 1j


# # Simple Example

# In the example below, we create a simulation which is 100w x 100h units in shape.  The wavelength within this is 15 units.
# We then create two sources.  The first is at x,y = 20,50 with an amplitude of 2 and a phase of -45 degrees. The second is at x,y = 80,50 with an amplitude of 1.5 and a phase of +45 degrees.

# The line below will create the simulation object.

# In[ ]:


sim = EMSim(shape=(100,100), WL0=15)


# Next we add two sources to the simulation.

# In[ ]:


sim.setSource((20,50), 2*exp(-J*2*pi*-45/360))
sim.setSource((80,50), 1.5*exp(-J*2*pi*+45/360))


# The line `sim.BuildSimBounds()` belwo creates the simulation bounds.  We must always do this **after** all of the geometry has been constructed as it will attempt to take into account structures at the edges.
# 
# You can use `sim.visualizeSimBokeh()` to create a picture of your simulation.  You can use your mouse wheel to zoom and the pan tool to explore it.  The parts within the gray dashed rectangle are the parts which are considered part of the simulation.  The parts outside were created by `BuildSimBounds()` in order to simulate absorbers and other boundary conditions.  The mouse tool-tip will tell you where your cursor is.  This can also be useful for verifying your design.
# 
# Visualizing the simulation is optional, but can be useful in order to make sure you have everything in the right spot prior to doing the relatively time consuming solve step.

# In[ ]:


sim.BuildSimBounds()
sim.visualizeSimBokeh(imageWidth=300)  # <-- Optional.


# The lines below will build the system of equations to be solved and then solve them.  They must always be run together.

# In[ ]:


sim.buildSparsePhysicsEqs()
sim.solve()


# Finally, we can visualize the simulation with the results.  In the first pair of plots, we see our simulation along with the fields, rendered as complex values.  Phase of the field is represented by the hue while the amplitude is represented by the brightness.  It will automatically scale for the maximum value, but it is sometimes useful to use the options `maxRangeField` and `maxRangeSource` and `maxRangeN` to fix the values.

# In[ ]:


sim.visualizeAllBokeh(maxRangeField=0.4)


# We can also visualize the field as an animation.

# In[ ]:


sim.animateFields()


# # Python/Numpy Construction

# In[ ]:


xSize = 100
ySize = 100
WL0 = 15
sim = EMSim(shape=(xSize,ySize), WL0=WL0)


# The EMSim object consists of four 2D numpy arrays and a bunch of methods to manipulate them.
# 
# Input Arrays:
# * `sim.source`: Complex Values. The energy sources.
# * `sim.nIndex`: Complex Values. The optical index of the medium.  Positive imaginary part indicates material absorption.
# * `sim.pType`: Integer Values.  The role of each "pixel" within the simulation.  
#  * It could be `sim.NORMAL_CODE = 0` in which the behavior there is governed by the optical index. 
#  * It could be `sim.ZERO_CODE = 1 ` in which the field is forced to be zero on this pixel.
#  * It could be `sim.DERZERO_CODE = 2` in which the derivative of the field is forced to be zero on this pixel.
#  
# Output Array:
# * `sim.field`: Complex Values.  The fields, which are populated through the use of `sim.solve()`.

# Below we can see the default values for the arrays.
# * Pixel is normal with an optical index of 1, supporting wave propagation.
# * There is no sources
# * The field is zero because we haven't solved it yet.

# In[ ]:


sim.pType[40,40]


# In[ ]:


sim.nIndex[40,40]


# In[ ]:


sim.source[40,40]


# In[ ]:


sim.field[40,40]


# These arrays can be manipulated directly using standard Numpy and Python operations.

# In[ ]:


for x in range(xSize):
    for y in range(ySize):
        underWave = y < 5*cos(2*pi*(x-50)/30) + 50
        inCircle = (x-50)**2 + (y-50)**2 < 40**2
        n = 1 + x/xSize
        if underWave and inCircle:
            sim.nIndex[x,y] = n


# In[ ]:


sim.visualizeSimBokeh(imageWidth=200)


# In[ ]:


for x in range(xSize):
    for y in range(ySize):
        inCircle1 = (x-20)**2 + (y-80)**2 < 10**2
        inCircle2 = (x-50)**2 + (y-80)**2 < 10**2
        inCircle3 = (x-80)**2 + (y-80)**2 < 10**2
        if inCircle1 or inCircle3:
            sim.pType[x,y] = sim.ZERO_CODE
        if inCircle2:
            sim.pType[x,y] = sim.DERZERO_CODE


# In[ ]:


sim.visualizeSimBokeh(imageWidth=200)


# In[ ]:


for x in range(xSize):
    amp = (1+.0001j)*exp(-(x-50)**2/30**2)
    ph = 2*pi*x/(2*WL0)
    s = amp*exp(J*ph)
    sim.source[x,5] += s


# In[ ]:


sim.visualizeSimBokeh(imageWidth=200)


# In[ ]:


sim.BuildSimBounds()
sim.buildSparsePhysicsEqs()
sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# In[ ]:


sim.animateFields()


# # Shapely Construction

# In[ ]:


import shapely as sh
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import translate, scale, rotate


# Shapely is a library which allows for creation and manipulation of geometric objects.  It is very mathematical in nature.  However, we can use it to do some pretty basic things.

# ## Shape Creation with Buffer, Union, Difference, and Intersection

# We want to work with objects that have area.  We can either begin with a `Polygon`, or we can use `pt.buffer()` on a `Point` or `LineString` to give one of these area.  From there, we can use methods like `sh.union()`, `sh.difference()`, and `sh.intersection()` to create more complicated geometries.
# 
# The full manual can be found here. (https://shapely.readthedocs.io/en/stable/manual.html)
# 

# In[ ]:


myTriangleRounded = Polygon([(10,10),(20,25),(30,10)]).buffer(5)
myTriangleRounded


# In[ ]:


triangleBiter = Point((10,10)).buffer(6)
triangleBiter


# In[ ]:


myShape = myTriangleRounded.difference(triangleBiter)
myShape


# In[ ]:


circleLeft = Point((40,40)).buffer(7)
circleRight = Point((50,40)).buffer(6)
lens = circleLeft.intersection(circleRight)
lens


# In[ ]:


myShape = myShape.union(lens)
myShape


# In[ ]:


xSize = 100
ySize = 100
WL0 = 15
sim = EMSim(shape=(xSize,ySize), WL0=WL0)


# In[ ]:


sim.setNIndexPolygon(myShape, 2)


# In[ ]:


myShape2 = myShape.buffer(-4)


# In[ ]:


sim.setPTypePolygon(myShape2, sim.ZERO_CODE)


# In[ ]:


sim.visualizeSimBokeh(imageWidth=200)


# ## Transformations: translate(), rotate(), and scale()

# We can also use methods like `translate()`, `rotate()`, and `scale()` to manipulate and copy these objects.

# In[ ]:


disk = Point(0,0).buffer(5.1)
disk


# In[ ]:


np.linspace(20,80,num=4, endpoint=True)


# In[ ]:


p = Polygon()
for x in np.linspace(20,80,num=4, endpoint=True):
    for y in np.linspace(20,80,num=4, endpoint=True):
        newDisk = translate(disk, x, y)
        p = p.union(newDisk)


# In[ ]:


p


# In[ ]:


xSize = 100
ySize = 100
WL0 = 15
sim = EMSim(shape=(xSize,ySize), WL0=WL0)
sim.setNIndexPolygon(p, 2)


# In[ ]:


sim.visualizeSimBokeh(imageWidth=200)


# ## Custom Objects - Waveguides

# I've also create a function to build my own waveguide object.  You could do similar to build lenses, parabolic dishes, and similar.

# In[ ]:


from CurveGenerator import buildCurvedWaveguide


# In[ ]:


wg1 = buildCurvedWaveguide([(50,10), (80,30), (55,60), (75,90), (10, 90)], rad=10, width=3)
wg1


# In[ ]:


xSize = 100
ySize = 100
WL0 = 15
sim = EMSim(shape=(xSize,ySize), WL0=WL0)
sim.setNIndexPolygon(wg1, 2)


# In[ ]:


sim.visualizeSimBokeh(imageWidth=200)


# In[ ]:




