#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from shapely.geometry import LineString, MultiPolygon, Polygon, MultiLineString, MultiPoint, Point
import itertools


# In[ ]:


import numpy as np


# In[ ]:


def findIntersection(line1, line2):
    """
    Given line1 = ((x1,y1), (x2,y2)) and line2 = ((x3, y3), (x4, y4)), both treated
    as infinite, this will return the point that is at the intersection.
    """
    (pt1, pt2) = line1
    (pt3, pt4) = line2
    (x1, y1) = pt1
    (x2, y2) = pt2
    (x3, y3) = pt3
    (x4, y4) = pt4
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return (px, py)


# In[ ]:


def TangentialCurve(pt1pt2pt3, r, maxCurve=5):
    """
    Given three points ((x1,y1), (x2,y2), (x3,y3)), representing two 
    line segments, this function will return the curve that is tangent
    to these to segments.
    
    If the segments are colinear, this will simply return pt2.
    """
    pt0, pt1, pt2 = np.array(pt1pt2pt3)
    v1 = pt1-pt0
    v2 = pt2-pt1
    v1Norm = np.linalg.norm(v1)
    v2Norm = np.linalg.norm(v2)
    v1N = v1/v1Norm
    v2N = v2/v2Norm
    s = np.sign(np.cross(v1N,v2N))
    if s == 0:
        pt1x, pt1y = pt1
        return [(pt1x, pt1y)]
    v1Orth = np.array([[0,-1],[1,0]])@v1N
    v2Orth = np.array([[0,-1],[1,0]])@v2N
    offset1 = (pt0+r*s*v1Orth, pt1+r*s*v1Orth)
    offset2 = (pt1+r*s*v2Orth, pt2+r*s*v2Orth)
    ptC = findIntersection(offset1, offset2)
    ptS = ptC - r*s*v1Orth
    ptE = ptC - r*s*v2Orth
    deltaAngle = s*np.arccos(np.dot(v1N,v2N))
    tX, tY = -s*v1Orth
    angleS = np.arctan2(tY, tX)
    angleE = angleS + deltaAngle
    deg = 2*np.pi/360
    n = int(round(abs(deltaAngle)/(maxCurve*deg)))
    angles = np.linspace(angleS, angleE, num=n)
    ptCx, ptCy = ptC
    curvePoints = [(np.cos(a)*r + ptCx, np.sin(a)*r + ptCy) for a in angles]
    return curvePoints


# In[ ]:


def SmoothCurve(guidePoints, r):
    """
    Given a set of guide points [(x1, y1), (x2, y2), ..., (xn, yn)], and a radius
    this will produce smoothed version of the curve where all corners are filleted
    to the given radius.
    """
    listOfLists = [[guidePoints[0]]] + [TangentialCurve(guidePoints[i:i+3], r) for i in range(0,len(guidePoints)-2)] + [[guidePoints[-1]]]
    return list(itertools.chain.from_iterable(listOfLists))


# In[ ]:


def buildCurvedWaveguide(guidePoints, rad, width):
    """
    Creates a shapely polygon representing a waveguide.  Given set of 
    guide points [(x1, y1), (x2, y2), ..., (xn, yn)], this function will
    smooth all inside corners and buffer the object to a given width.
    """
    return LineString(SmoothCurve(guidePoints, rad)).buffer(width/2, cap_style=2)


# In[ ]:


guidePoints = [(0,0), (30,0), (20,10), (30,10)]
rad = 3
width = 2
buildCurvedWaveguide(guidePoints, rad, width)


# In[ ]:


# def buildCurvedWaveguide(guidePoints, rad, width, delta=0.01):
#     """
#     Creates a rounded waveguide as a Shapely Polyon object from a list of guide points.
    
#     guidepoints = list of tuples [(0,0), (1,0), (2,1), (3,1)]
#     rad = radius of curvature on the waveguide.
#     width = width of the waveguide.
#     delta = small value used to deal with errors created by discretization of curves and 
#     floating point errors.  Larger values (ie 0.01) will make it more tolerant, but might have other side effects.
#     """
#     guideLine = LineString(guidePoints)
#     curvePolys = guideLine.buffer(rad).buffer(-rad-delta)
#     if isinstance(curvePolys, Polygon):
#         curvePolys = MultiPolygon([curvePolys])
#     straights = guideLine.difference(curvePolys.buffer(2*delta))
#     polyPerims = []
#     for pol in curvePolys:
#         boundary = pol.boundary
#         if boundary.type == 'MultiLineString':
#             for line in boundary:
#                 polyPerims.append(line)
#         else:
#             polyPerims.append(boundary)
#     mLineStr = MultiLineString(polyPerims)
#     curves = mLineStr.difference(guideLine.buffer(4*delta))
#     curveSketch = curves.union(straights)
#     waveguidePoly = curveSketch.buffer(rad)
#     waveguidePoly = waveguidePoly.buffer(-rad+width/2)
#     return waveguidePoly


# In[ ]:


# buildCurvedWaveguide([(0,0), (1,0), (2,1), (3,1)], rad=2, width = 0.3, delta=0.01)


# In[ ]:




