#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Helmholtz import EMSim


# In[ ]:


from CurveGenerator import buildCurvedWaveguide


# In[ ]:


get_ipython().run_line_magic('run', '"./Helmholtz.ipynb"')


# # Basic

# ## Abs All Sides (Default)

# In[ ]:


sim = EMSim(shape=(100,100), WL0=20)


# In[ ]:


sim.setSource((20,50), 1)


# In[ ]:


sim.BuildSimBounds()
sim.buildSparsePhysicsEqs()
sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# ## Periodic-X, Abs Top and Bottom

# In[ ]:


sim = EMSim(shape=(100,100), WL0=20, boundsLRBT=('per', 'per', 'abs', 'abs'))


# In[ ]:


sim.setSource((20,50), 1)


# In[ ]:


sim.BuildSimBounds()
sim.buildSparsePhysicsEqs()
sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# ## Periodic-Y, Abs Left and Right

# In[ ]:


sim = EMSim(shape=(100,100), WL0=20, boundsLRBT=('abs', 'abs', 'per', 'per'))


# In[ ]:


sim.setSource((20,50), 1)


# In[ ]:


sim.BuildSimBounds()
sim.buildSparsePhysicsEqs()
sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# ## Periodic-Y, Zero Left and derZero Right

# In[ ]:


sim = EMSim(shape=(100,100), WL0=20, boundsLRBT=('abs', 'abs', 'zero', 'derZero'))


# In[ ]:


sim.setSource((20,50), 1)


# In[ ]:


sim.BuildSimBounds()
sim.buildSparsePhysicsEqs()
sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# ## Periodic-Y, Zero Left and derZero Right

# In[ ]:


sim = EMSim(shape=(100,100), WL0=20, boundsLRBT=('zero', 'derZero', 'abs', 'abs'))


# In[ ]:


sim.setSource((20,50), 1)


# In[ ]:


sim.BuildSimBounds()
sim.buildSparsePhysicsEqs()
sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# ## Waveguide

# In[ ]:


sim = EMSim(shape=(300,100), WL0=20)


# In[ ]:


sim.setSource((20,50), 1)


# In[ ]:


sim.BuildSimBounds()


# In[ ]:


from CurveGenerator import buildCurvedWaveguide


# In[ ]:


wg1 = buildCurvedWaveguide(guidePoints=[(20,50), (50,50), (100,80), (175,80), (225,50), (265,50)], rad=50, width=5)


# In[ ]:


wg1


# In[ ]:


sim.setNIndexMultiPolygon(wg1, 2)


# In[ ]:


sim.visualizeSimBokeh()


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh(func='real')


# In[ ]:


sim.animateFields()


# In[ ]:




