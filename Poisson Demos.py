#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import shapely as sh


# In[ ]:


get_ipython().run_line_magic('run', '"./Poisson.ipynb"')


# # Demo 1 - Exploration of Important Methods

# In[ ]:


sim = ESim(shape=(100,100), bounds='per')


# Note that several options exist for `bounds`, including
#  - `per`: Periodic
#  - `abs`: Large margin followed by a $V = 0$ boundary.
#  - `pec`: PEC boundary with floating potential just outside simulation domain
#  - `zero`: $V=0$ boundary just outside simulation domain

# ### Setting Voltage

# In[ ]:


for y in range(5, 95):
    sim.setVoltage((10, y),  1)


# In[ ]:


ball1 = sh.geometry.Point(4,4).buffer(3)


# In[ ]:


sim.setVoltagePolygon(ball1, -1)


# ### Setting PEC

# In[ ]:


sim.setPEC((20,80))


# In[ ]:


bar = sh.geometry.LineString([(30,30),(40,40)]).buffer(1)


# In[ ]:


sim.setPECPolygon(bar)


# ### Setting Sources

# In[ ]:


sim.setSource((60,78), -5)


# In[ ]:


ball2 = sh.geometry.Point(80,50).buffer(10)


# In[ ]:


sim.setSourcePolygon(ball2, 0.01)


# ### Setting Permittivity

# In[ ]:


ball3 = sh.geometry.Point(61,64).buffer(10)


# In[ ]:


sim.setEpsPolygon(ball3, 2)


# In[ ]:


sim.setEps((61,64), 4)


# ### Executing the Simulation

# In[ ]:


sim.BuildSimBounds()


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.visualizeAllBokeh(maxRangeEps=2, maxRangeQ=2, maxRangeV=5)


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# In[ ]:


sim.

