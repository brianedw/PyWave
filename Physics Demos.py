#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', '"./Helmholtz.ipynb"')


# ## Absorbing Boundary Condition

# In[ ]:


sim = EMSim(shape=(100,20), WL0=20)


# In[ ]:


sim.pType[:,[0,1,18,19]] = sim.DERZERO_CODE


# In[ ]:


[sim.setSource((10,y), 1) for y in range(0,20)];


# In[ ]:


sim.BuildSimBounds()


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# In[ ]:


sim.animateFields()


# ## Zero Boundary Condition

# In[ ]:


sim = EMSim(shape=(100,20), WL0=20)


# In[ ]:


sim.setPTypeRect(pType="zero", xyRange=((99,99),(0,19)))


# In[ ]:


sim.pType[:,[0,1,18,19]] = sim.DERZERO_CODE


# In[ ]:


[sim.setSource((10,y), 1) for y in range(2,18)];


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# In[ ]:


sim.animateFields()


# ## Zero Derivative Boundary Condition

# In[ ]:


sim = EMSim(shape=(100,20), WL0=20)


# In[ ]:


sim.setPTypeRect(pType="derZero", xyRange=((99,99),(0,19)))


# In[ ]:


sim.pType[:,[0,1,18,19]] = sim.DERZERO_CODE


# In[ ]:


[sim.setSource((10,y), 1) for y in range(2,18)];


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# ## Optical Index

# In[ ]:


sim = EMSim(shape=(100,20), WL0=20)


# In[ ]:


sim.setNIndexRect(xyRange=((50,100),(0,19)), n=1.5)


# In[ ]:


sim.setPTypeRect(pType="derZero", xyRange=((0,99),(18,19)))
sim.setPTypeRect(pType="derZero", xyRange=((0,99),(0,1)))


# In[ ]:


[sim.setSource((50,y), 1) for y in range(2,18)];


# In[ ]:


sim.BuildSimBounds()


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# In[ ]:


sim.animateFields()


# ## Snell's Law

# In[ ]:


deg = 2*np.pi/360


# In[ ]:


sim = EMSim(shape=(300,300), WL0=20)


# In[ ]:


sim.nIndex[150:, :] = 1.5


# In[ ]:


xc, yc = 150, 150
angle = (180 + 45)*deg
width = 70*deg
rad = 145
angles = np.linspace(angle-width/2, angle+width/2, num=100)
amps = np.exp(-np.linspace(-2,2,num=100)**2)
for (angle, amp) in zip(angles, amps):
    x = int(round(rad*np.cos(angle) + xc))
    y = int(round(rad*np.sin(angle) + yc))
    sim.setSource(xy=(x,y), val=amp)


# In[ ]:


sim.BuildSimBounds()


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# Snell's Law
# $$n_2 \sin(\theta_2) = n_1 \sin(\theta_1) $$

# In[ ]:


exitAngle = np.arcsin(1/1.5*np.sin(45*deg))


# In[ ]:


p = sim.visualizeFieldBokeh(purpose='id', imageWidth=300)


# In[ ]:


p.add_layout(Arrow(end=OpenHead(line_color="black", line_width=4),
                   x_start=xc-100, y_start=yc-100, x_end=xc-25, y_end=yc-25, line_color="black"))
p.add_layout(Arrow(end=OpenHead(line_color="black", line_width=4),
                   x_end=xc-100, y_end=yc+100, x_start=xc-25, y_start=yc+25))
p.add_layout(Arrow(end=OpenHead(line_color="black", line_width=4),
                   x_start=xc, y_start=yc, x_end=xc+100*np.cos(exitAngle), y_end=yc+100*np.sin(exitAngle)))


# In[ ]:


show(p)


# In[ ]:


sim.animateFields()


# ## Snell's Law (Total Internal Reflection)

# In[ ]:


deg = 2*np.pi/360


# In[ ]:


sim = EMSim(shape=(300,300), WL0=20)


# In[ ]:


sim.nIndex[:150, :] = 2.0


# In[ ]:


xc, yc = 150, 150
angle = (180 + 55)*deg
width = 70*deg
rad = 145
angles = np.linspace(angle-width/2, angle+width/2, num=100)
amps = np.exp(-np.linspace(-2,2,num=100)**2)
for (angle, amp) in zip(angles, amps):
    x = int(round(rad*np.cos(angle) + xc))
    y = int(round(rad*np.sin(angle) + yc))
    sim.setSource(xy=(x,y), val=amp)


# In[ ]:


exitAngle = np.arcsin(2/1*np.sin(45*deg))


# In[ ]:


sim.BuildSimBounds()


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh()


# In[ ]:


sim.animateFields()


# ## Total Internal Reflection - Polymodal Waveguide

# In[ ]:


sim = EMSim(shape=(300,100), WL0=20)


# In[ ]:


sim.nIndex[:,35:65] = 1.5


# In[ ]:


sim.setSource((25,40), 1)


# In[ ]:


sim.BuildSimBounds()


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh(maxRangeField=0.2)


# ## Double Total Internal Reflection (Monomodal)

# In[ ]:


from CurveGenerator import buildCurvedWaveguide


# In[ ]:


sim = EMSim(shape=(300,100), WL0=20)


# In[ ]:


wg1 = buildCurvedWaveguide([(0,50), (60,50), (122, 76), (200, 15), (260,50), (300,50)], rad=50, width=10)
wg1


# In[ ]:


sim.setNIndexMultiPolygon(wg1, 1.5)


# In[ ]:


sim.setSource((10,50), 1)


# In[ ]:


sim.BuildSimBounds()


# In[ ]:


sim.buildSparsePhysicsEqs()


# In[ ]:


sim.solve()


# In[ ]:


sim.visualizeAllBokeh(maxRangeField=0.2)


# In[ ]:


sim.animateFields(maxRange=0.1)


# In[ ]:




