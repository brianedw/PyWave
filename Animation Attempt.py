#!/usr/bin/env python
# coding: utf-8

# # 1D Case

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import HTML
import numpy as np


# In[3]:


from matplotlib import _version


# In[4]:


_version.get_versions()


# In[5]:


def makeAnimation(xs, U):
    fig, ax = plt.subplots()
    ax.axis([min(xs),max(xs),-2,2])
    l, = ax.plot([],[])

    def animate(i):
        l.set_data(xs, np.real(np.exp(1j*2*np.pi*i/100)*U))

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=100-1)
    return HTML(ani.to_jshtml())


# In[6]:


xs = np.arange(0,10,0.1)
U = np.exp(-1j*(2*np.pi/1.0)*xs)
makeAnimation(xs,U)


# In[7]:


xs = np.arange(0,10,0.1)
U = np.exp(-1j*(2*np.pi/2)*xs)
makeAnimation(xs,U)


# # 2D Case

# In[20]:


fig = plt.figure()
ax = plt.axes(xlim=(-0.5, 4.5), ylim=(-.5, 4.5))
ax.axis('off')

aC=np.random.random((5,5))+1j*np.random.random((5,5))
im=plt.imshow(np.abs(aC),interpolation='none',)
# initialization function: plot the background of each frame
def init():
    im.set_data(np.abs(aC))
    return im

# animation function.  This is called sequentially
def animate(i):
    a=np.real(aC*np.exp(2*np.pi*1j*i/20))    # exponential decay of the values
    im.set_array(a)
    return im

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=20-1)
HTML(ani.to_jshtml())


# In[18]:





# In[ ]:




