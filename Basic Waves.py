#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('run', '"./Helmholtz.ipynb"')


# In[ ]:


emsim = EMSim(shape=(250,250), WL0=20)


# In[ ]:


emsim.setNIndexRect()


# In[ ]:


emsim.indRect(((0,3),(10,14)))


# In[ ]:




