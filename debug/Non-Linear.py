#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic('pylab inline')


# In[2]:


from classy import *


# In[3]:


images=image.load_images('/Users/bblais/Desktop/ai373/google/students/Michael_Andrejco/Sprint_5/Data')


# In[4]:


data=image.images_to_vectors(images)


# In[5]:


C=NumPyNetBackProp({
    'input':4800,               # number of features
    'output':(2,'linear'),  # number of classes
    'cost':'mse',
})


# In[6]:


C.fit(data.vectors,data.targets)


# In[ ]:





# In[ ]:




