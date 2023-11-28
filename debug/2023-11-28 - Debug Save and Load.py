#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
from classy import *


# In[2]:


data=load_excel('data/iris.xls',verbose=True)


# In[3]:


C=NaiveBayes()


# In[4]:


timeit(reset=True)
C.fit(data.vectors,data.targets)
print("Training time: ",timeit())


# In[5]:


C.predict(atleast_2d(array([1,2,3,4])))


# In[6]:


C.save('test_save_naivebayes.json')


# In[7]:


C1=NaiveBayes()


# In[8]:


C1.load('test_save_naivebayes.json')
C1.predict(atleast_2d(array([1,2,3,4])))


# In[9]:


C=kNearestNeighbor()
C.fit(data.vectors,data.targets)
C.predict(atleast_2d(array([1,2,3,4])))


# In[10]:


C.save('test_save_kNearestNeighbor.json')


# In[11]:


C1=kNearestNeighbor()


# In[12]:


S1=dir(C1)
S2=dir(C)


# In[13]:


set(S2)-set(S1)


# In[14]:


C1=kNearestNeighbor()
C1.load('test_save_kNearestNeighbor.json')
C1.predict(atleast_2d(array([1,2,3,4])))


# In[15]:


C=CSC()
C.fit(data.vectors,data.targets)
C.predict(atleast_2d(array([1,2,3,4])))


# In[16]:


C1=CSC()


# In[17]:


S1=dir(C1)
S2=dir(C)
set(S2)-set(S1)


# In[18]:


C.save('test_save_CSC.json')


# In[19]:


C1.load('test_save_CSC.json')
C1.predict(atleast_2d(array([1,2,3,4])))


# In[20]:


C=RCE()
C.fit(data.vectors,data.targets)
C.predict(atleast_2d(array([1,2,3,4])))


# In[21]:


C.save('test_save_RCE.json')


# In[22]:


C1=RCE()
C1.load('test_save_RCE.json')
C1.predict(atleast_2d(array([1,2,3,4])))


# In[23]:


C=NumPyNetBackProp({
    'input':4,               # number of features
    'output':(3,'linear'),  # number of classes
    'cost':'mse',
})


# In[24]:


C.fit(data.vectors,data.targets,epochs=3000)


# In[25]:


C.predict(atleast_2d(array([1,2,3,4])))


# In[26]:


C.save('test_save_nn.json')


# In[27]:


M=C.model


# In[28]:


get_ipython().run_line_magic('pinfo', 'M.save_model')


# In[29]:


get_ipython().run_line_magic('pinfo', 'M.load_model')


# In[30]:


L=C.model.__dict__['_net'][1]


# In[31]:


M.__dict__


# In[32]:


C.model.batch


# In[33]:


C1=NumPyNetBackProp({
    'input':4,               # number of features
    'output':(3,'linear'),  # number of classes
    'cost':'mse',
})


# In[34]:


C1.init_model(C.model.batch)

layer_weights=[L.weights if 'weights' in L.__dict__ else [] for L in C.model._net ]
layer_bias=[L.bias if 'bias' in L.__dict__ else [] for L in C.model._net ]

for L,W,B in zip(C1.model._net,layer_weights,layer_bias):
    if 'weights' in L.__dict__:
        L.weights=W
        L.bias=B

C1.model._fitted=True


# In[35]:


C1.predict(atleast_2d(array([1,2,3,4])))


# In[36]:


C.predict(atleast_2d(array([1,2,3,4])))


# In[37]:


C1=NumPyNetBackProp()
C1.load('test_save_nn.json')
C1.predict(atleast_2d(array([1,2,3,4])))


# In[ ]:




