#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic('pylab inline')


# In[2]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# In[3]:


import classy as cl


# In[4]:


def Xy_image(images):
    X=np.dstack(images.data)
    X=X.transpose([2,0,1]).astype(np.float)
    y=images.targets
    X = np.asarray([np.dstack((x, x, x)) for x in X])
    return X,y


# In[5]:


np.random.seed(124)

images=cl.image.load_images('data/digits')

images_train,images_test=cl.image.split(images,verbose=False)
cl.summary(images_train)
cl.summary(images_test)

X,y=Xy_image(images)




X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y,
                                                  test_size=.2,
                                                  random_state=42)



X_train2,y_train2=Xy_image(images_train)
X_test2,y_test2=Xy_image(images_test)


# In[6]:


plot(y_train1,'o')


# In[7]:


plot(y_train2,'o')


# In[14]:


hist(X_train1.ravel(),255);


# In[15]:


hist(X_train2.ravel(),255);


# In[16]:


get_ipython().magic('pinfo train_test_split')


# In[26]:


idx=np.array(range(len(y_train2)))
np.random.shuffle(idx)
y_train2=y_train2[idx]
X_train2=X_train2[idx,...]


# In[21]:


idx


# In[24]:


X_train2.shape


# In[27]:


a=[1,2,3,4]
np.random.shuffle(a)
a


# In[28]:


type(a)


# In[ ]:




