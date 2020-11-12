#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic('pylab inline')
from classy import *


# Note to self: the vectors should be shuffled, or convergence isn't great

# In[2]:


images=image.load_images('data/digits')


# In[3]:


summary(images)


# In[4]:


images.keys()


# ## Convolutional Neural Net

# In[5]:


images=image.load_images('data/digits')
num_classes=len(images.target_names)
images.data=[_/255.0 for _ in images.data]


C=NumPyNetImageNN(
    Convolutional_layer(size=3, filters=32, stride=1, pad=True, activation='Relu'),
    BatchNorm_layer(),
    Maxpool_layer(size=2, stride=1, padding=True),
    Connected_layer(outputs=100, activation='Relu'),
    BatchNorm_layer(),
    Connected_layer(outputs=num_classes, activation='Linear'),
    Softmax_layer(spatial=True, groups=1, temperature=1.),
    )


# In[6]:


images_train,images_test=image.split(images,verbose=False)
summary(images_train)
summary(images_test)


# In[7]:


C.fit(images_train,epochs=10,batch=128)


# In[8]:


C.predict(images_test)


# In[9]:


print("On Training Set:",C.percent_correct(images_train))
print("On Test Set:",C.percent_correct(images_test))


# In[ ]:




