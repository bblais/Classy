#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic('pylab inline')
from classy import *


# In[2]:


images=image.load_images('data/digits')


# In[3]:


summary(images)


# In[4]:


data=image.images_to_vectors(images)
data.vectors-=data.vectors.mean()
data.vectors/=data.vectors.std()


# In[5]:


data_train,data_test=split(data)
n=800
image.vector_to_image(data_train.vectors[n,:],(8,8))
title("The number %s" % str(data_train.target_names[data_train.targets[n]]))


# In[6]:


data_train.vectors.shape


# In[7]:


num_samples,num_features=data_train.vectors.shape
num_classes=len(data_train.target_names)


# ## Perceptron

# In[8]:


C=NumPyNetBackProp({
    'input':num_features,               # number of features
    'output':(num_classes,'linear'),  # number of classes
    'cost':'mse',
})


# In[9]:


C.fit(data_train.vectors,data_train.targets)


# In[10]:


print(("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets)))
print(("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets)))


# In[11]:


figure(figsize=(16,4))
for i,t in enumerate(data_train.target_names):
    subplot(2,10,i+1)
    vector=random_vector(data_train,t)
    image.vector_to_image(vector,(8,8))
    axis('off')
    
    subplot(2,10,i+11)
    image.vector_to_image(C.weights[0][:,i],(8,8))
    axis('off')
    


# ## Backprop

# In[12]:


C=NumPyNetBackProp({
    'input':num_features,               # number of features
    'hidden':[(12,'logistic'),],
    'output':(num_classes,'logistic'),  # number of classes
    'cost':'mse',
})


# In[13]:


C.fit(data_train.vectors,data_train.targets,epochs=5000)


# In[14]:


print(("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets)))
print(("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets)))


# In[15]:


weights_ih=C.weights[0]
weights_hy=C.weights[-1]


# In[16]:


weights_ih.shape


# In[17]:


for i in range(weights_ih.shape[1]):
    subplot(3,4,i+1)
    image.vector_to_image(weights_ih[:,i],(8,8))
    axis('off')


# ## Convolutional Neural Net

# In[18]:


images=image.load_images('data/digits')
C=NumPyNetImageNN(images)

