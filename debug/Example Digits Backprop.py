
# coding: utf-8

# In[1]:


from classy import *


# In[2]:

images=image.load_images('data/digits')


# In[3]:

data=image.images_to_vectors(images)
data.vectors-=data.vectors.mean()
data.vectors/=data.vectors.std()


# In[4]:

data_train,data_test=split(data)
image.vector_to_image(data_train.vectors[800,:],(8,8))


# ## Do Perceptron First

# In[5]:

C=Perceptron()


# In[6]:

timeit(reset=True)
C.fit(data_train.vectors,data_train.targets)
print(("Training time: ",timeit()))


# In[7]:

print(("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets)))
print(("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets)))


# In[8]:

data_train.target_names


# In[ ]:




# In[9]:

import matplotlib.pyplot as plt
plt.figure(figsize=(16,4))
for i,t in enumerate(data_train.target_names):
    plt.subplot(2,10,i+1)
    vector=random_vector(data_train,t)
    image.vector_to_image(vector,(8,8))
    plt.axis('off')
    
    plt.subplot(2,10,i+11)
    image.vector_to_image(C.weights[i,:],(8,8))
    plt.axis('off')
    


# ## Do Backprop

# In[10]:

C=BackProp(hidden_layer_sizes = [12])


# In[11]:

timeit(reset=True)
C.fit(data_train.vectors,data_train.targets)
print(("Training time: ",timeit()))


# In[12]:

print(("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets)))
print(("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets)))


# In[13]:

len(C.layers_coef_)


# In[14]:

C.layers_coef_[0].shape,C.layers_coef_[1].shape


# In[ ]:



