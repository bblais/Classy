#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *


# In[2]:


from classy import *


# In[4]:


images=image.load_images('images/square images/')


# In[5]:


data=image.images_to_vectors(images)


# In[32]:


utils.standardize(data)


# In[33]:


summary(data)


# In[34]:


data_train,data_test=split(data,test_size=0.2)


# In[35]:


C=NaiveBayes()
C.fit(data_train.vectors,data_train.targets)
print("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets))


# In[36]:


C.save('test_save_naivebayes_images.json')


# In[37]:


C1=NaiveBayes()
C1.load('test_save_naivebayes_images.json')
print("On Training Set:",C1.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C1.percent_correct(data_test.vectors,data_test.targets))


# In[ ]:





# In[38]:


C=kNearestNeighbor()
C.fit(data_train.vectors,data_train.targets)
print("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets))


# In[39]:


C.save('test_save_knn_images.json')


# In[40]:


C1=kNearestNeighbor()
C1.load('test_save_knn_images.json')
print("On Training Set:",C1.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C1.percent_correct(data_test.vectors,data_test.targets))


# In[41]:


C=RCE()
C.fit(data_train.vectors,data_train.targets)
print("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets))


# In[42]:


C.save('test_save_rce_images.json')


# In[43]:


C1=RCE()
C1.load('test_save_rce_images.json')
print("On Training Set:",C1.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C1.percent_correct(data_test.vectors,data_test.targets))


# In[44]:


C=CSC()
C.fit(data_train.vectors,data_train.targets)
print("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets))


# In[45]:


C.save('test_save_csc_images.json')


# In[46]:


C1=CSC()
C1.load('test_save_csc_images.json')
print("On Training Set:",C1.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C1.percent_correct(data_test.vectors,data_test.targets))


# In[47]:


number_of_features=data_train.vectors.shape[1]
number_of_categories=len(set(data_train.targets))  # the types of pieces
print("Number of features:",number_of_features)
print("Number of categories:",number_of_categories)


# ## Perceptron

# In[48]:


C=NumPyNetBackProp({
    'input':number_of_features,               # number of features
    'output':(number_of_categories,'linear'),  # number of classes
    'cost':'mse',
})
C.fit(data_train.vectors,data_train.targets,epochs=3000)


# In[49]:


print("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets))


# In[50]:


C.save('test_save_perceptron_images.json')


# In[51]:


C1=NumPyNetBackProp()
C1.load('test_save_perceptron_images.json')
print("On Training Set:",C1.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C1.percent_correct(data_test.vectors,data_test.targets))


# ## Backprop - Multilayer

# In[52]:


C=NumPyNetBackProp({
    'input':number_of_features,               # number of features
    'hidden':[(15,'logistic'),],   # this size is "arbitrary"
    'output':(number_of_categories,'logistic'),  # number of classes
    'cost':'mse',
})
C.fit(data_train.vectors,data_train.targets,epochs=3000)


# In[53]:


print("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets))


# In[54]:


C.save('test_save_backprop_images.json')


# In[55]:


C1=NumPyNetBackProp()
C1.load('test_save_backprop_images.json')
print("On Training Set:",C1.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C1.percent_correct(data_test.vectors,data_test.targets))


# In[ ]:




