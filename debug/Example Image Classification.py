
# coding: utf-8

# In[1]:


from classy import *


# ## Load the Images and Reshape into vectors-targets

# In[2]:

images=image.load_images('data/digits')


# In[3]:

data=image.images_to_vectors(images)


# In[4]:

data.vectors.shape


# In[5]:

data_train,data_test=split(data)


# ## View one of the vectors, and possibly save it to a file

# In[6]:

image.vector_to_image(data_train.vectors[800,:],(8,8))


# only do this if you want to save the actual image

# In[7]:

image.vector_to_image(data_train.vectors[800,:],(8,8),'test.png')


# ## Classification

# In[8]:

C=NaiveBayes()


# In[9]:

timeit(reset=True)
C.fit(data_train.vectors,data_train.targets)
print(("Training time: ",timeit()))


# In[10]:

print(("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets)))
print(("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets)))


# In[11]:

C=CSC()


# In[12]:

timeit(reset=True)
C.fit(data_train.vectors,data_train.targets)
print(("Training time: ",timeit()))


# In[13]:

print(("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets)))
print(("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets)))


# ## Footnote

# ### Loading Files with Patterns
# 
# Here is a little note about how to load data from folders, using the filenames and not the folder structure.  

# In[14]:

from classy import *


# here the pattern translates to (note the asterisks "*" in the pattern)
# 
# * "data/digits/(all folders)/(any .png file starting with 133)"
# * "data/digits/(all folders)/(any .png file starting with 123)"

# In[15]:

data=image.load_images_from_filepatterns(this='data/digits/*/133*.png',
                                         that='data/digits/*/123*.png',
                                         )


# In[16]:

summary(data)


# In[ ]:



