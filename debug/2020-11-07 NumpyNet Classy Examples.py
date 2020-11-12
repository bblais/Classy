#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic('pylab inline')
from classy import *


# In[2]:


data=load_excel('data/iris.xls')
data_train,data_test=split(data,test_size=0.2)


# In[6]:


plot(data_train.targets,'o')


# In[3]:


C=NumPyNetBackProp({
    'input':4,               # number of features
    'hidden':[(5,'logistic'),],
    'output':(3,'logistic'),  # number of classes
    'cost':'mse',
})


# In[4]:


C.fit(data_train.vectors,data_train.targets,epochs=10)


# In[5]:


print("On Training Set:",C.percent_correct(data_train.vectors,data_train.targets))
print("On Test Set:",C.percent_correct(data_test.vectors,data_test.targets))


# In[6]:


C.weights


# In[ ]:





# ## XOR Problem - Perceptron

# In[7]:


data=make_dataset(bob=[[0,0],[1,1]],sally=[[0,1],[1,0]])


# In[8]:


data


# In[9]:


C=NumPyNetBackProp({
    'input':2,               # number of features
    'output':(2,'linear'),  # number of classes
    'cost':'mse',
})


# In[10]:


C.fit(data.vectors,data.targets)


# In[11]:


print((C.predict(data.vectors)))
print(("On Training Set:",C.percent_correct(data.vectors,data.targets)))


# In[12]:


plot2D(data,classifier=C,axis_range=[-.55,1.5,-.5,1.5])


# ## XOR Problem - Backprop

# In[13]:


data.vectors


# In[14]:


data.targets


# In[15]:


C=NumPyNetBackProp({
    'input':2,               # number of features
    'hidden':[(5,'logistic'),],
    'output':(2,'logistic'),  # number of classes
    'cost':'mse',
})


# In[16]:


C.fit(data.vectors,data.targets,epochs=3000)


# In[17]:


print((C.predict(data.vectors)))
print(("On Training Set:",C.percent_correct(data.vectors,data.targets)))


# In[18]:


plot2D(data,classifier=C,axis_range=[-.55,1.5,-.5,1.5])


# In[19]:


print((data.vectors))
print()
print((data.targets))


# In[20]:


C.output(data.vectors)


# In[21]:


h,y=C.output(data.vectors)
print(h)
print() 
print((np.round(h)))
print()
print(y)


# In[ ]:





# In[23]:


print(around(C.weights[0],2))
around(C.weights[1],2)


# In[ ]:





# ## Curvy data

# In[24]:


figure(figsize=(8,8))
N=30
x1=randn(N)*.2
y1=randn(N)*.2

plot(x1,y1,'bo')

a=linspace(0,3*pi/2,N)
x2=cos(a)+randn(N)*.2
y2=sin(a)+randn(N)*.2

plot(x2,y2,'rs')

axis('equal')


# In[25]:


vectors=vstack([hstack([atleast_2d(x1).T,atleast_2d(y1).T]),
        hstack([atleast_2d(x2).T,atleast_2d(y2).T]),
        ])
targets=concatenate([zeros(N),ones(N)])
target_names=['center','around']
feature_names=['x','y']


# In[26]:


data=Struct(vectors=vectors,targets=targets,
                target_names=target_names,feature_names=feature_names)


# In[27]:


C=NumPyNetBackProp({
    'input':2,               # number of features
    'output':(2,'linear'),  # number of classes
    'cost':'mse',
})


# In[28]:


C.fit(data.vectors,data.targets)
print(("On Training Set:",C.percent_correct(data.vectors,data.targets)))
plot2D(data,classifier=C,axis_range=[-2,2,-2,2])


# In[30]:


C=NumPyNetBackProp({
    'input':2,               # number of features
    'hidden':[(5,'logistic'),],
    'output':(2,'logistic'),  # number of classes
    'cost':'mse',
})
C.fit(data.vectors,data.targets,epochs=3000)
print(("On Training Set:",C.percent_correct(data.vectors,data.targets)))
plot2D(data,classifier=C,axis_range=[-2,2,-2,2])


# In[31]:


C=NaiveBayes()
C.fit(data.vectors,data.targets)
print(("On Training Set:",C.percent_correct(data.vectors,data.targets)))
C.plot_centers()
plot2D(data,classifier=C,axis_range=[-2,2,-2,2])


# In[32]:


C=kNearestNeighbor()
C.fit(data.vectors,data.targets)
print(("On Training Set:",C.percent_correct(data.vectors,data.targets)))
plot2D(data,classifier=C,axis_range=[-2,2,-2,2])


# In[33]:


C=CSC()
C.fit(data.vectors,data.targets)
print(("On Training Set:",C.percent_correct(data.vectors,data.targets)))
C.plot_centers()
plot2D(data,classifier=C,axis_range=[-2,2,-2,2])


# In[ ]:





# ## 8x8 - Autoencoder

# In[3]:


vectors=eye(8)
targets=arange(8)
print((vectors,targets))


# In[4]:


import pandas as pd


# In[7]:


pd.DataFrame(vectors).to_excel('/Users/bblais/Downloads/eye.xlsx')


# In[43]:


targets


# In[60]:


C=NumPyNetBackProp({
    'input':8,               # number of features
    'hidden':[(3,'logistic'),],  # bottleneck (num hidden < num inputs)
    'output':(8,'logistic'),  # number of classes
    'cost':'mse',
})


# In[61]:


C.fit(vectors,targets,epochs=10000)
print(("On Training Set:",C.percent_correct(vectors,targets)))
print((C.predict(vectors)))


# In[62]:


h,y=C.output(vectors)


# In[63]:


around(h,2)


# In[64]:


h.round()


# In[65]:


y.round()


# In[66]:


C.predict(vectors)


# In[67]:


y.shape


# In[68]:


imshow(h,interpolation='nearest',cmap=cm.gray)
colorbar()


# In[69]:


weights_xh,weights_hy=C.weights


# In[73]:


plot(weights_xh,'-o');


# In[72]:


plot(weights_hy,'-o');


# In[ ]:




