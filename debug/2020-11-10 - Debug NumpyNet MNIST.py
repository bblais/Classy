#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().magic('pylab inline')


# - numpynet from sklearn mnist data set works
# - numpynet/classy from digits data set doesn't work
# - trying numpynet with digits data set here works
# - what is the numpynet/classy difference?

# In[2]:


'''
Little example on how to use the Network class to create a model and perform
a basic classification of the MNIST dataset
'''

#from NumPyNet.layers.input_layer import Input_layer
from NumPyNet.layers.connected_layer import Connected_layer
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from NumPyNet.layers.maxpool_layer import Maxpool_layer
from NumPyNet.layers.softmax_layer import Softmax_layer
# from NumPyNet.layers.dropout_layer import Dropout_layer
# from NumPyNet.layers.cost_layer import Cost_layer
# from NumPyNet.layers.cost_layer import cost_type
from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
from NumPyNet.network import Network
from NumPyNet.optimizer import Adam
# from NumPyNet.optimizer import Adam, SGD, Momentum
from NumPyNet.utils import to_categorical
from NumPyNet.utils import from_categorical
from NumPyNet.metrics import mean_accuracy_score

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


def accuracy (y_true, y_pred):
  '''
  Temporary metrics to overcome "from_categorical" missing in standard metrics
  '''
  truth = from_categorical(y_true)
  predicted = from_categorical(y_pred)
  return mean_accuracy_score(truth, predicted)




# In[3]:


import classy as cl


# In[4]:


digits = datasets.load_digits()
X, y = digits.images, digits.target


# In[5]:


X.shape


# In[6]:


imshow(X[100,:,:])


# In[7]:


y


# In[8]:


images=cl.image.load_images('data/digits')


# In[9]:


X=np.dstack(images.data)
X=X.transpose([2,0,1])
X.shape


# In[10]:


n=500
imshow(X[n,:,:])
title(images.target_names[images.targets[n]])


# ## compare shapes

# In[11]:


digits = datasets.load_digits()
X, y = digits.images, digits.target
print(X.shape,y.shape)
X = np.asarray([np.dstack((x, x, x)) for x in X])
y=to_categorical(y).reshape(len(y), 1, 1, -1)
print(X.shape,y.shape)


# In[12]:


images=cl.image.load_images('data/digits')
X=np.dstack(images.data)
X=X.transpose([2,0,1]).astype(np.float)
y=images.targets
print(X.shape,y.shape)
X = np.asarray([np.dstack((x, x, x)) for x in X])
y=to_categorical(y).reshape(len(y), 1, 1, -1)
print(X.shape,y.shape)


# In[22]:


def Xy_image(images,shuffle=True):
    import numpy as np
    
    X=np.dstack(images.data)
    X=X.transpose([2,0,1]).astype(np.float)
    y=images.targets
    X = np.asarray([np.dstack((x, x, x)) for x in X])
    
    if shuffle:
        idx=np.array(range(len(y)))
        np.random.shuffle(idx)
        y=y[idx]
        X=X[idx,...]
    
    return X,y


# In[23]:


np.random.seed(124)

images=cl.image.load_images('data/digits')

images_train,images_test=cl.image.split(images,verbose=False)
cl.summary(images_train)
cl.summary(images_test)

X,y=Xy_image(images)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                  test_size=.2,
                                                  random_state=42)



X_train,y_train=Xy_image(images_train)
X_test,y_test=Xy_image(images_test)


batch = 128
num_classes = len(set(y))

# del X, y

# normalization to [0, 1]
X_train *= 1. / 255.
X_test  *= 1. / 255.

# reduce the size of the data set for testing
############################################

# train_size = 512
# test_size  = 300



# X_train = X_train[:train_size, ...]
# y_train = y_train[:train_size]
# X_test  = X_test[ :test_size,  ...]
# y_test  = y_test[ :test_size]

############################################

n_train = X_train.shape[0]
n_test  = X_test.shape[0]

# transform y to array of dimension 10 and in 4 dimension
y_train = to_categorical(y_train).reshape(n_train, 1, 1, -1)
y_test  = to_categorical(y_test).reshape(n_test, 1, 1, -1)

# Create the model and training
model = Network(batch=batch, input_shape=X_train.shape[1:])

model.add(Convolutional_layer(size=3, filters=32, stride=1, pad=True, activation='Relu'))

model.add(BatchNorm_layer())

model.add(Maxpool_layer(size=2, stride=1, padding=True))

model.add(Connected_layer(outputs=100, activation='Relu'))

model.add(BatchNorm_layer())

model.add(Connected_layer(outputs=num_classes, activation='Linear'))

model.add(Softmax_layer(spatial=True, groups=1, temperature=1.))
# model.add(Cost_layer(cost_type=cost_type.mse))

# model.compile(optimizer=SGD(lr=0.01, decay=0., lr_min=0., lr_max=np.inf))
model.compile(optimizer=Adam(), metrics=[accuracy])

print('*************************************')
print('\n Total input dimension: {}'.format(X_train.shape), '\n')
print('**************MODEL SUMMARY***********')

model.summary()

print('\n***********START TRAINING***********\n')

# Fit the model on the training set
model.fit(X=X_train, y=y_train, max_iter=10, verbose=True)

print('\n***********START TESTING**************\n')


print(X_test.shape,y_test.shape)
# Test the prediction with timing
loss, out = model.evaluate(X=X_test, truth=y_test, verbose=True)

truth = from_categorical(y_test)
predicted = from_categorical(out)
accuracy_score  = mean_accuracy_score(truth, predicted)

print('\nLoss Score: {:.3f}'.format(loss))
print('Accuracy Score: {:.3f}'.format(accuracy_score))
# SGD : best score I could obtain was 94% with 10 epochs, lr = 0.01 %
# Momentum : best score I could obtain was 93% with 10 epochs
# Adam : best score I could obtain was 95% with 10 epochs


# In[14]:


L=model._net[1]


# In[13]:


L.weights.shape


# In[7]:


num_filters=L.weights.shape[-1]

w=L.weights
w=w-w.min()
w=w/w.max()
figure(figsize=(6,12))
for f in range(num_filters):
    subplot(8,4,f+1)
    im=w[:,:,:,f]    
    imshow(im)
    axis('off')


# In[8]:


X.max()


# In[9]:


X.min()


# In[10]:


X_train.min()


# In[11]:


X_train.max()


# In[12]:


digits = datasets.load_digits()
X, y = digits.images, digits.target


# In[13]:


X.shape


# In[15]:


imshow(X[0,:,:])
colorbar()


# In[16]:


X[0,:,:]


# In[ ]:




