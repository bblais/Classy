
import numpy as np
import pylab as pl
from . import utils
from sklearn.utils import check_X_y,check_array
from sklearn.neural_network import MLPClassifier as MultilayerPerceptronClassifier
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import SGDClassifier

from .supervised_numpynet import *

import json
class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                return obj.tolist()
            else:
                return [self.default(obj[i]) for i in range(obj.shape[0])]
        return json.JSONEncoder.default(self, obj)


class GenericClassifier(object):
    def percent_correct(self,vectors,targets):
        return self.score(vectors,targets)*100.0
    def predict_names(self,vectors,names):
        result=self.predict(vectors)
        return [names[i] for i in result]

class SVM(SVC,GenericClassifier):
    pass

class LogisticRegression(LogReg,GenericClassifier):
    pass

   
class BackProp(MultilayerPerceptronClassifier,GenericClassifier):
    def __init__(self,**kwargs):
        if 'tol' not in kwargs:
            kwargs['tol']=1e-7
    
        MultilayerPerceptronClassifier.__init__(self,**kwargs)
        self.equivalent={'weights':'coefs_',
        }
                         
        self.__dict__.update(self.equivalent)
        
    def fit(self,*args,**kwargs):
            
        MultilayerPerceptronClassifier.fit(self,*args,**kwargs)
        for name in self.equivalent:
            super(MultilayerPerceptronClassifier,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
    

    def output(self, X):
        """Fit the model to the data X and target y.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        array, shape (n_samples)
        Predicted target values per element in X.
        """

        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + \
            [self.n_outputs_]

        # Initialize layers
        activations = []
        activations.append(X)

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0],
                                         layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations)
        y_pred = activations[-1]

        return activations[1:]


from sklearn.neighbors import KNeighborsClassifier
class kNearestNeighbor(KNeighborsClassifier,GenericClassifier):
    
    def __init__(self,k=5):
        self.k=k        
        KNeighborsClassifier.__init__(self,n_neighbors=k)
        self.components=['classes_', 'k','n_features_in_','_fit_X',
                        '_fit_method',
                        #'_tree',
                        '_y',
                        'classes_',
                        'effective_metric_',
                        'effective_metric_params_',
                        'n_features_in_',
                        'n_samples_fit_',
                        'outputs_2d_']

        self.as_array=['_fit_X','_y','classes_']
        self.equivalent={}

    def save(self,filename):
        D={}
        for key in self.components:
            D[key]=self.__getattribute__(key)

        
        with open(filename, 'w') as f:
            json.dump(D,f, sort_keys=True, indent=4,cls=NumpyAwareJSONEncoder)        
        
    def load(self,filename):
        from sklearn.neighbors import KDTree 

        with open(filename, 'r') as f:
            D=json.load(f)

        for key in self.components:
            val=D[key]

            if key in self.as_array:
                val=np.array(val)

            try:
                super(KNeighborsClassifier,self).__setattr__(key,val)
            except AttributeError:
                print("Can't",key,val)
                raise

        for name in self.equivalent:
            super(KNeighborsClassifier,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))

        self._tree=KDTree(self._fit_X)



from sklearn.naive_bayes import GaussianNB
class NaiveBayes(GaussianNB,GenericClassifier):

    def __init__(self):
        GaussianNB.__init__(self)
        self.var_smoothing=1e-2  # make it much more stable
        self.equivalent={'means':'theta_',
                         'stddevs':'var_',
                         'fraction':'class_prior_'}

        self.components=['class_count_','class_prior_','classes_','n_features_in_',
        'var_','theta_','epsilon_',]

        #self.__dict__.update(self.equivalent)

    def fit(self,*args,**kwargs):
            
        GaussianNB.fit(self,*args,**kwargs)
        try:
            v=self.__getattribute__('var_')
        except AttributeError:  # changing the names of features hack
            self.var_=self.sigma_

        for name in self.equivalent:
            super(GaussianNB,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
    
        self.stddevs=np.sqrt(self.stddevs)

    def anotherfit(self, X, y):
        X,y=check_X_y(X,y)
            
        GaussianNB.fit(self,X,y)
    
        for name in self.equivalent:
            super(GaussianNB,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
    
    def predict_probability(X):
        return predict_proba(X)

    def plot_centers(self):
        ax=pl.gca().axis()
        colors=utils.bold_colors
        angle=np.linspace(0,2*np.pi,100)  
        i=0
        for c,r in zip(self.means,self.stddevs):
            pl.plot(c[0],c[1],'*',color=colors[i],markersize=15)
            i+=1
        i=0
        for c,r in zip(self.means,self.stddevs):
            for k in range(3):        
                xd=np.cos(angle)*r[0]*(k+1) + c[0]
                yd=np.sin(angle)*r[1]*(k+1) + c[1]
                pl.plot(xd,yd,'-',linewidth=3,color='k',alpha=0.5)
            i+=1
            
        #pl.axis('equal')
        pl.gca().axis(ax)


    def save(self,filename):
        D={}
        for key in self.components:
            D[key]=self.__getattribute__(key)

        
        with open(filename, 'w') as f:
            json.dump(D,f, sort_keys=True, indent=4,cls=NumpyAwareJSONEncoder)        

    def load(self,filename):
        with open(filename, 'r') as f:
            D=json.load(f)

        for key in self.components:
            val=D[key]

            try:
                val[0]
                val=np.array(val)
            except TypeError:
                pass

        for name in self.equivalent:
            super(GaussianNB,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
        self.stddevs=np.sqrt(self.stddevs)
    

    def __getattr__(self, item):
        translate={'var_':'sigma_',
                  }

        if item in translate:
            value=self.__dict__[translate[item]]
            self.__dict__[item]=value
        else:
            raise AttributeError(item)
        
        
        return value

from sklearn.linear_model import Perceptron as skPerceptron
class Perceptron(skPerceptron,GenericClassifier):

    def __init__(self,number_of_iterations=50,tol=1e-3):
        skPerceptron.__init__(self,shuffle=True,max_iter=number_of_iterations,tol=tol)
        
        self.equivalent={'weights':'coef_',
                         'biases':'intercept_',
                         }
        #self.__dict__.update(self.equivalent)

    def fit(self,*args,**kwargs):
            
        skPerceptron.fit(self,*args,**kwargs)
        for name in self.equivalent:
            super(skPerceptron,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))

    def output(self,vectors):
        return self.decision_function(vectors)    
        
        
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.base import BaseEstimator, ClassifierMixin

class RCEsk(BaseEstimator, ClassifierMixin):
    def __init__(self, metric='euclidean',r_min=0.1,r_max=1.0,r_step=1e-30,verbose=False):
        self.r_min=r_min
        self.r_max=r_max
        self.r_step=r_step
        self.metric = metric
        self.centers_=np.array([],dtype=float)
        self.radii_=np.array([],dtype=float)
        self.targets_=np.array([],dtype=int)
        self.verbose=verbose
        
    def _add_center(self,center,radius,target):
        try:
            center=center.toarray()  # deal with sparse
        except AttributeError:
            pass
        
        center=np.array(center,dtype=float)
        radius=np.array([radius],dtype=float)
        target=np.array([target],dtype=int)
        if len(self.centers_)==0:
            self.centers_=center
            self.targets_=target
            self.radii_=radius
        else:
            self.centers_=np.vstack( (self.centers_,center) )
            self.targets_=np.concatenate( (self.targets_,target) )
            self.radii_=np.concatenate( (self.radii_,radius) )
        
        
    def fit(self, X, y):
        X,y=check_X_y(X,y)
        # X, y = check_arrays(X, y, sparse_format="csr")
        # y = column_or_1d(y, warn=True)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError('y has fewer than 2 classes')

        if len(self.centers_)>0:
            assert len(self.centers_[0])==n_features
        
        # first pass
        pass_number=0
        for v,t in zip(X,y):  # Go through all of the data points
            v=v.reshape(1, -1)

            if len(self.centers_)==0:
                self._add_center(v,self.r_max,t)
                continue
            
            match=self.targets_[ (pairwise_distances(v,self.centers_,metric=self.metric)<self.radii_).ravel() ]
            
            # if a point is not already in a sphere, of correct category, 
            # add a sphere, centered at that point, of the correct category
            if not t in match:  
                self._add_center(v,self.r_max,t)
                continue
        pass_number+=1
        if self.verbose:
            print("%d clusters." % (len(self.centers_)))

        # second pass
        stop=False
        while not stop:
            old_radii_=self.radii_.copy()
            for v,t in zip(X,y):  # Go through all of the data points (again) 
                v=v.reshape(1, -1)

                D=pairwise_distances(v,self.centers_,metric=self.metric).ravel()
                within_centers=(D<self.radii_)
                matched=(t==self.targets_) & (within_centers)
                
                # not already in a sphere, of correct category --> add a sphere, 
                # centered at that point, of the correct category
                if not any(matched):
                    self._add_center(v,self.r_max,t)
                    continue

                not_matched=(t!=self.targets_) & (within_centers)
                # in a sphere of wrong category -- > shrink the wrong sphere as much as possible
                self.radii_[not_matched]-=D[not_matched]-self.r_step
                self.radii_[self.radii_<self.r_min]=self.r_min
                
            pass_number+=1
            
            if self.verbose:
                print("%d clusters." % (len(self.centers_)))
            if len(old_radii_)!=len(self.radii_):
                continue
            # Repeat until no changes    
            if sum(abs(self.radii_-old_radii_))<1e-10:
                stop=True
                
                
    def predict(self,X):
        X = check_array(X)        
        if len(self.centers_)==0:
            raise AttributeError("Model has not been trained yet.")
        
        result=[]
        for vector in X:
            vector=vector.reshape(1, -1)

            D=pairwise_distances(vector, self.centers_, metric=self.metric)/self.radii_
            result.append(self.targets_[D.argmin()])
            
        return np.array(result)
        
        
class RCE(RCEsk,GenericClassifier):

    def __init__(self, **kwargs):


        RCEsk.__init__(self, **kwargs)
        self.equivalent={'centers':'centers_',
                         'radii':'radii_',
                         'targets':'targets_'}
        self.components=['centers_','radii_','targets_']
        self.as_array=self.components
                         
        self.__dict__.update(self.equivalent)

    def fit(self,*args,**kwargs):
        RCEsk.fit(self,*args,**kwargs)
        for name in self.equivalent:
            super(RCE,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))

    def plot_centers(self):
        colors=utils.bold_colors
        for c,r,t in zip(self.centers_,self.radii_,self.targets_):
            pl.plot(c[0],c[1],'*',color=colors[t])
        angle=np.linspace(0,2*np.pi,100)  
        for c,r,t in zip(self.centers_,self.radii_,self.targets_):
            xd=np.cos(angle)*r + c[0]
            yd=np.sin(angle)*r + c[1]
            pl.plot(xd,yd,'-',color=colors[t])
            
        pl.axis('equal')
        
    def save(self,filename):
        D={}
        for key in self.components:
            D[key]=self.__getattribute__(key)

        
        with open(filename, 'w') as f:
            json.dump(D,f, sort_keys=True, indent=4,cls=NumpyAwareJSONEncoder)        

    def load(self,filename):

        with open(filename, 'r') as f:
            D=json.load(f)

        for key in self.components:
            val=D[key]

            if key in self.as_array:
                val=np.array(val)

            try:
                super(RCEsk,self).__setattr__(key,val)
            except AttributeError:
                print("Can't",key,val)
                raise

        for name in self.equivalent:
            super(RCEsk,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))

        
        
class CSCsk(BaseEstimator, ClassifierMixin):
    def __init__(self, metric='euclidean',r_step=1e-30,verbose=False):
        self.r_step=r_step
        self.metric = metric
        self.centers_=np.array([],dtype=float)
        self.radii_=np.array([],dtype=float)
        self.targets_=np.array([],dtype=int)
        self.verbose=verbose
        
    def _add_center(self,center,radius,target):
        try:
            center=center.toarray()  # deal with sparse
        except AttributeError:
            pass

        center=np.array(center,dtype=float)
        radius=np.array([radius],dtype=float)
        target=np.array([target],dtype=int)
        if len(self.centers_)==0:
            self.centers_=center
            self.targets_=target
            self.radii_=radius
        else:
            self.centers_=np.vstack( (self.centers_,center) )
            self.targets_=np.concatenate( (self.targets_,target) )
            self.radii_=np.concatenate( (self.radii_,radius) )
        
        
    def fit(self, X, y):
        X,y=check_X_y(X,y)
        # X, y = check_arrays(X, y, sparse_format="csr")
        # y = column_or_1d(y, warn=True)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        self.classes_ = classes
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError('y has fewer than 2 classes')

        if len(self.centers_)>0:
            assert len(self.centers_[0])==n_features

        radii=[]
        count=[]
        # first pass - only need the radii, because the vectors and the targets are already stored
        pass_number=0
        i=0
        for v,t in zip(X,y):  
            v=v.reshape(1, -1)

            D=pairwise_distances(v,X).ravel()
            r=max(D[y!=t].min()-1e-10,1e-10)
            radii.append(r)
    
            within=D[y==t]<=r
            count.append(within.sum())
    
            i+=1

        radii=np.array(radii)
        count=np.array(count)

        # second pass
        added=[]
        for v,t in zip(X,y): # Go through all of the data points
            #Select the sphere that contains that point, 
            # and the largest number of other points, 
            # and add it to the final spheres list
            v=v.reshape(1, -1)
            D=pairwise_distances(v,X).ravel()
            within_centers=(D<=radii)
            matched=(t==y) & (within_centers)
            idx=np.arange(len(y))
            idx_matched=idx[matched]
            best=idx_matched[np.argmax(count[matched])]
    
            if not best in added:
                self._add_center(X[best],radii[best],y[best])
                added.append(best)
        
            pass_number+=1
                
    def predict(self,X):
        X = check_array(X)        
        if len(self.centers_)==0:
            raise AttributeError("Model has not been trained yet.")
        
        result=[]
        for vector in X:
            vector=vector.reshape(1, -1)
            D=pairwise_distances(vector, self.centers_, metric=self.metric)/self.radii_
            result.append(self.targets_[D.argmin()])
            
        return np.array(result)
        
        

class CSC(CSCsk,GenericClassifier):

    def __init__(self, **kwargs):


        CSCsk.__init__(self, **kwargs)
        self.equivalent={'centers':'centers_',
                         'radii':'radii_',
                         'targets':'targets_'}
                         
        self.components=['centers_','radii_','targets_']
        self.as_array=self.components



        self.__dict__.update(self.equivalent)

    def fit(self,*args,**kwargs):
        CSCsk.fit(self,*args,**kwargs)
        for name in self.equivalent:
            super(CSC,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))

    def plot_centers(self):
        colors=utils.bold_colors
        for c,r,t in zip(self.centers_,self.radii_,self.targets_):
            pl.plot(c[0],c[1],'*',color=colors[t])
        angle=np.linspace(0,2*np.pi,100)  
        for c,r,t in zip(self.centers_,self.radii_,self.targets_):
            xd=np.cos(angle)*r + c[0]
            yd=np.sin(angle)*r + c[1]
            pl.plot(xd,yd,'-',color=colors[t])
            
        pl.axis('equal')
                
    def save(self,filename):
        D={}
        for key in self.components:
            D[key]=self.__getattribute__(key)

        
        with open(filename, 'w') as f:
            json.dump(D,f, sort_keys=True, indent=4,cls=NumpyAwareJSONEncoder)        

    def load(self,filename):

        with open(filename, 'r') as f:
            D=json.load(f)

        for key in self.components:
            val=D[key]

            if key in self.as_array:
                val=np.array(val)

            try:
                super(CSCsk,self).__setattr__(key,val)
            except AttributeError:
                print("Can't",key,val)
                raise

        for name in self.equivalent:
            super(CSCsk,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))




# from http://danielfrg.com/blog/2013/07/03/basic-neural-network-python/

from scipy import optimize
class NN_1HLsk(BaseEstimator, ClassifierMixin):
    
    def __init__(self, hidden_layer_size=25, reg_lambda=0, epsilon_init=0.12, opti_method='TNC', maxiter=500):
        self.reg_lambda = reg_lambda
        self.epsilon_init = epsilon_init
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = self.sigmoid
        self.activation_func_prime = self.sigmoid_prime
        self.method = opti_method
        self.maxiter = maxiter
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        sig = self.sigmoid(z)
        return sig * (1 - sig)
    
    def sumsqr(self, a):
        return np.sum(a ** 2)
    
    def rand_init(self, l_in, l_out):
        return np.random.rand(l_out, l_in + 1) * 2 * self.epsilon_init - self.epsilon_init
    
    def pack_thetas(self, t1, t2):
        return np.concatenate((t1.reshape(-1), t2.reshape(-1)))
    
    def unpack_thetas(self, thetas, input_layer_size, hidden_layer_size, num_labels):
        t1_start = 0
        t1_end = hidden_layer_size * (input_layer_size + 1)
        t1 = thetas[t1_start:t1_end].reshape((hidden_layer_size, input_layer_size + 1))
        t2 = thetas[t1_end:].reshape((num_labels, hidden_layer_size + 1))
        return t1, t2
    
    def _forward(self, X, t1, t2):
        m = X.shape[0]
        ones = None
        if len(X.shape) == 1:
            ones = np.array(1).reshape(1,)
        else:
            ones = np.ones(m).reshape(m,1)
        
        # Input layer
        a1 = np.hstack((ones, X))
        
        # Hidden Layer
        z2 = np.dot(t1, a1.T)
        a2 = self.activation_func(z2)
        a2 = np.hstack((ones, a2.T))
        
        # Output layer
        z3 = np.dot(t2, a2.T)
        a3 = self.activation_func(z3)
        return a1, z2, a2, z3, a3
    
    def function(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)
        
        m = X.shape[0]
        Y = np.eye(num_labels)[y]
        
        _, _, _, _, h = self._forward(X, t1, t2)
        costPositive = -Y * np.log(h).T
        costNegative = (1 - Y) * np.log(1 - h).T
        cost = costPositive - costNegative
        J = np.sum(cost) / m
        
        if reg_lambda != 0:
            t1f = t1[:, 1:]
            t2f = t2[:, 1:]
            reg = (self.reg_lambda / (2 * m)) * (self.sumsqr(t1f) + self.sumsqr(t2f))
            J = J + reg
        return J
        
    def function_prime(self, thetas, input_layer_size, hidden_layer_size, num_labels, X, y, reg_lambda):
        t1, t2 = self.unpack_thetas(thetas, input_layer_size, hidden_layer_size, num_labels)
        
        m = X.shape[0]
        t1f = t1[:, 1:]
        t2f = t2[:, 1:]
        Y = np.eye(num_labels)[y]
        
        Delta1, Delta2 = 0, 0
        for i, row in enumerate(X):
            a1, z2, a2, z3, a3 = self._forward(row, t1, t2)
            
            # Backprop
            d3 = a3 - Y[i, :].T
            d2 = np.dot(t2f.T, d3) * self.activation_func_prime(z2)
            
            Delta2 += np.dot(d3[np.newaxis].T, a2[np.newaxis])
            Delta1 += np.dot(d2[np.newaxis].T, a1[np.newaxis])
            
        Theta1_grad = (1 / m) * Delta1
        Theta2_grad = (1 / m) * Delta2
        
        if reg_lambda != 0:
            Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (reg_lambda / m) * t1f
            Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (reg_lambda / m) * t2f
        
        return self.pack_thetas(Theta1_grad, Theta2_grad)
    
    def fit(self, X, y):
        num_features = X.shape[0]
        input_layer_size = X.shape[1]
        num_labels = len(set(y))
        
        theta1_0 = self.rand_init(input_layer_size, self.hidden_layer_size)
        theta2_0 = self.rand_init(self.hidden_layer_size, num_labels)
        thetas0 = self.pack_thetas(theta1_0, theta2_0)
        
        options = {'maxiter': self.maxiter}
        _res = optimize.minimize(self.function, thetas0, jac=self.function_prime, method=self.method, 
                                 args=(input_layer_size, self.hidden_layer_size, num_labels, X, y, 0), options=options)
        
        self.t1, self.t2 = self.unpack_thetas(_res.x, input_layer_size, self.hidden_layer_size, num_labels)
    
    def predict(self, X):
        return self.predict_proba(X).argmax(0)
    
    def predict_proba(self, X):
        _, _, _, _, h = self._forward(X, self.t1, self.t2)
        return h                

class NN_1HL(NN_1HLsk,GenericClassifier):
    def __init__(self,N, **kwargs):
        NN_1HLsk.__init__(self,hidden_layer_size=N, **kwargs)
        self.equivalent={}
                         
        self.__dict__.update(self.equivalent)

    def fit(self,*args,**kwargs):
        NN_1HLsk.fit(self,*args,**kwargs)
        for name in self.equivalent:
            super(NN_1HL,self).__setattr__(name,
                self.__getattribute__(self.equivalent[name]))


