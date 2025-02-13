
import numpy as np
import pylab as pl
from . import utils
from sklearn.utils import check_X_y,check_array
from sklearn.neural_network import MLPClassifier as MultilayerPerceptronClassifier
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import SGDClassifier

#from .supervised_numpynet import *
from .supervised_jax import *

from functools import wraps

def reshape_vectors(func):
    @wraps(func)
    def wrapper(self,*args, **kwargs):
        # Reshape the first argument (assume it's a NumPy array)
        original_shape = args[0].shape
        reshaped_vectors = args[0].reshape(original_shape[0], -1)
        
        # Call the original function with the reshaped array
        new_args = (reshaped_vectors, *args[1:])
        result = func(self,*new_args, **kwargs)

        return result
    
    return wrapper


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
    
    @reshape_vectors
    def percent_correct(self,vectors,targets):
        return self.score(vectors,targets)*100.0

    @reshape_vectors
    def predict_names(self,vectors,names):
        result=self.predict(vectors)
        return [names[i] for i in result]
    def confusion_matrix(self,data_test):
        from pandas import DataFrame
        predictions=self.predict(data_test.vectors)
        confusion_mat=np.zeros((len(data_test.target_names),len(data_test.target_names)),int)
        for predict_i,n0 in enumerate(data_test.target_names):
            for true_i,n1 in enumerate(data_test.target_names):
                confusion_mat[predict_i,true_i]=int(np.sum((data_test.targets==true_i) & (predictions==predict_i)))        
        df=DataFrame(data=confusion_mat,
                        columns=[f'True {_}' for _ in data_test.target_names],
                        index=[f'Predicted {_}' for _ in data_test.target_names])

        return df
    

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
        self.components=['classes_', 'k','_fit_X',
                        '_fit_method',
                        #'_tree',
                        '_y',
                        'classes_',
                        'effective_metric_',
                        'n_samples_fit_',
                        'effective_metric_params_',
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

    @reshape_vectors
    def fit(self,*args,**kwargs):
        KNeighborsClassifier.fit(self,*args,**kwargs)

from sklearn.naive_bayes import GaussianNB
class NaiveBayes(GaussianNB,GenericClassifier):

    def __init__(self):
        GaussianNB.__init__(self)
        self.var_smoothing=1e-2  # make it much more stable
        self.equivalent={'means':'theta_',
                         'stddevs':'var_',
                         'fraction':'class_prior_'}

        self.components=['class_count_','class_prior_','classes_',
        'var_','theta_','epsilon_',]

        #self.__dict__.update(self.equivalent)

    @reshape_vectors
    def fit(self,*args,**kwargs):
            
        vectors=args[0]
        shape=vectors.shape
        vectors = vectors.reshape(vectors.shape[0], -1)
        GaussianNB.fit(self,*args,**kwargs)
        vectors = vectors.reshape(shape)
        try:
            v=self.__getattribute__('var_')
        except AttributeError:  # changing the names of features hack
            self.var_=self.sigma_

        for name in self.equivalent:
            super(GaussianNB,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
    
        self.stddevs=np.sqrt(self.stddevs)

    @reshape_vectors
    def anotherfit(self, X, y):
        X,y=check_X_y(X,y)
            
        GaussianNB.fit(self,X,y)
    
        for name in self.equivalent:
            super(GaussianNB,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
    
    @reshape_vectors
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
            super(GaussianNB,self).__setattr__(key,val)

        self.apply_translate()

        for name in self.equivalent:
            super(GaussianNB,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
        self.stddevs=np.sqrt(self.stddevs)
    
    def apply_translate(self):
        translate=[ ('var_','sigma_'),
                    ]                  

        for key1,key2 in translate:
            if key1 in self.__dict__:
                self.__dict__[key2]=self.__dict__[key1]
            elif key2 in self.__dict__:
                self.__dict__[key1]=self.__dict__[key2]
            else:
                raise AttributeError((key1,key2))

    def __getattr__(self, item):
        translate={'var_':'sigma_',
                  }

        if item in translate:
            value=self.__dict__[translate[item]]
            self.__dict__[item]=value
        else:
            raise AttributeError(item)
        
        
        return self[item]

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
        
    @reshape_vectors
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
                
                
    @reshape_vectors
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

    @reshape_vectors
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
        
        
    @reshape_vectors
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
                
    @reshape_vectors
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

    @reshape_vectors
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



