import numpy as np
import pylab as pl
import utils
from sklearn.utils import check_X_y,check_array
from multilayer_perceptron  import MultilayerPerceptronClassifier
from mlp import MLPClassifier
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.linear_model import SGDClassifier

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

class BackProp2(MLPClassifier,GenericClassifier):
    def __init__(self,**kwargs):
        if 'n_hidden' not in kwargs:
            raise ValueError,"Must specify n_hidden"
    
        MLPClassifier.__init__(self,**kwargs)
        self.equivalent={'weights_xh':'weights1_',
                        'weights_hy':'weights2_',
                        'bias_h':'bias1_',
                        'bias_y':'bias2_',
                        }
        self.shuffle_data=True
        self.max_epochs=50

    def fit(self,*args,**kwargs):
        if 'max_epochs' not in kwargs:
            max_epochs=self.max_epochs

        self.max_epochs=max_epochs
            
        MLPClassifier.fit(self,*args,
            max_epochs=self.max_epochs,
            shuffle_data=self.shuffle_data,
            **kwargs)
        for name in self.equivalent:
            super(MLPClassifier,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
    

    def output(self, X):
        n_samples = X.shape[0]
        x_hidden = np.empty((n_samples, self.n_hidden))
        x_output = np.empty((n_samples, self.n_outs))
        self._forward(None, X, slice(0, n_samples), x_hidden, x_output)
        return x_hidden,x_output

    
class BackProp(MultilayerPerceptronClassifier,GenericClassifier):
    def __init__(self,**kwargs):
        if 'n_hidden' not in kwargs:
            raise ValueError,"Must specify n_hidden"
        if 'tol' not in kwargs:
            kwargs['tol']=1e-7
    
        MultilayerPerceptronClassifier.__init__(self,**kwargs)
        self.equivalent={'weights_xh':'coef_hidden_',
                        'weights_hy':'coef_output_',
                        'bias_h':'intercept_hidden_',
                        'bias_y':'intercept_output_',
                        }
        
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
        X = check_array(X)

        a_hidden = self.activation_func(safe_sparse_dot(X, self.coef_hidden_) +
                                        self.intercept_hidden_)
        output = safe_sparse_dot(a_hidden, self.coef_output_) +\
            self.intercept_output_
        if output.shape[1] == 1:
            output = output.ravel()

        return a_hidden,output


from sklearn.neighbors import KNeighborsClassifier
class kNearestNeighbor(KNeighborsClassifier,GenericClassifier):
    
    def __init__(self,k=5):
        self.k=k        
        KNeighborsClassifier.__init__(self,n_neighbors=k)
       
        
from sklearn.naive_bayes import GaussianNB
class NaiveBayes(GaussianNB,GenericClassifier):

    def __init__(self):
        GaussianNB.__init__(self)
        self.equivalent={'means':'theta_',
                         'stddevs':'sigma_',
                         'fraction':'class_prior_'}
        #self.__dict__.update(self.equivalent)

    def fit(self,*args,**kwargs):
            
        GaussianNB.fit(self,*args,**kwargs)
        for name in self.equivalent:
            super(GaussianNB,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
    
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


from sklearn.linear_model import Perceptron as skPerceptron
class Perceptron(skPerceptron,GenericClassifier):

    def __init__(self,number_of_iterations=50):
        skPerceptron.__init__(self,shuffle=True,n_iter=number_of_iterations)
        
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
        self.centers_=np.array([],dtype=np.float)
        self.radii_=np.array([],dtype=np.float)
        self.targets_=np.array([],dtype=np.int)
        self.verbose=verbose
        
    def _add_center(self,center,radius,target):
        try:
            center=center.toarray()  # deal with sparse
        except AttributeError:
            pass
        
        center=check_array(center,dtype=np.float)
        radius=np.array([radius],dtype=np.float)
        target=np.array([target],dtype=np.int)
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
            print "%d clusters." % (len(self.centers_))

        # second pass
        stop=False
        while not stop:
            old_radii_=self.radii_.copy()
            for v,t in zip(X,y):  # Go through all of the data points (again) 
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
                print "%d clusters." % (len(self.centers_))
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
            D=pairwise_distances(vector, self.centers_, metric=self.metric)/self.radii_
            result.append(self.targets_[D.argmin()])
            
        return np.array(result)
        
        
class RCE(RCEsk,GenericClassifier):

    def __init__(self, **kwargs):


        RCEsk.__init__(self, **kwargs)
        self.equivalent={'centers':'centers_',
                         'radii':'radii_',
                         'targets':'targets_'}
                         
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
        
        
        
class CSCsk(BaseEstimator, ClassifierMixin):
    def __init__(self, metric='euclidean',r_step=1e-30,verbose=False):
        self.r_step=r_step
        self.metric = metric
        self.centers_=np.array([],dtype=np.float)
        self.radii_=np.array([],dtype=np.float)
        self.targets_=np.array([],dtype=np.int)
        self.verbose=verbose
        
    def _add_center(self,center,radius,target):
        try:
            center=center.toarray()  # deal with sparse
        except AttributeError:
            pass

        center=check_array(center,dtype=np.float)
        radius=np.array([radius],dtype=np.float)
        target=np.array([target],dtype=np.int)
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
            D=pairwise_distances(v,X).ravel()
            r=max(D[y!=t].min()-1e-10,1e-10)
            radii.append(r)
    
            within=D[y==t]<=r
            count.append(within.sum())
    
            i+=1

        radii=np.array(radii)
        count=np.array(count)

        # second pass
        for v,t in zip(X,y): # Go through all of the data points
            #Select the sphere that contains that point, 
            # and the largest number of other points, 
            # and add it to the final spheres list
        
            D=pairwise_distances(v,X).ravel()
            within_centers=(D<=radii)
            matched=(t==y) & (within_centers)
            idx=np.arange(len(y))
            idx_matched=idx[matched]
            best=idx_matched[np.argmax(count[matched])]
    
    
            self._add_center(X[best],radii[best],y[best])
        
            pass_number+=1
                
    def predict(self,X):
        X = check_array(X)        
        if len(self.centers_)==0:
            raise AttributeError("Model has not been trained yet.")
        
        result=[]
        for vector in X:
            D=pairwise_distances(vector, self.centers_, metric=self.metric)/self.radii_
            result.append(self.targets_[D.argmin()])
            
        return np.array(result)
        
        
class CSC(CSCsk,GenericClassifier):

    def __init__(self, **kwargs):


        CSCsk.__init__(self, **kwargs)
        self.equivalent={'centers':'centers_',
                         'radii':'radii_',
                         'targets':'targets_'}
                         
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
                