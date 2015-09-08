#from sklearn.utils.validation import check_arrays, atleast2d_or_csr, column_or_1d
from sklearn.utils.extmath import safe_sparse_dot
import autoencoder
from sklearn.utils import check_X_y
from autoencoder import Autoencoder as AutoEncoder
from sklearn.decomposition import RandomizedPCA
from numpy import array
from datasets import Struct
from copy import deepcopy

class GenericFilter(object):

    def fit_transform_data(self,data):
        new_data=Struct()
        new_data.DESCR="Transformed"
        
        for key in data:
            if key=='vectors' or key=='feature_names':
                continue
            new_data[key]=deepcopy(data[key])
            
        new_data.vectors=self.fit_transform(data.vectors)
        new_data.feature_names=['F%d' % (f+1) for f in range(new_data.vectors.shape[1])]
        
        return new_data


    def transform_data(self,data):
        new_data=Struct()
        new_data.DESCR="Transformed"
        
        for key in data:
            if key=='vectors' or key=='feature_names':
                continue
            new_data[key]=deepcopy(data[key])
            
            
        new_data.vectors=self.transform(data.vectors)
        new_data.feature_names=['F%d' % (f+1) for f in range(new_data.vectors.shape[1])]
        
        return new_data


class AutoEncoder(autoencoder.Autoencoder,GenericFilter):

    def __init__(self,*args,**kwargs):    
        autoencoder.Autoencoder.__init__(self,*args,**kwargs)
        self.equivalent={'weights_xh':'coef_hidden_',
                        'weights_hy':'coef_output_',
                    }
    def fit(self,*args,**kwargs):
        autoencoder.Autoencoder.fit(self,*args,**kwargs)
        for name in self.equivalent:
            super(autoencoder.Autoencoder,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
                        
    def fit_transform(self,*args,**kwargs):
        result=autoencoder.Autoencoder.fit_transform(self,*args,**kwargs)
        for name in self.equivalent:
            super(autoencoder.Autoencoder,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))

        return result

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
        X = atleast2d_or_csr(X)

        a_hidden = self.activation_func(safe_sparse_dot(X, self.coef_hidden_) +
                                        self.intercept_hidden_)
        output = safe_sparse_dot(a_hidden, self.coef_output_) +\
            self.intercept_output_
        if output.shape[1] == 1:
            output = output.ravel()

        return a_hidden,output

    def plot(self,only=None):
        from pylab import plot,subplot,sqrt,ceil,title
    
        weights=self.weights_xh.T
        
        if only is None:
            only=range(len(weights))
    
        L=len(only)
        c=ceil(sqrt(L))
        r=ceil(L/c)


        for i,idx in enumerate(only):
            w=weights[idx]
            subplot(r,c,i+1)
            plot(w,'-o')
            title('Filter %d' % (idx))
            
    def imshow(self,shape,only=None):
        from pylab import subplot,imshow,cm,title,sqrt,ceil

        weights=self.weights_xh.T
        
        if only is None:
            only=range(len(weights))
    
        L=len(only)
        c=ceil(sqrt(L))
        r=ceil(L/c)
    
        for i,idx in enumerate(only):
            w=weights[idx]
            w=w.reshape(shape)
            subplot(r,c,i+1)
            imshow(w,cmap=cm.gray,interpolation='nearest')
            title('Filter %d' % (idx))
            
            
from sklearn.mixture import GMM
            
class GMM1D(GenericFilter):
    def __init__(self,number_of_gaussians):
        self.number_of_gaussians=number_of_gaussians
        if isinstance(number_of_gaussians,int):
            self.number_of_gaussians=[number_of_gaussians]
            
            
    def fit(self,X):
        pass
        
    def transform(self,X):
        newX=[]
        models=[GMM(M) for M in self.number_of_gaussians]
        
        for v in X:
            vec=[]
            for model in models:
                model.fit(v)
                means=model.means_.ravel()
                stddevs=model.covars_.ravel()

                for m,s in zip(means,stddevs):
                    vec.append(m)
                    vec.append(s)
            
            newX.append(vec)
        
        newX=array(newX)
        
        return newX
        
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)        

class PCA(RandomizedPCA,GenericFilter):
    def __init__(self,*args,**kwargs):    
        RandomizedPCA.__init__(self,*args,**kwargs)
        self.equivalent={'weights':'components_',
                        'components':'components_',
                        }
                        
                        
    def fit(self,*args,**kwargs):
        RandomizedPCA.fit(self,*args,**kwargs)
        for name in self.equivalent:
            super(RandomizedPCA,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))
                        
    def fit_transform(self,*args,**kwargs):
        result=RandomizedPCA.fit_transform(self,*args,**kwargs)
        for name in self.equivalent:
            super(RandomizedPCA,self).__setattr__(name,self.__getattribute__(self.equivalent[name]))

        return result
        
        
    def plot(self,only=None):
        from pylab import plot,subplot,sqrt,ceil,title
    
    
    
        if only is None:
            only=range(len(self.weights))
    
        L=len(only)
        c=ceil(sqrt(L))
        r=ceil(L/c)


        for i,idx in enumerate(only):
            w=self.weights[idx]
            subplot(r,c,i+1)
            plot(w,'-o')
            title('PC %d' % (idx))
            
    def imshow(self,shape,only=None):
        from pylab import subplot,imshow,cm,title,sqrt,ceil
        if only is None:
            only=range(len(self.weights))
    
        L=len(only)
        c=ceil(sqrt(L))
        r=ceil(L/c)
    
        for i,idx in enumerate(only):
            w=self.weights[idx]
            w=w.reshape(shape)
            subplot(r,c,i+1)
            imshow(w,cmap=cm.gray,interpolation='nearest')
            title('PC %d' % (idx))
            