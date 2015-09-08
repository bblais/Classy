import classy.datasets
from Struct import Struct
import numpy as np
from numpy import sqrt,sum,exp,pi,min,max,linspace

def normal(x,mu,sd):
    return 1.0/sqrt(2*pi*sd**2)*exp(-(x-mu)**2/(2*sd**2))

def overlap(means_,covars_):
    # http://en.wikipedia.org/wiki/Bhattacharyya_distance
    # overlap is a dot product

    s1,s2=covars_
    m1,m2=means_

    minx=min([m1-4*s1,m2-4*s2])
    maxx=min([m1+4*s1,m2+4*s2])
    
    x=linspace(minx,maxx,1000)
    dx=x[1]-x[0]
    
    BC=sum(dx*sqrt(normal(x,m1,s1)*normal(x,m2,s2)))
    
    
    return BC

def GMM_features_from_1D_vectors2(origdata,number_of_gaussians_list,verbose=True):
    from sklearn.mixture import GMM
    
    data=Struct(origdata)
    data.vectors=[]
    
    data.feature_names=[]
    for M in number_of_gaussians_list:
        for G in range(M):
            data.feature_names+=['M%d mu%d' % (M,G+1),'M%d sd%d' % (M,G+1)]
    
    for X in origdata.vectors:
        vec=[]
        for M in number_of_gaussians_list:
            model = GMM(M).fit(X)
            means=model.means_.ravel()
            stddevs=model.covars_.ravel()

            for m,s in zip(means,stddevs):
                vec.append(m)
                vec.append(s)
            
        data.vectors.append(vec)

    data.vectors=np.array(data.vectors)

    if verbose:
        classy.datasets.summary(data)
        
    return data
    

def GMM_features_from_1D_vectors(origdata,number_of_gaussians,verbose=True):
    from sklearn.mixture import GMM
    
    data=Struct(origdata)
    data.vectors=[]
    
    data.feature_names=[]
    for i in range(number_of_gaussians):
        data.feature_names+=['mu%d' % (i+1),'sd%d' % (i+1)]
    L=number_of_gaussians
    for i in range(L):
        for j in range(i+1,L):
            data.feature_names+=['overlap %d-%d' % (i+1,j+1)]

    for X in origdata.vectors:
        model = GMM(number_of_gaussians).fit(X)
        means=model.means_.ravel()
        stddevs=model.covars_.ravel()
        vec=[]
        for m,s in zip(means,stddevs):
            vec.append(m)
            vec.append(s)

        L=number_of_gaussians
        for i in range(L):
            for j in range(i+1,L):
                vec.append(overlap([means[i],means[j]],[stddevs[i],stddevs[j]]))
            
        data.vectors.append(vec)

        
    data.vectors=np.array(data.vectors)
        
    if verbose:
        classy.datasets.summary(data)
        
    return data
        
  