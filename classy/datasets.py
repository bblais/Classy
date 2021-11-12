from __future__ import print_function
import numpy as np
import sklearn.datasets
from .Struct import Struct
from sklearn.feature_extraction import DictVectorizer
from copy import deepcopy as copy_data

def make_dataset(**kwargs):
    from numpy import array
    
    feature_names=kwargs.pop('feature_names',None)
    
    vectors=[]
    targets=[]
    target_names=[]
    for k,key in enumerate(kwargs):
        for v in kwargs[key]:
            vectors.append(v)
            targets.append(k)
        target_names.append(key)
        
    if feature_names is None:
        feature_names=[str(_) for _ in range(len(v))]
        
    dataset=Struct(vectors=array(vectors),
                  targets=array(targets),
                  target_names=target_names,
                  feature_names=feature_names)
    
    return dataset

def remap_targets(dataset1,dataset2=None,new_target_names=None):

    if not dataset2 is None:
        new_dataset2=Struct()
        for key in dataset2:
            new_dataset2[key]=dataset2[key]

    new_dataset1=Struct()
    for key in dataset1:
        new_dataset1[key]=dataset1[key]
    
    if new_target_names is None:
        new_target_names=dataset1.target_names[:]
        
        if not dataset2 is None:
            for t in dataset2.target_names:
                if t not in new_target_names:
                    new_target_names.append(t)
                
    
    new_dataset1.target_names=new_target_names
    if not dataset2 is None:
        new_dataset2.target_names=new_target_names
    
    new_dataset1.targets=new_dataset1.targets.copy()
    if not dataset2 is None:
        new_dataset2.targets=new_dataset2.targets.copy()
    

    for i,t in enumerate(dataset1.targets):
        new_dataset1.targets[i]=new_target_names.index(dataset1.target_names[t])
        
    if not dataset2 is None:
        for i,t in enumerate(dataset2.targets):
            new_dataset2.targets[i]=new_target_names.index(dataset2.target_names[t])

    if not dataset2 is None:
        return new_dataset1,new_dataset2
    else:
        return new_dataset1


def load_csv(fname,max_lines=None,sparse=False,verbose=True):
    with open(fname) as fid:
        lines=fid.readlines()
    
    # first row are the labels
    r=0
    rowdata=lines[r].split(",")
    parts=[x.strip().strip('"').strip("'") for x in rowdata if x]
    
    if parts[-1].lower().startswith('categor') or parts[-1].lower().startswith('target'):
        with_target=True
        feature_names=[str(_) for _ in parts[:-1]]
    else:
        with_target=False
        feature_names=[str(_) for _ in parts]

    mapping=[]
    targets=[]
    
    count=0
    for r in lines[1:]:
        if not r:
            continue
        rowdata=r.split(",")
        parts=[x.strip().strip('"').strip("'") for x in rowdata if x]
    
        if with_target:
            strings=parts[0:-1]
        else:
            strings=parts

        row_mapping={}
        for s,f in zip(strings,feature_names):
            try:
                row_mapping[f]=float(s)
            except ValueError:
                row_mapping[f]=s
    
        mapping.append(row_mapping)

        if with_target:
            label=str(parts[-1])
            try:
                float_label=float(label)
                if (float_label-int(float_label))==0.0:
                    label=int(float_label)
            except ValueError:
                pass
            targets.append(label)

        count+=1
        if not max_lines is None and count>=max_lines:
            break


    vectorizer = DictVectorizer(sparse=sparse)
    vectors = vectorizer.fit_transform(mapping) 

    if with_target:
        target_names=sorted(list(set(targets)))
        target=[]
        for t in targets:
            target.append(target_names.index(t))
        target=np.array(target)
    else:
        target=None
        target_names=[]

    dataset=Struct(vectors=vectors,targets=target,
                target_names=target_names,feature_names=vectorizer.get_feature_names(),
                vectorizer=vectorizer)
    
    if verbose:
        summary(dataset)
    
    
    return dataset


def load_csv_orig(fname,max_lines=None,verbose=True):
    with open(fname) as fid:
        lines=fid.readlines()
        
    vectors=[]
    targets=[]

    # first row are the labels
    r=0
    rowdata=lines[r].split(",")
    parts=[x for x in rowdata if x]
    
    if parts[-1].lower().startswith('categor') or parts[-1].lower().startswith('target'):
        with_target=True
        feature_names=[str(_).strip('"').strip("'") for _ in parts[:-1]]
    else:
        with_target=False
        feature_names=[str(_).strip('"').strip("'") for _ in parts]
    
    count=0
    for r in lines[1:]:
        if not r:
            continue
        rowdata=r.split(",")
        parts=[x.strip().strip('"').strip("'") for x in rowdata if x]
    
        if with_target:
            strings=parts[0:-1]
        else:
            strings=parts

        vector=[]
        for k,x in enumerate(strings):
            try:
                val=float(x)
            except ValueError:
                val=np.nan
    
            vector.append(val)
    
        vectors.append(vector)

        if with_target:
            label=str(parts[-1]).strip()
            try:
                float_label=float(label)
                if (float_label-int(float_label))==0.0:
                    label=int(float_label)
            except ValueError:
                pass
            targets.append(label)
        
        
        count+=1
        if not max_lines is None and count>=max_lines:
            break
        
    if with_target:
        target_names=sorted(list(set(targets)))
        target=[]
        for t in targets:
            target.append(target_names.index(t))
        target=np.array(target)
    else:
        target=None
        target_names=[]
    
    
    vectors=np.array(vectors,dtype=np.float)
    
    dataset=Struct(vectors=vectors,targets=target,target_names=target_names,feature_names=feature_names)
    
    if verbose:
        summary(dataset)
    
    
    return dataset

def save_csv(fname,data):
    with open(fname,'w') as fid:
        if 'targets' in data and not data['targets'] is None:
            fid.write(','.join([str(_) for _ in data.feature_names]+["Target"]))
            fid.write('\n')
    
            for v,t in zip(data.vectors,data.targets):
                fid.write(','.join([str(_) for _ in v]+[str(data.target_names[t])]))
                fid.write('\n')
        else:
            fid.write(','.join([str(_) for _ in data.feature_names]))
            fid.write('\n')
    
            for v in data.vectors:
                fid.write(','.join([str(_) for _ in v]))
                fid.write('\n')

def load_excel(fname,max_lines=None,sparse=False,verbose=True,sheet=None):
    import xlrd
    book = xlrd.open_workbook(fname)
    
    if sheet is None:        
        sh = book.sheet_by_index(0)
    elif isinstance(sheet,int):
        sh = book.sheet_by_index(sheet)
    else:
        sh = book.sheet_by_name(sheet)

    if verbose:
        print(sh.name, sh.nrows, sh.ncols)

    mapping=[]
    targets=[]

    # first row are the labels
    r=0
    rowdata=sh.row(r)
    parts=[x.value for x in rowdata if x.value!='']
    
    if parts[-1].lower().startswith('categor') or parts[-1].lower().startswith('target'):
        with_target=True
        feature_names=[str(_) for _ in parts[:-1]]
    else:
        with_target=False
        feature_names=[str(_) for _ in parts]
    
    if not max_lines is None:
       max_lines+=1
    else:
        max_lines=sh.nrows 

    for r in range(1,max_lines):
        rowdata=sh.row(r)
        if not rowdata[0].ctype:  # empty cell
            continue
 
        parts=[x.value for x in rowdata if x.value!='']
        
        if with_target:
            strings=parts[0:-1]
        else:
            strings=parts

        row_mapping={}
        for s,f in zip(strings,feature_names):
            try:
                row_mapping[f]=float(s)
            except ValueError:
                row_mapping[f]=s
    
        mapping.append(row_mapping)

        if with_target:
            x=parts[-1]
            label=str(x)
            try:
                if int(x)==x:
                    label=int(x)
                else:
                    label=str(x)
            except ValueError:
                pass
            targets.append(label)
            
    if with_target:
        target_names=sorted(list(set(targets)))
        target=[]
        for t in targets:
            target.append(target_names.index(t))
        target=np.array(target)
    else:
        target=None
        target_names=[]

    vectorizer = DictVectorizer(sparse=sparse)
    vectors = vectorizer.fit_transform(mapping) 


    dataset=Struct(vectors=vectors,targets=target,
                target_names=target_names,feature_names=vectorizer.get_feature_names(),
                vectorizer=vectorizer)
    
    if verbose:
        summary(dataset)
    
    
    return dataset


def load_excel_orig(fname,max_lines=None,verbose=True,sheet=None):
    import xlrd
    book = xlrd.open_workbook(fname)
    
    if sheet is None:        
        sh = book.sheet_by_index(0)
    elif isinstance(sheet,int):
        sh = book.sheet_by_index(sheet)
    else:
        sh = book.sheet_by_name(sheet)

    if verbose:
        print(sh.name, sh.nrows, sh.ncols)
    
    vectors=[]
    targets=[]

    # first row are the labels
    r=0
    rowdata=sh.row(r)
    parts=[x.value for x in rowdata if x.value!='']
    
    if parts[-1].lower().startswith('categor') or parts[-1].lower().startswith('target'):
        with_target=True
        feature_names=[str(_) for _ in parts[:-1]]
    else:
        with_target=False
        feature_names=[str(_) for _ in parts]
    
    if not max_lines is None:
       max_lines+=1
    else:
        max_lines=sh.nrows 
    
    for r in range(1,max_lines):
        rowdata=sh.row(r)
        if not rowdata[0].ctype:  # empty cell
            continue
 
        parts=[x.value for x in rowdata if x.value!='']
        
        if with_target:
            strings=parts[0:-1]
        else:
            strings=parts

        vector=[]
        for k,x in enumerate(strings):
            try:
                val=float(x)
            except ValueError:
                val=np.nan
        
            vector.append(val)
        
        vectors.append(vector)
    
        if with_target:
            x=parts[-1]
            label=str(x)
            try:
                if int(x)==x:
                    label=int(x)
                else:
                    label=str(x)
            except ValueError:
                pass
            targets.append(label)
            
    if with_target:
        target_names=sorted(list(set(targets)))
        target=[]
        for t in targets:
            target.append(target_names.index(t))
        target=np.array(target)
    else:
        target=None
        target_names=[]
    
    vectors=np.array(vectors,dtype=np.float)
    
    dataset=Struct(vectors=vectors,targets=target,target_names=target_names,feature_names=feature_names)
    if verbose:
        summary(dataset)
    
    return dataset

def vectors_to_image(vectors,binary=False,axis='square'):
    from scipy.sparse import issparse
    import pylab as pl
    
    if issparse(vectors):
        vectors=vectors.todense()
        
    if binary:
        vectors=vectors>0
        
        
    if axis=='square':
        shape=vectors.shape
        aspect=shape[1]/shape[0]
    elif axis=='equal':
        aspect=1
    else:
        raise ValueError("Unknown axis option %s" % str(axis))

    pl.imshow(vectors,cmap=pl.cm.gray,aspect=aspect,interpolation='nearest')
    
    

    
def summary(data):
    from scipy.sparse import issparse
    
    if "DESCR" in data and data.DESCR=="Images":
        print("Images")
        print("%d images of shape %s" % (len(data.data),str(data.data[0].shape)))
    
        try:
            if not data.targets is None:
                data.targets
                print("Target values given.")
            else:
                print("No Target values.")
        except (KeyError,AttributeError):
                print("No Target values.")
                
        try:            
            print("Target names:", end=' ')
            if data.target_names:
                print(", ".join(["'%s'" % name for name in data.target_names]))
            else:
                print("[None]")

            for i,name in enumerate(data.target_names):
                L=len(data.targets[data.targets==i])
                print("[%s]: %d files" % (name,L))



        except (KeyError,AttributeError):
            pass
    elif "DESCR" in data and data.DESCR=="Sequences":
        print("Sequences")
        print("%d sequences of median length %d" % (len(data.data),
                                    np.median([len(_) for _ in data.data])))
                                    
        print("Unique letters:",np.unique(''.join(data.data)))
        try:            
            print("Target names:", end=' ')
            if data.target_names:
                print(", ".join(["'%s'" % name for name in data.target_names]))
            else:
                print("[None]")
        except (KeyError,AttributeError):
            pass
        
    else:

        print("%d vectors of length %d" % (data.vectors.shape[0],data.vectors.shape[1]))
        try:
            print("Feature names:", end=' ')
            
            if len(data.feature_names)<15:            
                print(", ".join(["'%s'" % name for name in data.feature_names]))
            else:
                print(", ".join(["'%s'" % name for name in data.feature_names[:5]]), end=' ')
                print(" , ... , ", end=' ')
                print(", ".join(["'%s'" % name for name in data.feature_names[-5:]]), end=' ')
                print(" (%d features)" % (len(data.feature_names)))
                
        except (KeyError,AttributeError):
            pass
        
        try:
            if not data.targets is None:
                data.targets
                print("Target values given.")
            else:
                print("No Target values.")
        except (KeyError,AttributeError):
            pass
        try:
            print("Target names:", end=' ')
            if data.target_names:
                print(", ".join(["'%s'" % name for name in data.target_names]))
            else:
                print("[None]")
        except (KeyError,AttributeError):
            pass

        print("Mean: ",data.vectors.mean(axis=0))
        if not issparse(data.vectors):
            print("Median: ",np.median(data.vectors,axis=0))
            print("Stddev: ",np.std(data.vectors,axis=0))        
        
        
    
def save_ssdf(fname,data):
    from . import ssdf
    from .Struct import Struct

    data=Struct(data)
    
    if 'vectors' in data:
        data.vectors=data.vectors.tolist()
    if 'targets' in data and not data['targets'] is None:
        data.targets=data.targets.tolist()
    else:
        data.targets=None
        
    if 'data' in data:
        data.data=data.data.tolist()
    
    ssdf.save(fname,dict(data))
    
def load_ssdf(fname,verbose=True):
    from . import ssdf
    from .Struct import Struct
    s=ssdf.load(fname)
    data=Struct(s.__dict__)
    data.targets=np.array(data.targets)
    
    if 'vectors' in data:
        data.vectors=np.array(data.vectors)
        
    if 'data' in data:
        data.data=np.array(data.data)
    
    if verbose:
        summary(data)
    return data    
    
def split(data,test_size=None,train_size=None,shuffle=False,verbose=True):
    from sklearn.model_selection import train_test_split
    from copy import deepcopy

    d1=Struct(split=True)
    d2=Struct(split=True)    
    for name in data:
        if name=='targets' or name=='vectors':
            pass
        d1[name]=data[name]
        d2[name]=data[name]
    
    if 'targets' in data and not data['targets'] is None:
        v1,v2,t1,t2=train_test_split(data.vectors,data.targets,test_size=test_size,train_size=train_size)    
        d1.vectors=v1
        d2.vectors=v2
        d1.targets=t1
        d2.targets=t2
    else:
        v1,v2=train_test_split(data.vectors,test_size=test_size,train_size=train_size)
        d1.vectors=v1
        d2.vectors=v2

    if verbose:
        print("Original vector shape: ",data.vectors.shape)
        print("Train vector shape: ",d1.vectors.shape)
        print("Test vector shape: ",d2.vectors.shape)

    return d1,d2        
        
def random_vector(data,targets=None):
    from random import choice
    from numpy import where
    if targets is None:
        return choice(data.vectors)
    
    if isinstance(targets,str):
        targets=[targets]
        
    idx=[]
    for name in targets:
        t=data.target_names.index(name)
        idx.extend(where(data.targets==t)[0])

    i=choice(idx)
    return data.vectors[i]

def extract_features(data,idx):
    from copy import deepcopy
    d1=Struct(extract=True)
    for name in data:
        if name=='vectors':
            pass
        d1[name]=data[name]
    
    if 'feature_names' in data:
        if idx[0] in data.feature_names:  # given strings
            idx=[data.feature_names.index(_) for _ in idx]

        d1.feature_names=[data.feature_names[_i] for _i in idx]
    
    d1.vectors=data.vectors[:,idx]
    
    return d1
    
def extract_vectors(data,idx):
    from copy import deepcopy
    d1=Struct(extract=True)
    for name in data:
        if name=='targets' or name=='vectors':
            pass
        d1[name]=data[name]
    
    d1.vectors=data.vectors[idx,:]
    if 'targets' in data and not data['targets'] is None:
        d1.targets=data.targets[idx]
        
    if 'files' in data and not data['files'] is None:
        d1.files=data.files[idx]
    
    return d1    

def randbetween(low,high,N):
    return np.random.rand(N)*(high-low)+low

def double_moon_data(d=1,w=1,r=3,N=1000):
    NA=N//2
    NB=N-NA

    θ=randbetween(0,180,NA)
    R=randbetween(r-w/2,r+w/2,NA)

    x=R*np.cos(np.radians(θ))
    y=R*np.sin(np.radians(θ))

    A_vectors=np.stack((x,y)).T


    θ=randbetween(180,360,NA)
    R=randbetween(r-w/2,r+w/2,NA)

    x=R*np.cos(np.radians(θ))+r
    y=R*np.sin(np.radians(θ))-d

    B_vectors=np.stack((x,y)).T

    data=make_dataset(Alice=A_vectors,Bob=B_vectors)    

    return data    