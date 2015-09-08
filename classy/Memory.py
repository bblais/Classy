"""Convenient wrapper around ssdf"""

import ssdf
import gzip
import os
from numpy import array,iterable

def Save(object, filename='_memory_.ssdf'):
    """Saves an object to disk
    
    Example:  Save([1,2,3])
    """
    s=ssdf.new()
    s.object=object
    ssdf.save(filename,s)   

def Struct2Dict(S):

    if isinstance(S,ssdf.Struct):
        d={}
        for key in S:
            if isinstance(key,unicode):  # get rid of unicode!
                key2=str(key)
            else:
                key2=key
        
            d[key2]=Struct2Dict(S[key])
        return d
    elif isinstance(S,list):
        d=[Struct2Dict(x) for x in S]
        return d
    elif isinstance(S,tuple):
        d=[Struct2Dict(x) for x in S]
        return tuple(d)
    else:
        return S

def Load(filename='_memory_.ssdf'):
    """Loads an object from disk

    Example:  a=Load()
    """
    
    s=ssdf.load(filename)
    object=s.object
    
    obj=Struct2Dict(object)
    return obj

def Remember(*args,**kwargs):

    try:
        filename=kwargs['filename']
    except KeyError:
        filename='_memory_.ssdf'

    if len(args)>0:
        Save(args,filename)
        return

    Q=Load(filename)
    if len(Q)==1:
        Q=Q[0]
        
    return Q
    
data={}
data['default']=[]
def reset(name=None):
    global data
    
    if name==None:
        data={}
        data['default']=[]
    else:
        data[name]=[]
    
def store(*args,**kwargs):
    global data
    
    if 'name' in kwargs:
        name=kwargs['name']
    else:
        name='default'
    
    if name not in data:
        data[name]=[]
        
    if not args:
        data[name]=[]
    
    if not data[name]:
        for arg in args:
            data[name].append([arg])
            
    else:
        for d,a in zip(data[name],args):
            d.append(a)
    

def recall(name='default'):
    global data
    
    if name not in data:
        data[name]=[]
    
    for i in range(len(data[name])):
        data[name][i]=array(data[name][i])
    
    ret=tuple(data[name])
    if len(ret)==1:
        return ret[0]
    else:
        return ret




if __name__ == "__main__":
    import sys
    import os.path
    
    class Object:
        x = 7
        y = "This is an object."
    
    filename = sys.argv[1]
    if os.path.isfile(filename):
        o = load(filename)
        print "Loaded %s" % o
    else:
        o = Object()
        save(o, filename)
        print "Saved %s" % o

