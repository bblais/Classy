import classy.datasets
from Struct import Struct
import os
import glob
import numpy as np

def load_sequences(fname,sheet=None,verbose=True):
    import xlrd

    base,ext=os.path.splitext(fname)
    data=Struct()
    data.DESCR="Sequences"
    
    data.data=[]
    data.targets=[]
    
    if ext=='.xls' or ext=='.xlsx':
        book = xlrd.open_workbook(fname)
    
        if sheet is None:        
            sh = book.sheet_by_index(0)
        elif isinstance(sheet,int):
            sh = book.sheet_by_index(sheet)
        else:
            sh = book.sheet_by_name(sheet)
            
        r=0
        rowdata=sh.row(r)
        parts=[x.value for x in rowdata if x.value!='']
        if parts[-1].lower().startswith('categor') or parts[-1].lower().startswith('target'):
            with_target=True
            if verbose:
                print "Target Column Found"
            
            assert len(parts)==2
        else:
            if verbose:
                print "Target Column Not Found"
            with_target=False
            assert len(parts)==1
            
        targets=[]
        for r in range(1,sh.nrows):
            rowdata=sh.row(r)
            if not rowdata[0].ctype:  # empty cell
                continue
        
            sequence=str(rowdata[0].value)
            data.data.append(sequence)
            
            if with_target:
                x=rowdata[1].value
                if int(x)==x:
                    target=int(x)
                else:
                    target=str(x)
                
                targets.append(target)
            else:
                target=None
    else:
        raise ValueError,"%s file not implemented" % ext
        
    if with_target:
        target_names=sorted(list(set(targets)))
        target=[]
        for t in targets:
            target.append(target_names.index(t))
        data.targets=np.array(target)
        data.target_names=target_names
    else:
        target_names=[]
        data.targets=None
        data.target_names=target_names

    data.letters=''.join(np.unique(''.join(data.data)))

    if verbose:
        classy.datasets.summary(data)
        
    return data
    
def vector_to_sequence(vector,letters):
    sequence_length=len(vector)/len(letters)
    
    seq=''
    for s in range(sequence_length):
        subvec=vector[(s*len(letters)):((s+1)*len(letters))]
        seq+=letters[subvec.argmax()]
        
    return seq
    
    
def chunk_to_vector(chunk,letters):
    subvectors=[]
    for c in chunk:
        v=np.zeros(len(letters))
        v[letters.index(c)]=1.0
        subvectors.append(v)
        
    vector=np.concatenate(subvectors)
    return vector
    
def sequences_to_vectors(origdata,testdata=None,chunksize=5,verbose=True):

    if testdata is None:

        data=Struct()
        data.target_names=origdata.target_names
        data.vectors=[]
        data.targets=[]
        data.shape=(chunksize,)
        data.letters=origdata.letters
        
        for seq,t in zip(origdata.data,origdata.targets):
    
            for i in range(len(seq)-chunksize+1):
                chunk=seq[i:(i+chunksize)]
                data.vectors.append(chunk_to_vector(chunk,data.letters))
                data.targets.append(t)
          
          
        data.vectors=np.array(data.vectors,dtype=np.float)
        data.targets=np.array(data.targets)
      
        data.feature_names=[]
        for i in range(chunksize):
            for c in data.letters:
                data.feature_names.append('%c%d' % (c,i))
            
        if verbose:
            classy.datasets.summary(data)
            
        return data
    else:
    
        data1,data2=classy.datasets.remap_targets(origdata,testdata)
        data1.shape=(chunksize,)
        data1.letters=''.join(np.unique(origdata.letters+testdata.letters))
        data2.shape=data1.shape
        data2.letters=data2.letters
        
        data=Struct()
        data.target_names=data1.target_names
        data.vectors=[]
        data.targets=[]
        data.shape=(chunksize,)
        data.letters=data1.letters
        for seq,t in zip(data1.data,data1.targets):
    
            for i in range(len(seq)-chunksize+1):
                chunk=seq[i:(i+chunksize)]
                data.vectors.append(chunk_to_vector(chunk,data.letters))
                data.targets.append(t)
          
          
        data1.vectors=np.array(data.vectors,dtype=np.float)
        data1.targets=np.array(data.targets)
      
        data.feature_names=[]
        for i in range(chunksize):
            for c in data.letters:
                data.feature_names.append('%c%d' % (c,i))
        data1.feature_names=data.feature_names
        
        data=Struct()
        data.target_names=data2.target_names
        data.vectors=[]
        data.targets=[]
        data.shape=(chunksize,)
        data.letters=data2.letters
        for seq,t in zip(data2.data,data2.targets):
    
            for i in range(len(seq)-chunksize+1):
                chunk=seq[i:(i+chunksize)]
                data.vectors.append(chunk_to_vector(chunk,data.letters))
                data.targets.append(t)
          
          
        data2.vectors=np.array(data.vectors,dtype=np.float)
        data2.targets=np.array(data.targets)
      
        data.feature_names=[]
        for i in range(chunksize):
            for c in data.letters:
                data.feature_names.append('%c%d' % (c,i))
        data2.feature_names=data.feature_names       
        
        data1.DESCR="Sequence Vectors"
        data2.DESCR="Sequence Vectors"
        
        if verbose:
            classy.datasets.summary(data1)
            print "==="
            classy.datasets.summary(data2)
        
        
        return data1,data2
        
        
        
        
        
        
        
        
        