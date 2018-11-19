from PIL import Image
from .Struct import Struct
import os
import glob
import numpy as np
import classy.datasets
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def split(images,test_size=0.2,verbose=True):
    from numpy import where,array
    from random import shuffle
    
    d1=Struct(split=True)
    d2=Struct(split=True)  
    
    
    skip_names=['files','data','targets']
    for name in images:
        if name in skip_names:
            pass
        d1[name]=images[name]
        d2[name]=images[name]
        
    
    num_targets=len(images.target_names)
    d1.targets=[]
    d2.targets=[]
    
    d1.data=[]
    d2.data=[]
    
    d1.files=[]
    d2.files=[]
    
    for k in range(num_targets):
        idx=where(images.targets==k)[0]
        
        N=len(idx)
        
        if test_size<1: # fraction
            N_test=int(test_size*N)+1
        else:
            N_test=test_size
            
        N_train=N-N_test
        shuffle(idx)
        
        for i in idx[:N_test]:
            d1.targets.append(images.targets[i])
            d1.files.append(images.files[i])
            d1.data.append(images.data[i])
        for i in idx[N_test:]:
            d2.targets.append(images.targets[i])
            d2.files.append(images.files[i])
            d2.data.append(images.data[i])
            
    d1.targets=array(d1.targets,dtype=np.int32)
    d2.targets=array(d2.targets,dtype=np.int32)
    
    if verbose:
        print("Files in Test Set:")
        print("\t",','.join(d1.files))
        print("Files in Train Set:")
        print("\t",','.join(d2.files))
        
    return d1,d2

def show_images(images,which_images=None,max_images=None):
    from pylab import imshow,subplot,sqrt,ceil,title,cm,gca
    from random import shuffle
    
    if which_images is None:
        which_images=list(range(len(images.data)))
        
    if isinstance(which_images[0],str):  # target names
        which_names=which_images
        which_images=[]
        for idx in range(len(images.data)):
            name=images.target_names[images.targets[idx]]
            if name in which_names:
                which_images.append(idx)
        
    if not max_images is None:
        shuffle(which_images)
        which_images=which_images[:max_images]
    
        
        
    if not which_images:
        raise ValueError("No images selected")
        
    L=len(which_images)
    c=ceil(sqrt(L))
    r=ceil(L/c)

    for i,idx in enumerate(which_images):
        im=images.data[idx]
        name=images.target_names[images.targets[idx]]
        subplot(r,c,i+1)
        imshow(im,interpolation='nearest',cmap=cm.gray)
        title(name)
        
        if i<(L-c):
            gca().set_xticklabels([])
            
        if i%c!=0:
            gca().set_yticklabels([])
        

def show_image_vector(vector,shape):
    from matplotlib.pyplot import imshow
    from matplotlib.cm import gray
    
    im=vector.reshape(shape)
    imshow(im,interpolation='nearest',cmap=gray)

def vector_to_image(vector,shape,fname=None):
    from PIL import Image
    from matplotlib.pyplot import imshow
    from matplotlib.cm import gray
    
    arr=vector.reshape(shape)
    
    if fname is None:    
        imshow(arr,interpolation='nearest',cmap=gray)
    else:
        if arr.max()>255:
            arr=np.array(arr,dtype=np.uint16)
        elif arr.max()>1:
            arr=np.array(arr,dtype=np.uint8)
        else:
            arr=np.array(arr*2**16,dtype=np.uint16)
            
        im=Image.fromarray(arr)
        im.save(fname)

def array_to_image_struct(arr):
    if isinstance(arr,list):
        N=len(arr)
        data=Struct()
        data.DESCR="Images"
        data.files=[None]*N
        data.data=arr
        data.targets=[0]*N
        data.target_names=['None']*N
        
        
    else:
        data=Struct()
        data.DESCR="Images"
        data.files=[None]
        data.data=[arr]
        data.targets=[0]
        data.target_names=['None']

    return data


def load_images_from_filepatterns(**kwargs):
    from glob import glob
    data=Struct()
    data.DESCR="Images"
    data.files=[]
    data.data=[]
    data.targets=[]
    data.target_names=[]
    
    filenames={}
    verbose=None
    for key in sorted(kwargs):
        if key=='verbose':
            verbose=kwargs[key]
            continue
            
        if isinstance(kwargs[key],str):
            fnames=glob(kwargs[key])
        else:
            fnames=kwargs[key]
            
        if not fnames:
            continue
            
        data.target_names.append(key)
        filenames[key]=fnames
        
    if verbose is None:
        verbose=True
        
    if not data.target_names:
        print("No images matching the patterns found.")
        return None
        
    for i,name in enumerate(data.target_names):
        values=filenames[name]
        if verbose:
            print("[%s]: %d files found" % (name,len(values)))
            for v in values:
                print("\t%s" % v)
        
        data.files.extend(values)
        data.targets.extend([i]*len(values))
    data.targets=np.array(data.targets,dtype=np.int32)
    
    for fname in data.files:
        im=Image.open(fname)
        if im.mode=='1' or im.mode=='LA':
            im=im.convert('L')
        ima=np.asarray(im)

        if len(ima.shape)==3:
            ima=ima[:,:,:3]  # take out the alpha channel if it exists

        data.data.append(ima)
        
    return data


def process_images(filter,newdirname='.',resize=None,colormode=None,ext=None):
    # filter='caltech101/*/*.jpg'

    # newdirname='blah'
    # resize=(300,200)
    # colormode='color'
    # ext='.png'


    mode={'color':'RGB','gray':'L','bw':'1'}
    revmode={'RGB':'color','L':'gray','1':'bw'}

    files=glob.glob(filter)
    files=[_ for _ in files if 'desktop.ini' not in _]
          
    im2=None
    for fname in files:
        if os.path.isdir(fname):
            continue

        try:
            im=Image.open(fname)
            if im.mode=='LA':
                im=im.convert('L')
            if im.mode=='RGBA':
                im=im.convert('RGB')

        except IOError:
            print("failed to open %s" % fname)
            
        im_orig=im
        
        if not resize is None:
            im=im.resize(resize)
            
        if not colormode is None:
            im=im.convert(mode[colormode])
            
        if im is im_orig:
            print("%s: size %s mode %s" % (fname,im.size,revmode.get(im.mode,im.mode)))
        else:
            newfname=os.path.join(newdirname,fname)
            if not ext is None:
                ext=ext.replace(".","")
                ext="."+ext
                newfname,oldext=os.path.splitext(newfname)
                newfname=newfname+ext
            print("%s: size %s mode %s -> %s: size %s mode %s" % (fname,
                                                  im_orig.size,
                                                  revmode.get(im_orig.mode,im_orig.mode),
                                                  newfname,
                                                  im.size,
                                                  revmode.get(im.mode,im.mode),
                                                  ))
                
            dname,junk=os.path.split(newfname)
            if not os.path.exists(dname):
                os.makedirs(dname)
                
            im.save(newfname)
            


def load_images(dirname,test_dirname=None,filter='*.*',max_per_folder=None,verbose=True,make_grayscale=False):
    data=Struct()
    data.DESCR="Images"
    data.files=[]
    data.data=[]
    data.targets=[]
    data.target_names=[]
    
    files=os.listdir(dirname)
    
    for f in files:
        if os.path.isdir(os.path.join(dirname,f)):
            data.target_names.append(f)

    if data.target_names:
        for i,name in enumerate(data.target_names):
            files_filter=os.path.join(dirname,name,filter)
            values=glob.glob(files_filter)

            values=[_ for _ in values if 'desktop.ini' not in _]

            if not max_per_folder is None:
                if verbose:
                    print("[%s]: %d files found...%s used." % (name,len(values),max_per_folder))
                values=values[:max_per_folder]
            else:
                if verbose:
                    print("[%s]: %d files found" % (name,len(values)))
            
            data.files.extend(values)
            data.targets.extend([i]*len(values))
        data.targets=np.array(data.targets,dtype=np.int32)
            
    else:
        data.targets=None
        name='None'
        files_filter=os.path.join(dirname,filter)
        values=glob.glob(files_filter)
        if not max_per_folder is None:
            if verbose:
                print("[%s]: %d files found...%s used." % (name,len(values),max_per_folder))
            values=values[:max_per_folder]
        else:
            if verbose:
                print("[%s]: %d files found" % (name,len(values)))
        
        data.files.extend(values)


    all_same_size=True
    size=None
    for fname in data.files:
        im=Image.open(fname)
        if im.mode=='1' or im.mode=='LA':
            im=im.convert('L')

        if make_grayscale:
            im=im.convert('L')
            
        ima=np.asarray(im)

        if len(ima.shape)==3:
            ima=ima[:,:,:3]  # take out the alpha channel if it exists

        if size is None:
            size=ima.shape
        else:
            if ima.shape!=size:
                all_same_size=False

        data.data.append(ima)
        
    if not all_same_size:
        print("Warning: not all images the same size.")

    return data

def images_to_vectors(origdata,truncate=False,verbose=True):

    same_shape=True
    first_time=True
    for im in origdata.data:
        shape=im.shape
        if first_time:
            smallest_shape=im.shape
            first_time=False
            
        if im.shape!=smallest_shape:
            if not truncate:
                raise ValueError('Not all images have the same shape')
            smallest_shape=[min(x,y) for x,y in zip(im.shape,smallest_shape)]
            same_shape=False
            
    data=Struct()
    data.target_names=origdata.target_names
    data.targets=origdata.targets
    data.files=origdata.files
    data.vectors=[]
    data.shape=smallest_shape

    for ima in origdata.data:
    
        if not same_shape:
            if len(smallest_shape)==2:
                ima=ima[:smallest_shape[0],:smallest_shape[1]]
            elif len(smallest_shape)==3:
                ima=ima[:smallest_shape[0],:smallest_shape[1],:smallest_shape[2]]
            else:
                raise ValueError(">3D shapes not supported")
                
        vec=ima.ravel()
        vec=vec.astype(np.float)
        
        data.vectors.append(vec)
        
    data.vectors=np.array(data.vectors)
    data.feature_names=['p%d' % p for p in range(data.vectors.shape[1])]
    if verbose:
        classy.datasets.summary(data)
        
    return data

def extract_patches_2d_nooverlap(ima,patch_size,max_patches=1e500):
    patches=[]
    pr,pc=patch_size
    ir,ic=ima.shape[:2]
    r=0
    while (r+pr)<=ir:
        c=0
        while (c+pc)<=ic:
            patches.append(ima[r:(r+pr),c:(c+pc),...])
            c+=pc
        r+=pr
        
    patches=np.array(patches)
    return patches
        
def reconstruct_from_patches_2d_nooverlap(patches,original_shape):
    ima=np.zeros(original_shape)
    patch_size=patches[0].shape
    pr,pc=patch_size
    ir,ic=ima.shape[:2]
    
    count=0
    r=0
    while (r+pr)<=ir:
        c=0
        while (c+pc)<=ic:
            patch=patches[count]            
            ima[r:(r+pr),c:(c+pc)]=patch
            c+=pc
            count+=1
        r+=pr
    
    return ima        
    
    
def images_to_patch_vectors(origdata,patch_size,max_patches=None,overlap=True,
                with_transparent_patches=True,grayscale=True,
                verbose=True):
    from sklearn.feature_extraction.image import extract_patches_2d
    data=Struct()
    data.DESCR="Patches"
    data.target_names=origdata.target_names
    data.files=origdata.files
    data.targets=[]
    data.vectors=[]
    data.overlap=overlap
    data.with_transparent_patches=with_transparent_patches
    data.grayscale=grayscale
    
    data.shape=patch_size
    data.original_shapes=[]
    data.original_targets=origdata.targets
    data.original_vector_number=[]

    for k,ima in enumerate(origdata.data):
        if not origdata.targets is None:
            target=origdata.targets[k]
        else:
            target=None
        data.original_shapes.append(ima.shape)
        
        if overlap:
            patches=extract_patches_2d(ima, patch_size, max_patches=max_patches)
        else:
            patches=extract_patches_2d_nooverlap(ima, patch_size, max_patches=max_patches)
            
        for patch_num in range(patches.shape[0]):
            patch=patches[patch_num,...]
            if not with_transparent_patches:
                if len(patch.shape)==3:
                    visible=patch[:,:,2]
                    if not visible.all():
                        continue
                elif (patch==1.0).any():
                    continue
                

            if grayscale:
                try:
                    patch=patch[:,:,0]
                except IndexError:
                    pass
                
            vec=patch.ravel()
            data.vectors.append(vec)
            
            if not target is None:
                data.targets.append(target)
                
            data.original_vector_number.append(k)
           
           

    data.vectors=np.array(data.vectors,dtype=np.float)
    data.original_vector_number=np.array(data.original_vector_number,dtype=np.int32)
    if not target is None:
        data.targets=np.array(data.targets)
    else:
        data.targets=None


    L=np.prod(patch_size)
    if len(data.vectors[0])==L:  # 1 channel
        data.feature_names=['%d' % _i for _i in range(L)]
    elif len(data.vectors[0])==3*L:  # 3 channels:
        data.feature_names=['r%d' % _i for _i in range(L)]+['g%d' % _i for _i in range(L)]+['b%d' % _i for _i in range(L)]
    elif len(data.vectors[0])==4*L:  # 4 channels:
        data.feature_names=['r%d' % _i for _i in range(L)]+['g%d' % _i for _i in range(L)]+['b%d' % _i for _i in range(L)]+['a%d' % _i for _i in range(L)]
    else:
        data.feature_names=['%d' % _i for _i in range(len(data.vectors[0]))]
    if verbose:
        classy.datasets.summary(data)
    
    return data
    
def patch_vectors_to_images(origdata,verbose=True):
    from sklearn.feature_extraction.image import reconstruct_from_patches_2d

    data=Struct()
    data.DESCR="Images"
    data.target_names=origdata.target_names
    data.files=origdata.files
    data.targets=origdata.original_targets
    data.data=[]
    
    max_vector_number=len(data.targets)
    
    patch_array=[]
    for c in range(max_vector_number):
        patches=[vec.reshape(origdata.shape) 
                        for vec,i in zip(origdata.vectors,origdata.original_vector_number) if i==c]
        patch_array=np.array(patches)
    
        if origdata.overlap:
            data.data.append(reconstruct_from_patches_2d(patch_array,origdata.original_shapes[c]))
        else:
            data.data.append(reconstruct_from_patches_2d_nooverlap(patch_array,origdata.original_shapes[c]))
    
    if verbose:
        classy.datasets.summary(data)

    return data
    
    
        
def images_to_random_pixel_vectors(origdata,number_of_pixels,maximum_points=None,verbose=True):
    from random import shuffle
    sz=number_of_pixels
    
    data=Struct()
    data.DESCR="Pixel Vectors"
    data.target_names=origdata.target_names
    data.files=origdata.files
    data.targets=[]
    data.vectors=[]
    data.original_targets=origdata.targets
    data.original_vector_number=[]
    
    for k,im in enumerate(origdata.data):
        if not origdata.targets is None:
            target=origdata.targets[k]
        else:
            target=None
            
        mx=im.max()
        
        try:    
            grayim=im[:,:,0]
            visible=im[:,:,3]
            X=grayim[visible>0]
        except IndexError:  # not transparent
            grayim=im
            X=grayim[(grayim<mx) & (grayim>0)]
            
        if not maximum_points is None:
            X=X[:maximum_points]
        
        
        shuffle(X)
        L=len(X)
    
    
        for i in range(L//sz):
            vec=X[(i*sz):((i+1)*sz)].ravel()
            data.vectors.append(vec)
            if not target is None:
                data.targets.append(target)
            data.original_vector_number.append(k)
            
    data.feature_names=list(range(sz))
    
    data.vectors=np.array(data.vectors)
    if not target is None:
        data.targets=np.array(data.targets)
    else:
        data.targets=None
    data.original_vector_number=np.array(data.original_vector_number,dtype=np.int32)
    if verbose:
        classy.datasets.summary(data)
        
    return data
        