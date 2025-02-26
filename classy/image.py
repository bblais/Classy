from PIL import Image,UnidentifiedImageError
from .Struct import Struct
import os
import glob
import numpy as np
import classy.datasets
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def split(images,test_size=0.2,verbose=True,shuffle=True):
    from numpy import where,array
    import numpy.random
    
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
    
    if shuffle:
        idx=np.array(range(len(d1.targets)))
        np.random.shuffle(idx)
        d1.targets=d1.targets[idx]
        d1.files=[d1.files[i] for i in idx]
        d1.data=[d1.data[i] for i in idx]


        idx=np.array(range(len(d2.targets)))
        np.random.shuffle(idx)
        d2.targets=d2.targets[idx]
        d2.files=[d2.files[i] for i in idx]
        d2.data=[d2.data[i] for i in idx]


    if verbose:
        print("Files in Test Set:")
        print("\t",','.join(d1.files))
        print("Files in Train Set:")
        print("\t",','.join(d2.files))
        
    return d2,d1

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


def load_images_from_filepatterns(delete_alpha=False,**kwargs):
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

        if delete_alpha and len(ima.shape)==3:
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
            


def load_images(dirname,test_dirname=None,filter='*.*',delete_alpha=False,
                    max_per_folder=None,verbose=True,make_grayscale=False):
    
    import zipfile
    import numpy as np
    from PIL import Image
    from io import BytesIO

    data=Struct()
    data.DESCR="Images"
    data.files=[]
    data.data=[]
    data.targets=[]
    data.target_names=[]
    
    if ".zip" in dirname:  # zip file
        zipname=dirname
        if zipname.endswith('/'):
            zipname=zipname[:-1]

        parts=zipname.split('/')
        zipname=parts[0]

        if len(parts)>1:
            rest='/'.join(parts[1:])+"/"
        else:
            rest=''

        with zipfile.ZipFile(zipname, 'r') as z:
            filenames = [f for f in z.namelist() 
                        if not '__MACOSX' in f and
                        not '.DS_Store' in f and
                        not 'Icon' in f and
                        not 'desktop.ini' in f and
                        not '.ipynb_checkpoints' in f and
                        f.startswith(rest)
                        ]

            new_filenames=[f.replace(rest,'') for f in filenames]
            #filenames=['/'.join(f.split('/')[1:]) for f in filenames]

            correct_folder_structure=all([len(f.split('/')[1:])<=2 for f in new_filenames])
            if not correct_folder_structure:
                print([f for f in new_filenames if f.endswith('/')])


            assert correct_folder_structure,"Not correct folder structure"

            target_names=[f.replace('/','') for f in new_filenames if f.endswith('/')]

            for i,name in enumerate(target_names):
                files=[f for f,f2 in zip(filenames,new_filenames)
                                            if f2.startswith(name+"/") and
                                            not f2.endswith('/')] 
                data.files.extend(files)
                data.targets.extend([i]*len(files))
                if verbose:
                    print("[%s]: %d files found" % (name,len(files)))
                
            data.target_names=target_names

            all_same_size=True
            size=None
            for fname in data.files:
                with z.open(fname) as file:
                    #img = Image.open(BytesIO(file.read())).convert('RGB')
                    try:
                        img=Image.open(BytesIO(file.read()))
                    except UnidentifiedImageError:
                        print(fname)
                        raise

                    if img.mode=='1' or img.mode=='LA':
                        img=img.convert('L')

                    if make_grayscale:
                        img=img.convert('L')
                    img=np.array(img)

                    if size is None:
                        size=img.shape
                    else:
                        if img.shape!=size:
                            all_same_size=False

                    
                    data.data.append(img)
            
            if not all_same_size:
                print("Warning: not all images the same size.")

        data.targets=np.array(data.targets,dtype=np.int32)
        return data

    elif not os.path.isdir(dirname):  # this should be a filename, or a regex
        base,fname=os.path.split(dirname)
        if not base:
            base='./'

        dirname=base
        filter=fname
    else:

        files=os.listdir(dirname)
        
        for f in files:
            if ".ipynb_checkpoints" in f:
                    continue
            if ".DS_Store" in f:
                    continue
            if "desktop.ini" in f.lower():
                    continue
            if os.path.isdir(os.path.join(dirname,f)):
                if ".ipynb_checkpoints" in f:
                    continue
                data.target_names.append(f)
            else:
                print("Expecting a folder of target-named folders.  Found ",os.path.join(dirname,f))

    if data.target_names:
        found_zero=False
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
                if len(values)==0:
                    found_zero=True

            data.files.extend(values)
            data.targets.extend([i]*len(values))
        data.targets=np.array(data.targets,dtype=np.int32)
        if found_zero:
            warnings.warn("""Look's like your folders are not organized as expected. We're expecting something like
%s/
    target_name1/
        image1.jpg                    
        image2.jpg                    
        image3.jpg                    
    target_name2/
        image4.jpg                    
        image5.jpg                    
        image6.jpg                    
etc...                    
                    """ % (dirname))
            
    else:
        data.targets=None
        name='None'
        files_filter=os.path.join(dirname,filter)
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


    all_same_size=True
    size=None
    for fname in data.files:
        im=Image.open(fname)
        if im.mode=='1' or im.mode=='LA':
            im=im.convert('L')

        if make_grayscale:
            im=im.convert('L')
            
        ima=np.asarray(im)

        if delete_alpha and len(ima.shape)==3:
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

def images_to_vectors(origdata,truncate=False,full=False,verbose=True):

    same_shape=True
    first_time=True
    smallest_shape=None
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
            
    if smallest_shape is None:
        raise ValueError("No images read.")

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
                
        if not full:
            vec=ima.ravel().astype(float)
        else:
            vec=ima.astype(float)
        
        data.vectors.append(vec)
        
    data.vectors=np.array(data.vectors)
    data.feature_names=['p%d' % p for p in range(np.prod(data.vectors.shape[1:]))]
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
           
           

    data.vectors=np.array(data.vectors,dtype=float)
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
        


def read_warped_image(filename):
    import cv2
    import numpy as np

    # Load the image
    image = cv2.imread(filename)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour that best matches the game board
    board_contour = None
    for contour in contours:
        # Approximate the contour to reduce the number of points
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if this contour has 4 points (for a rectangular shape)
        if len(approx) == 4:
            board_contour = approx
            break

    if board_contour is not None:
        # Sort the corners in the order: top-left, top-right, bottom-right, bottom-left
        corners = np.array([point[0] for point in board_contour])
        sum_pts = corners.sum(axis=1)
        diff_pts = np.diff(corners, axis=1)
        top_left = corners[np.argmin(sum_pts)]
        bottom_right = corners[np.argmax(sum_pts)]
        top_right = corners[np.argmin(diff_pts)]
        bottom_left = corners[np.argmax(diff_pts)]
        ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

        # Define the width and height of the new "top-down" view
        width = int(max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left)
        ))
        height = int(max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left)
        ))

        # Define the destination points for the perspective transform
        destination_corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Compute the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(ordered_corners, destination_corners)
        
        # Apply the perspective transformation
        warped_image = cv2.warpPerspective(image, matrix, (width, height))

        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        warped_image=cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB) 
    else:
        warped_image=None
        ordered_corners=None
        print("Could not find a rectangular contour in the image.")

    info={}
    info['ordered_corners']=ordered_corners
    info['edges']=edges

    
    return image,warped_image,info