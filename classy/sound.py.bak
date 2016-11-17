def wav2image(fname,verbose=True):
    import glob,os
    from scipy.misc import imsave
    arr=wav2spectrogram(fname)
    
    if not isinstance(arr,list):
        arr=[arr]
        
    fnames=glob.glob(fname)

    if not fnames:
        if verbose:
            print "No files found with pattern %s" % fname
            return

    for fname,im in zip(fnames,arr):
        base,ext=os.path.splitext(fname)
        fname=base+".png"
        if verbose:
            print fname
        imsave(fname, im)
    
def slicewav(fname,t,verbose=True):
    import glob,os
    from scipy.io import wavfile

    fnames=glob.glob(fname)
    if not fnames:
        if verbose:
            print "No files found with pattern %s" % fname
            return
    
    for fname in fnames:
        if '_slice_' in fname:
            if verbose:
                print "Skipping slice file...",fname
            continue
            
        rate,data=wavfile.read(fname)

        idx_step=int(rate*t)
        num_idx=data.shape[0]//idx_step
        if verbose:
            print "Slicing file...",fname
        
        for i in range(num_idx):
            start=i*idx_step
            end=(i+1)*idx_step
            new_data=data[start:end,:]
            base,ext=os.path.splitext(fname)
            newfname=base+"_slice_t%d_t%d.wav" % (i*t,(i+1)*t)
            wavfile.write(newfname,rate,new_data)
            if verbose:
                print "\t",newfname
                
    
    
    
def wav2spectrogram(fname,plot=False,save=False,freq_min=0,freq_max=5000):
    import glob
    import numpy as np
    import os
    from numpy import pi,linspace,log10,flipud
    from matplotlib.mlab import specgram
    from pylab import imshow,gca,draw,show,title,xlabel,ylabel
    from scipy.io import wavfile
    from Memory import Remember

    fnames=glob.glob(fname)
    if not fnames:
        print "No files found with pattern %s" % fname
        return

    
    Zs=[]
    for fname in fnames:
    
        sampling,data=wavfile.read(fname)
        
        data=data[:,0]
        Pxx, freqs, bins = specgram(data[0:sampling*10],NFFT=2048)
        
        dx=1.0/sampling
        dk=1.0/(len(data)*dx)*2*pi
        
        kmin=(-1/(2.0*dx))*2*pi
        kmax=1/(2.0*dx)*2*pi
        
        fmax=1/(2.0*dx)
        
        t=linspace(0,len(data)*dx,len(bins))
        f=linspace(0,fmax,len(freqs))
        
        Z = 10. * log10(Pxx)
        
        idx=f<freq_max
        f=f[idx]
        Z=Z[idx,:]
    
        idx=f>freq_min
        f=f[idx]
        Z=Z[idx,:]
    
        Z = flipud(Z)
        xextent = 0, np.amax(bins)
        xmin, xmax = xextent
        extent = t[0],t[-1],f[0],f[-1]
        dt=t[1]-t[0]
        df=f[1]-f[0]
        
        if plot:
            im = imshow(Z, extent=extent, interpolation='nearest')
            gca().axis('auto')
            title('dt=%.2e, df=%.2f, shape=(%dx%d)' % (dt,
                                                       df,
                                                       Z.shape[0],
                                                       Z.shape[1]))
            xlabel('time (sec)')
            ylabel('freq (Hz)')
            draw()
            show()
    
        if save:
            base,ext=os.path.splitext(fname)
            newfname=base+"_spectrogram.ssdf"
            print "Saving %s..." % (newfname),
            data={'im':Z,'im_scale_shift':[1,0],
                'extent':extent,
                'dt':dt,'df':df}
            Remember(data,filename=newfname)
            print "done."
        
        Zs.append(Z)
        
    if len(Zs)==1:
        return Z
    else:
        return Zs
