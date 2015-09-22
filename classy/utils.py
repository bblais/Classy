import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['axes.grid']=True


def bigfonts(size=20,family='sans-serif'):

    from matplotlib import rc

    rc('font',size=size,family=family)
    rc('axes',labelsize=size)
    rc('axes',titlesize=size)
    rc('xtick',labelsize=size)
    rc('ytick',labelsize=size)
    rc('legend',fontsize=size)

bigfonts()

# Create color maps
light_colors=['#FFAAAA', '#AAFFAA', '#AAAAFF',
                            '#CC66CC','#FFFF99','#33FF66']
bold_colors=['#FF0000', '#00FF00', '#0000FF',
                             '#CC0099','#FFFF00','#339933']

def setminmax(data,min=None,max=None):
    if min is None:
        min=data.vectors.min()
    if max is None:
        max=data.vectors.max()
        
    mn=data.vectors.min()
    mx=data.vectors.max()
    
    data.vectors-=mn
    data.vectors/=(mx-mn)
    data.vectors*=(max-min)
    data.vectors+=min
        

def standardize(data):
    from sklearn.preprocessing import scale
    scale(data.vectors,copy=False)
    

def cross_validation(cl,vectors,targets,cv=5):
    from numpy import mean,std,sqrt
    from sklearn.cross_validation import cross_val_score
    scores = cross_val_score(cl, vectors, targets, cv=cv)
    estimate=mean(scores)
    estimate_err=2*std(scores)/sqrt(len(scores))
    result='%.3f +- %.3f' % (estimate,estimate_err)
    
    return scores, result

def leaveoneout_cross_validation(cl,vectors,targets):
    from sklearn.cross_validation import LeaveOneOut
    return cross_validation(cl,vectors,targets,cv=LeaveOneOut(len(vectors)))

def time2str(t):

    minutes=60
    hours=60*minutes
    days=24*hours
    years=365*days
    
    yr=int(t/years)
    t-=yr*years

    dy=int(t/days)
    t-=dy*days
    
    hr=int(t/hours)
    t-=hr*hours

    mn=int(t/minutes)
    t-=mn*minutes

    sec=t

    s=""
    if yr>0:
        s+=str(yr)+" years "
    if hr>0:
        s+=str(hr)+" hours "
    if mn>0:
        s+=str(mn)+" minutes "        
        
    s+=str(sec)+" seconds "


    return s

def timeit(reset=False):
    from time import time
    
    global _timeit_time
    
    try:
        if _timeit_time is None:
            pass
        # is defined
    except NameError:
        _timeit_time=time()
        print "Time Reset"
        return
    
    if reset:
        _timeit_time=time()
        print "Time Reset"
        return

    return time2str(time()-_timeit_time)

def plot_feature_combinations(data,feature_names=None,figsize=(20,20)):
    import numpy as np
    import pylab as pl
    import classy.datasets
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(light_colors[:len(data.target_names)])
    cmap_bold = ListedColormap(bold_colors[:len(data.target_names)])
    assert len(data.target_names)<=len(bold_colors),"Only %d target colors implemented." % (len(cmap_bold.colors))
    
    if feature_names is None:
        feature_names=data.feature_names
        
        
    L=len(feature_names)
    pl.figure(figsize=figsize)

    for j,f1 in enumerate(feature_names):
    
        for i,f2 in enumerate(feature_names):
        
            ax = pl.subplot2grid((L,L),(i, j))
            
            
            if i==0 and j==0:
                if not data.targets is None:
                    lines=[]
                    for ti,t in enumerate(data.target_names):
                        line = Line2D(range(2), range(2), linestyle='-', marker='o',color=cmap_bold.colors[ti])
                        lines.append(line)
        
                    pl.legend(lines,data.target_names,loc='center')

                    
            if f1==f2:
                ax.set_axis_off()
                continue
                
            subset=classy.datasets.extract_features(data,[f1,f2])
            
            plot2D(subset,legend_location=None)
            if j>0 and i>1:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            if j>1:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            if i<(L-1) and j<(L-1):
                ax.set_xlabel('')
                ax.set_xticklabels([])
            if i<(L-2):
                ax.set_xlabel('')
                ax.set_xticklabels([])


def plot2D(data,classifier=None,axis_range=None,
    number_of_grid_points=100,legend_location='best'):

    import numpy as np
    import pylab as pl
    from matplotlib.lines import Line2D

    from matplotlib.colors import ListedColormap



    assert len(data.feature_names)==2,"Function only works for 2D data."
    assert len(data.target_names)<=len(bold_colors),"Only %d target colors implemented." % (len(cmap_bold.colors))

    
    cmap_light = ListedColormap(light_colors[:len(data.target_names)])
    cmap_bold = ListedColormap(bold_colors[:len(data.target_names)])


    
    if axis_range is None:        
        x_max, y_max=data.vectors.max(axis=0)
        x_min, y_min=data.vectors.min(axis=0)
    else:
        x_min,x_max,y_min,y_max=axis_range
        
    if not classifier is None:
        xx, yy = np.meshgrid(np.linspace(x_min,x_max,number_of_grid_points),
                             np.linspace(y_min,y_max,number_of_grid_points))
        Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])  
        Z = Z.reshape(xx.shape)  
        pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    if not data.targets is None:
        pl.scatter(data.vectors[:, 0], data.vectors[:, 1], c=data.targets, cmap=cmap_bold)
    else:
        pl.scatter(data.vectors[:, 0], data.vectors[:, 1], c='k')
    
    pl.xlabel(data.feature_names[0])
    pl.ylabel(data.feature_names[1])
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)

    if not data.targets is None:
        lines=[]
        for ti,t in enumerate(data.target_names):
            line = Line2D(range(2), range(2), linestyle='-', marker='o',color=cmap_bold.colors[ti])
            lines.append(line)
        
        if not legend_location is None:
            pl.legend(lines,data.target_names,loc=legend_location)
