import numpy as np

def findpeaks(xarray, x_mindist=5., maxima_minval=0., minima=False, minima_maxval=0.):

    pk_inds = np.where(np.diff(xarray)*np.roll(np.diff(xarray),1)<=0)[0]
    
    if len(pk_inds)==0:
        return [],[]
    
    #remove peaks on the edges:
    if pk_inds[0] == 0:
        pk_inds = pk_inds[1:]
    if pk_inds[-1] == xarray.shape[0]-1:
        pk_inds = pk_inds[:-1]
    
    max_inds = np.array(filter(lambda x: xarray[x-1]<=xarray[x] and xarray[x+1]<=xarray[x], pk_inds))
    if not len(max_inds)==0:
        max_vals = xarray[max_inds]
        max_inds = np.delete(max_inds, np.where(max_vals<maxima_minval))
        max_inds = np.array(sorted(max_inds, key=lambda a: xarray[a], reverse=True))
    max_inds_filt = max_inds
    
    if minima:
        min_inds = np.array(filter(lambda x: xarray[x-1]>=xarray[x] and xarray[x+1]>=xarray[x], pk_inds))
        if not len(min_inds)==0:
            min_vals = xarray[min_inds]
            min_inds = np.delete(min_inds, np.where(min_vals>minima_maxval))
            min_inds = np.array(sorted(min_inds, key=lambda a: xarray[a]))
        min_inds_filt = min_inds
    
    if x_mindist>0:
        
        skip_pks = np.empty([0])
        max_inds_filt = []
        
        for i,mxind in enumerate(max_inds):
            if i in skip_pks:
                continue
            closepks = filter(lambda x: abs(x[1]-mxind)<x_mindist, enumerate(max_inds))
            closepks = filter(lambda x: x[0] not in skip_pks, closepks)
            skip_pks = np.append(skip_pks, [closepk[0] for closepk in closepks])
            closepks = np.array([closepk[1] for closepk in closepks])
            closepks_dists = np.array(abs(closepks-mxind))
            winnerpk = closepks[np.argmin(closepks_dists)]
            max_inds_filt.append(winnerpk)
            skip_pks = np.array(filter(lambda x: max_inds[int(x)] != winnerpk, skip_pks))
        
        if minima:
            skip_pks = np.empty([0])
            min_inds_filt = []
            for i,mnind in enumerate(min_inds):
                if i in skip_pks:
                    continue
                closepks = filter(lambda x: abs(x[1]-mnind)<x_mindist, enumerate(min_inds))
                closepks = filter(lambda x: x[0] not in skip_pks, closepks)
                skip_pks = np.append(skip_pks, [closepk[0] for closepk in closepks])
                closepks = np.array([closepk[1] for closepk in closepks])
                closepks_dists = np.array(abs(closepks-mnind))
                winnerpk = closepks[np.argmin(closepks_dists)]
                min_inds_filt.append(winnerpk)
                skip_pks = np.array(filter(lambda x: min_inds[int(x)] != winnerpk, skip_pks))

    if minima:
        return np.array(max_inds_filt), np.array(min_inds_filt)
    else:
        return np.array(max_inds_filt)

#######################################################################


def findpeaks2D(xmatrix, x_mindist=0., y_mindist=0., xy_mindist=0, maxima_minval=0., minima=False, minima_maxval=0.):

    diffmat0 = np.diff(xmatrix, axis=0)[:,:-1]
    diffmat1 = np.diff(xmatrix, axis=1)[:-1,:]
    
    pk_inds1_r, pk_inds1_c = np.where(diffmat0*np.roll(diffmat0,1, axis=0)<=0)
    pks_inds1M = np.zeros((len(pk_inds1_r), 2))
    pks_inds1M[:,0] = pk_inds1_r
    pks_inds1M[:,1] = pk_inds1_c
    
    pk_inds2_r, pk_inds2_c = np.where(diffmat1*np.roll(diffmat1,1, axis=1)<=0)
    pks_inds2M = np.zeros((len(pk_inds2_r), 2))
    pks_inds2M[:,0] = pk_inds2_r
    pks_inds2M[:,1] = pk_inds2_c
    
    intersect_pksM = np.array(filter(lambda x: all(np.isin(x, pks_inds2M)), pks_inds1M))
    
    #remove peaks on the edges:
    startedge_idx = filter(lambda x: any(intersect_pksM[x]==0), range(len(intersect_pksM)))
    endedge_idx = filter(lambda x: intersect_pksM[x,0]==xmatrix.shape[0]-1 or intersect_pksM[x,1]==xmatrix.shape[1]-1, range(len(intersect_pksM)))
    intersect_pksM = np.delete(intersect_pksM, startedge_idx, axis=0)
    intersect_pksM = np.delete(intersect_pksM, endedge_idx, axis=0)
    
    intersect_pksM = intersect_pksM.astype(int)
    
    max_inds = np.array(filter(lambda x: (xmatrix[x[0]-1,x[1]]<=xmatrix[x[0],x[1]]
                               and xmatrix[x[0]+1,x[1]]<=xmatrix[x[0],x[1]]
                               and xmatrix[x[0],x[1]-1]<=xmatrix[x[0],x[1]]
                               and xmatrix[x[0],x[1]+1]<=xmatrix[x[0],x[1]]),
                               intersect_pksM))
    max_vals = xmatrix[max_inds[:,0], max_inds[:,1]]
    max_inds = np.delete(max_inds, np.where(max_vals<maxima_minval), axis=0)

    max_inds = np.array(sorted(max_inds, key=lambda a: xmatrix[a[0],a[1]], reverse=True))
    max_inds_filt = max_inds

    if minima:
        min_inds = np.array(filter(lambda x: xmatrix[x[0]-1,x[1]]>=xmatrix[x[0],x[1]]
                               and xmatrix[x[0]+1,x[1]]>=xmatrix[x[0],x[1]]
                               and xmatrix[x[0],x[1]-1]>=xmatrix[x[0],x[1]]
                               and xmatrix[x[0],x[1]+1]>=xmatrix[x[0],x[1]],
                               intersect_pksM))
        min_vals = xmatrix[min_inds[:,0], min_inds[:,1]]
        min_inds = np.delete(min_inds, np.where(min_vals>minima_maxval), axis=0)
        min_inds = np.array(sorted(max_inds, key=lambda a: xmatrix[a[0],a[1]], reverse=True))
        min_inds_filt = max_inds
    
    if type(xy_mindist) is list:
        xy_mindist = np.linalg.norm(xy_mindist)
        
    if any((x_mindist>0, y_mindist>0, xy_mindist>0)):
        
        skip_pks = np.empty([0])
        max_inds_filt = []
        for i,mxind in enumerate(max_inds):
            if i in skip_pks:
                continue
            if x_mindist>0:
                closepks = filter(lambda x: abs(x[1][0]-mxind[0])<x_mindist, enumerate(max_inds))
                closepks = filter(lambda x: x[0] not in skip_pks, closepks)
                skip_pks = np.append(skip_pks, [closepk[0] for closepk in closepks])
                closepks = np.array([closepk[1] for closepk in closepks])
                closepks_dists = np.array(abs(closepks[:,0]-mxind[0]))            
                winnerpk = closepks[np.argmin(closepks_dists),:]
                max_inds_filt.append(winnerpk)
                skip_pks = np.array(filter(lambda x: max_inds[int(x),:] != winnerpk, skip_pks))
            if y_mindist>0:
                closepks = filter(lambda x: abs(x[1][1]-mxind[1])<x_mindist, enumerate(max_inds))
                closepks = filter(lambda x: x[0] not in skip_pks, closepks)
                skip_pks = np.append(skip_pks, [closepk[0] for closepk in closepks])
                closepks = np.array([closepk[1] for closepk in closepks])
                closepks_dists = np.array(abs(closepks[:,1]-mxind[1]))
                winnerpk = closepks[np.argmin(closepks_dists),:]
                max_inds_filt.append(winnerpk)
                skip_pks = np.array(filter(lambda x: max_inds[int(x),:] != winnerpk, skip_pks))
            if xy_mindist>0:
                closepks = filter(lambda x: abs(np.linalg.norm(x[1]-mxind))<xy_mindist, enumerate(max_inds))
                closepks = filter(lambda x: x[0] not in skip_pks, closepks)
                skip_pks = np.append(skip_pks, [closepk[0] for closepk in closepks])
                closepks = np.array([closepk[1] for closepk in closepks])
                closepks_dists = np.linalg.norm(closepks-mxind, axis=1)
                winnerpk = closepks[np.argmin(closepks_dists),:]
                max_inds_filt.append(winnerpk)
                skip_pks = np.array(filter(lambda x: max_inds[int(x),:] != winnerpk, skip_pks))
        
        if minima:
            skip_pks = np.empty([0])
            min_inds_filt = []
            for i,mnind in enumerate(min_inds):
                if i in skip_pks:
                    continue
                if x_mindist>0:
                    closepks = filter(lambda x: abs(x[1][0]-mnind[0])<x_mindist, enumerate(min_inds))
                    closepks = filter(lambda x: x[0] not in skip_pks, closepks)
                    skip_pks = np.append(skip_pks, [closepk[0] for closepk in closepks])
                    closepks = np.array([closepk[1] for closepk in closepks])
                    closepks_dists = np.array(abs(closepks[:,0]-mnind[0]))            
                    winnerpk = closepks[np.argmin(closepks_dists),:]
                    min_inds_filt.append(winnerpk)
                    skip_pks = np.array(filter(lambda x: min_inds[int(x),:] != winnerpk, skip_pks))
                if y_mindist>0:
                    closepks = filter(lambda x: abs(x[1][1]-mnind[1])<x_mindist, enumerate(min_inds))
                    closepks = filter(lambda x: x[0] not in skip_pks, closepks)
                    skip_pks = np.append(skip_pks, [closepk[0] for closepk in closepks])
                    closepks = np.array([closepk[1] for closepk in closepks])
                    closepks_dists = np.array(abs(closepks[:,1]-mnind[1]))
                    winnerpk = closepks[np.argmin(closepks_dists),:]
                    min_inds_filt.append(winnerpk)
                    skip_pks = np.array(filter(lambda x: min_inds[int(x),:] != winnerpk, skip_pks))
                if xy_mindist>0:
                    closepks = filter(lambda x: abs(np.linalg.norm(x[1]-mnind))<xy_mindist, enumerate(min_inds))
                    closepks = filter(lambda x: x[0] not in skip_pks, closepks)
                    skip_pks = np.append(skip_pks, [closepk[0] for closepk in closepks])
                    closepks = np.array([closepk[1] for closepk in closepks])
                    closepks_dists = np.linalg.norm(closepks-mnind, axis=1)
                    winnerpk = closepks[np.argmin(closepks_dists),:]
                    min_inds_filt.append(winnerpk)
                    skip_pks = np.array(filter(lambda x: min_inds[int(x),:] != winnerpk, skip_pks))
    
    if minima:
        return np.array(max_inds_filt), np.array(min_inds_filt)
    else:
        return np.array(max_inds_filt)
