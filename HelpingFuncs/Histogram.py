import numpy as np
import matplotlib.pyplot as plt
from HelpingFuncs.peakdetect import peakdet


def movinghist(array, W, nbins=None, ws=None, normalize=True, PlotFlag=True):
    
    N = int(len(array))

    if ws is None:
        ws = W/10
        
    if nbins is None:
        nbins = 20
    
    h, bins = np.array([]), np.array([])
    
    N_segs = int((N / ws)-(W-ws)/ws +1)
    result = np.zeros((nbins, N_segs))
    sample_idx = np.zeros(N_segs)
    for i in range(N_segs):
        h, bins = np.histogram(array, bins=nbins)
        bins = bins[:-1] + (bins[1]-bins[0])/2.
        if normalize:
            h /= np.sum(h)
        result[:,i] = h
        sample_idx[i] = int((i*ws+W/2.))
    
    if PlotFlag:
        plt.figure(figsize=[10,5])
        plt.imshow(result, origin="lower", extent=[sample_idx[0], sample_idx[-1], bins[0], bins[-1]], aspect="auto", cmap='jet') 
        plt.xlabel('Sample Index')
        plt.ylabel('Bins')
        plt.title('Moving Histogram')

        plt.show()
    
    return result, bins

def ISImovinghist(Signal, fs, W, nbins=None, hrange=None, ws=None, normalize=True, PlotFlag=True, FreqAxis=True):
    
    N = int(len(Signal))
    T = (N/fs)
    time_v = np.linspace(1./fs, T, N) #ms

    if ws is None:
        ws = W/10
        
    if nbins is None:
        nbins = 20
    
    N_segs = int((N / ws)-(W-ws)/ws +1)
    result = np.zeros((nbins, N_segs))
    time_idx = np.zeros(N_segs)
    for i in range(N_segs):
        maxtbl, mintbl = peakdet(Signal[i*ws:i*ws+W], delta=np.std(Signal)/2)
        maxindsP = maxtbl[:,0].astype(np.int)
        ISIarray = np.diff(time_v[maxindsP])
        if FreqAxis:
            ISIarray = 1/(ISIarray/1000) #Hz
        
        h, bins = np.histogram(ISIarray, bins=nbins, range=hrange)
        bins = bins[:-1] + (bins[1]-bins[0])/2.
        if normalize:
            h = h.astype(np.float)/np.sum(h)
        result[:,i] = h
        time_idx[i] = time_v[int((i*ws+W/2.))]
    
    if PlotFlag:
        plt.figure(figsize=[10,5])
        plt.imshow(result, origin="lower", extent=[time_idx[0], time_idx[-1], bins[0], bins[-1]], aspect="auto", cmap='jet') 
        plt.xlabel('Time')
        plt.ylabel('Bins')
        plt.title('Moving Histogram')

        plt.show()
    
    return result, bins, time_idx

def ISIrate_stat(Signal, fs, W=2**13, ws=None, bins=None, hrange=None):
    
    N = int(len(Signal))
    T = (N/fs)
    time_v = np.arange(0, T, 1./fs) #ms
    
    if ws is None:
        ws = W/10.
    if bins is None:
        bins = 60
    if hrange is None:
        hrange = (0, 300)
    
    N_segs = int((N-W)/ws+2)
    time_ISIs = time_v[np.arange(W/2, (N_segs)*ws+(W/2), ws).astype(int)]
    ISIwinners = np.zeros(N_segs)
    for i in range(N_segs):
        maxtbl, mintbl = peakdet(Signal[int(i*ws):int(i*ws+W)], delta=np.std(Signal)/2)
        maxinds = maxtbl[:,0].astype(np.int)
        ISIs = np.diff(time_v[maxinds])
        ISIs = 1/(ISIs/1000) #Hz
        h, b = np.histogram(ISIs, bins=bins, range=hrange)
        ISIwinners[i] = b[:-1][np.argmax(h)]

    return ISIwinners, time_ISIs
