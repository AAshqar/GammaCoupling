from spectrum import pmtm

def comp_mtspectrogram(Signal, fs, W, ws=None, NFFT=None, freq_limit=None, normalize=True, NW=2.5 PlotFlag=True):
    
    N = int(len(Signal))
    T = N/fs
    
    if normalize:
        Signal -= np.mean(Signal)

    if ws is None:
        ws = W/10
    
    if NFFT is None:
        NFFT = W*2
        
    fmax = fs/2
    
    freq_vect = np.linspace(0, fmax, NFFT/2)
    if not freq_limit is None:
        freq_vect = freq_vect[np.where(freq_vect<=freq_limit)]
    
    N_segs = int((N / ws)-(W-ws)/ws +1)
    result = np.zeros((NFFT/2, N_segs))
    for i in range(N_segs):
        data = Signal[i*ws:i*ws+W]
        a = pmtm(data, NFFT=NFFT, NW=NW, method='eigen', show=False)
        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
        result[:,i] = Sks[:NFFT/2]/W
        
    if not freq_limit is None:
        Signal_MTS = np.squeeze(result[np.where(freq_vect/Hz<=300),:])
    else:
        Signal_MTS = np.squeeze(result)
    
    if PlotFlag:
        figure(figsize=[10,10])
        imshow(Signal_MTS, origin="lower", extent=[0, T, freq_vect[0], freq_vect[-1]], aspect="auto", cmap='jet') 
        xlabel('Time (ms)')
        ylabel('Frequency (Hz)')
        title('Multitaper Spectrogram')

        show()
    
    return Signal_MTS
