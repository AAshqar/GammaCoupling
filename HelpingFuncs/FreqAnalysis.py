from spectrum import pmtm
import numpy as np
import matplotlib.pyplot as plt

def comp_mtspectrogram(Signal, fs, W=2**13, ws=None, NFFT=None, freq_limit=None, freq_lowerlimit=None, unbias=True, NW=2.5, avgspectrum=False, movingW=False, PlotFlag=True):
    
    N = int(len(Signal))
    T = (N/fs)*1000.
    
    if unbias:
        Signal -= np.mean(Signal)
        
    fmax = np.round(fs/2.)
    
    if avgspectrum and not movingW:
        
        NFFT = 2**(N-1).bit_length()
        freq_vect = np.linspace(0, fmax, NFFT/2)

        a = pmtm(Signal, NFFT=NFFT, NW=NW, method='eigen', show=False)
        Signal_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
        if not freq_limit is None:
            Signal_MTS = Signal_MTS[np.where(freq_vect<=freq_limit)]
            freq_vect = freq_vect[np.where(freq_vect<=freq_limit)]
        if not freq_lowerlimit is None:
            Signal_MTS = Signal_MTS[np.where(freq_vect>freq_lowerlimit)]
            freq_vect = freq_vect[np.where(freq_vect>freq_lowerlimit)]
            
        if PlotFlag:
            plt.figure(figsize=[10,5])
            plt.plot(freq_vect, Signal_MTS) 
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.title('Multitaper Spectrum')
            plt.show()
    
    else:

        if ws is None:
            ws = int(np.round(W/10.))

        if NFFT is None:
            NFFT = W*2

        N_segs = int((N-W)/ws+2)
        result = np.zeros((NFFT/2, N_segs))
        for i in range(N_segs):
            data = Signal[i*ws:i*ws+W]
            data -= np.mean(data)
            a = pmtm(data, NFFT=NFFT, NW=NW, method='eigen', show=False)
            Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
            result[:,i] = Sks[:NFFT/2]/W
            
        freq_vect = np.linspace(0, fmax, NFFT/2)
        
        if not freq_limit is None:
            Signal_MTS = np.squeeze(result[np.where(freq_vect<=freq_limit),:])
            freq_vect = freq_vect[np.where(freq_vect<=freq_limit)]
        else:
            Signal_MTS = np.squeeze(result)
            
        if not freq_lowerlimit is None:
            Signal_MTS = np.squeeze(Signal_MTS[np.where(freq_vect>freq_lowerlimit),:])
            freq_vect = freq_vect[np.where(freq_vect>freq_lowerlimit)]
        else:
            Signal_MTS = np.squeeze(Signal_MTS)
            
        if PlotFlag and not avgspectrum:
            plt.figure(figsize=[10,5])
            plt.imshow(Signal_MTS, origin="lower", extent=[0, T, freq_vect[0], freq_vect[-1]], aspect="auto", cmap='jet') 
            plt.xlabel('Time (ms)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Multitaper Spectrogram')
            plt.show()
            
        if avgspectrum:
            Signal_MTS = np.mean(Signal_MTS, axis=1)
            if PlotFlag:
                plt.figure(figsize=[10,5])
                plt.plot(freq_vect, Signal_MTS) 
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power')
                plt.title('Multitaper Spectrum')
                plt.show()

    return Signal_MTS, freq_vect
