from brian2 import *
import numpy as np
from collections import Counter
import operator
from scipy.signal import butter, lfilter
from skimage.feature import peak_local_max
import re
import tables
import pickle
import gc
import time
from Network_utils import *
from PlottingFuncs import *
from HelpingFuncs.peakfinders import * 
from HelpingFuncs.FreqAnalysis import comp_mtspectrogram
from HelpingFuncs.Histogram import movinghist


def ExtractNeuronsFeats(filename, N_p=4000, N_i=1000, W=2**13, sim_dt=0.02, montdt=0.2, start_time=200, ws=None, NFFT=None, freq_limit=300, NW=2.5, midfreq_def=105., CircPhase=False, save_filename=None):
    
    fs = np.round(1./(sim_dt*1e-3),1)
    
    if ws is None:
        ws = int(np.round(W/10.))
    if NFFT is None:
        NFFT = W*2.

    f = tables.open_file(filename, 'r')
    SpikeM_i_Int = f.root.SpikeM_i_Int.read()
    SpikeM_i_Pyr = f.root.SpikeM_i_Pyr.read()
    SpikeM_t_Int = f.root.SpikeM_t_Int.read()
    SpikeM_t_Pyr = f.root.SpikeM_t_Pyr.read()
    PopRateSigWhole_Pyr = f.root.PopRateSig_Pyr.read()
    PopRateSigWhole_Int = f.root.PopRateSig_Int.read()
    PopRateSigWhole_Full = (4*PopRateSigWhole_Pyr+1*PopRateSigWhole_Int)/5.
    PopRateSig_Pyr = np.copy(PopRateSigWhole_Pyr[int(start_time/sim_dt):])
    PopRateSig_Int = np.copy(PopRateSigWhole_Int[int(start_time/sim_dt):])
    PopRateSig_Full = np.copy((4*PopRateSigWhole_Pyr[int(start_time/sim_dt):]+1*PopRateSigWhole_Int[int(start_time/sim_dt):])/5.)
    IsynP_Pyr = np.mean(f.root.IsynP_Pyr.read()[int(start_time/montdt):,:], axis=0)
    IsynI_Pyr = np.mean(f.root.IsynI_Pyr.read()[int(start_time/montdt):,:], axis=0)
    IsynP_Int = np.mean(f.root.IsynP_Int.read()[int(start_time/montdt):,:], axis=0)
    IsynI_Int = np.mean(f.root.IsynI_Int.read()[int(start_time/montdt):,:], axis=0)
    f.close()
    
    print('  Loaded raw data...')
    
    N_whole = len(PopRateSigWhole_Pyr)
    T_whole = (N_whole/fs)*1000.
    time_whole = np.arange(0,T_whole,sim_dt)
    time_v = time_whole[int(start_time/sim_dt):]
    timeI_whole = np.arange(0,T_whole,montdt)
    timeI_v = timeI_whole[int(start_time/montdt):]
    N = int(len(time_v))
    N_segs = int((N-W)/ws+2)
    MTS_time = time_v[np.arange(W/2, (N_segs)*ws+(W/2), ws).astype(int)]
    fmax = np.round(fs/2.,1)
    freq_vect = np.linspace(0, fmax, NFFT/2)
    freq_vect = freq_vect[np.where(freq_vect<=freq_limit)]
    
    # Pyr.
    MTS_Pyr, freq_vect = comp_mtspectrogram(PopRateSig_Pyr, fs=fs, freq_limit=freq_limit, W=W, PlotFlag=False)
    # Int.
    MTS_Int, freq_vect = comp_mtspectrogram(PopRateSig_Int, fs=fs, freq_limit=freq_limit, W=W, PlotFlag=False)
    # Full.
    MTS_Full, freq_vect = comp_mtspectrogram(PopRateSig_Full, fs=fs, freq_limit=freq_limit, W=W, PlotFlag=False)

    print('  Extracted Spectrograms...')
    
    pyrthresh = np.mean(MTS_Pyr)+2*np.std(MTS_Pyr)
    MaxInds_Pyr = peak_local_max(MTS_Pyr, threshold_abs=pyrthresh, min_distance=2)
    intthresh = np.mean(MTS_Int)+2*np.std(MTS_Int)
    MaxInds_Int = peak_local_max(MTS_Int, threshold_abs=intthresh, min_distance=2)
    fullthresh = np.mean(MTS_Full)+2*np.std(MTS_Full)
    MaxInds_Full = peak_local_max(MTS_Full, threshold_abs=fullthresh, min_distance=2)
    
    nW,binW = histogram(freq_vect[MaxInds_Full[:,0]], bins=20)
    binW += 0.5*np.diff(binW)[0]
    binW = binW[:-1]
    winfreq = binW[np.argmax(nW)]
    mindif = 20/np.diff(binW)[0]
    histmxind_W = findpeaks(nW, x_mindist=mindif, maxima_minval=np.mean(nW[nW>0]))
    if len(histmxind_W)>1:
        midfreq = np.mean(binW[histmxind_W])
    else:
        midfreq = midfreq_def
        
    LowModeInds_Pyr = np.array([(f,t) for f,t in MaxInds_Pyr if freq_vect[f]<midfreq])
    LowModeInds_Int = np.array([(f,t) for f,t in MaxInds_Int if freq_vect[f]<midfreq])
    LowModeInds_Full = np.array([(f,t) for f,t in MaxInds_Full if freq_vect[f]<midfreq])
    HighModeInds_Pyr = np.array([(f,t) for f,t in MaxInds_Pyr if freq_vect[f]>=midfreq])
    HighModeInds_Int = np.array([(f,t) for f,t in MaxInds_Int if freq_vect[f]>=midfreq])
    HighModeInds_Full = np.array([(f,t) for f,t in MaxInds_Full if freq_vect[f]>=midfreq])    
    
    NeuronFRs_Pyr = np.zeros([N_p])
    NeuronFRs_Int = np.zeros([N_i])
    SpikeTrains_Pyr = []
    SpikeTrains_Int = []
    SpikeTrainSigs_Pyr1 = np.zeros([len(time_v), int(N_p/2.)])
    SpikeTrainSigs_Pyr2 = np.zeros([len(time_v), int(N_p/2.)])
    SpikeTrainSigs_Int = np.zeros([len(time_v), N_i])
    for n in range(N_p):
        spktrn_pyr = np.round(SpikeM_t_Pyr[np.where(SpikeM_i_Pyr==n)[0]]*1000,2)
        spktrn_pyr = spktrn_pyr[spktrn_pyr>=start_time]
        timeinds_pyr = ((spktrn_pyr-start_time)/sim_dt).astype(int)
        if len(timeinds_pyr) != 0:
            if n<N_p/2:
                SpikeTrainSigs_Pyr1[timeinds_pyr, n] = 1
            else:
                SpikeTrainSigs_Pyr2[timeinds_pyr, n-N_p/2] = 1
        SpikeTrains_Pyr.append(spktrn_pyr)
        NeuronFRs_Pyr[n] = len(spktrn_pyr)/((T_whole-start_time)/1000.)
        if n < N_i:
            spktrn_int = np.round(SpikeM_t_Int[np.where(SpikeM_i_Int==n)[0]]*1000,2)
            spktrn_int = spktrn_int[spktrn_int>=start_time]
            timeinds_int = np.round((spktrn_int-start_time)/sim_dt).astype(int)
            if len(timeinds_int) != 0:
                SpikeTrainSigs_Int[timeinds_int, n] = 1   
            SpikeTrains_Int.append(spktrn_int)
            NeuronFRs_Int[n] = len(spktrn_int)/((T_whole-start_time)/1000.)
    ####################################

    print('  Constructed spike trains. Extracting phases...')
    
    bins_vect = np.arange(0,361,5)
    bins_vectC = np.arange(-180,181,5)
    sig_pyr = np.copy(PopRateSig_Pyr)
    sig_int = np.copy(PopRateSig_Int)
    NeuronPhases_Pyr = {'PING':[], 'ING':[]}
    NeuronPhases_Int = {'PING':[], 'ING':[]}
    PrefPhase_Pyr = {'PING':np.zeros([N_p]), 'ING':np.zeros([N_p])}
    PrefPhase_Int = {'PING':np.zeros([N_i]), 'ING':np.zeros([N_i])}
    PrefPhasePr_Pyr = {'PING':np.zeros([N_p]), 'ING':np.zeros([N_p])}
    PrefPhasePr_Int = {'PING':np.zeros([N_i]), 'ING':np.zeros([N_i])}

    NeuronFRsModes_Pyr = {'PING':np.zeros([N_p]), 'ING':np.zeros([N_p])}
    NeuronFRsModes_Int = {'PING':np.zeros([N_i]), 'ING':np.zeros([N_i])}
    
    if CircPhase:
        NeuronPhasesC_Pyr = {'PING':[], 'ING':[]}
        NeuronPhasesC_Int = {'PING':[], 'ING':[]}
        PrefPhaseC_Pyr = {'PING':np.zeros([N_p]), 'ING':np.zeros([N_p])}
        PrefPhaseC_Int = {'PING':np.zeros([N_i]), 'ING':np.zeros([N_i])}
        PrefPhasePrC_Pyr = {'PING':np.zeros([N_p]), 'ING':np.zeros([N_p])}
        PrefPhasePrC_Int = {'PING':np.zeros([N_i]), 'ING':np.zeros([N_i])}

    
    for n in range(N_p):
        NeuronPhases_Pyr['PING'].append([])
        NeuronPhases_Pyr['ING'].append([])
        if n<N_i:
            NeuronPhases_Int['PING'].append([])
            NeuronPhases_Int['ING'].append([])
        if CircPhase:
            for n in range(N_p):
                NeuronPhasesC_Pyr['PING'].append([])
                NeuronPhasesC_Pyr['ING'].append([])
                if n<N_i:
                    NeuronPhasesC_Int['PING'].append([])
                    NeuronPhasesC_Int['ING'].append([])

    for indi, LMind in enumerate(LowModeInds_Pyr):
        raw_ind = np.where(time_v==MTS_time[LMind[1]])[0][0]
        burstraw = sig_pyr[int(raw_ind-W/2):int(raw_ind+W/2)]
        burstraw -= np.mean(burstraw)
        burstpeaks = peakdet(burstraw, np.std(burstraw)/5.)
        burstpeaks = (burstpeaks[0][:,0]).astype(int) #indices

        spktrn_pyr1 = np.copy(SpikeTrainSigs_Pyr1[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_pyr1, neurons_pyr1 = np.where(spktrn_pyr1==1)
        for t,n in zip(times_pyr1,neurons_pyr1):
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360
            else:
                if t>burstpeaks[-1]:
                    period = burstpeaks[-1]-burstpeaks[-2]
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360
            NeuronPhases_Pyr['PING'][:int(N_p/2.)][n].append(phase%360)
        NeuronFRsModes_Pyr['PING'][:int(N_p/2.)] += np.sum(spktrn_pyr1, axis=0)

        spktrn_pyr2 = np.copy(SpikeTrainSigs_Pyr2[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_pyr2, neurons_pyr2 = np.where(spktrn_pyr2==1)
        for t,n in zip(times_pyr2,neurons_pyr2):
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360
            else:
                if t>burstpeaks[-1]:
                    period = burstpeaks[-1]-burstpeaks[-2]
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360
            NeuronPhases_Pyr['PING'][int(N_p/2.):][n].append(phase%360)
        NeuronFRsModes_Pyr['PING'][int(N_p/2.):] += np.sum(spktrn_pyr2, axis=0)

        spktrn_int = np.copy(SpikeTrainSigs_Int[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_int, neurons_int = np.where(spktrn_int==1)
        for t,n in zip(times_int,neurons_int):
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360
            else:
                if t>burstpeaks[-1]:
                    period = burstpeaks[-1]-burstpeaks[-2]
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360
            NeuronPhases_Int['PING'][n].append(phase%360)
        NeuronFRsModes_Int['PING'] += np.sum(spktrn_int, axis=0)

    NeuronFRsModes_Pyr['PING'] /= np.float(W*len(LowModeInds_Pyr)*sim_dt)/1000.
    NeuronFRsModes_Int['PING'] /= np.float(W*len(LowModeInds_Pyr)*sim_dt)/1000.
    
    for indi, HMind in enumerate(HighModeInds_Int):
        raw_ind = np.where(time_v==MTS_time[HMind[1]])[0][0]

        burstraw = sig_int[int(raw_ind-W/2):int(raw_ind+W/2)]
        burstraw -= np.mean(burstraw)

        burstpeaks = peakdet(burstraw, np.std(burstraw)/5.)
        burstpeaks = (burstpeaks[0][:,0]).astype(int) #indices

        spktrn_pyr1 = np.copy(SpikeTrainSigs_Pyr1[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_pyr1, neurons_pyr1 = np.where(spktrn_pyr1==1)
        for t,n in zip(times_pyr1,neurons_pyr1):
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360
            else:
                if t>burstpeaks[-1]:
                    period = burstpeaks[-1]-burstpeaks[-2]
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360
            NeuronPhases_Pyr['ING'][:int(N_p/2.)][n].append(phase%360)
        NeuronFRsModes_Pyr['ING'][:int(N_p/2.)] += np.sum(spktrn_pyr1, axis=0)

        spktrn_pyr2 = np.copy(SpikeTrainSigs_Pyr2[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_pyr2, neurons_pyr2 = np.where(spktrn_pyr2==1)
        for t,n in zip(times_pyr2,neurons_pyr2):
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360
            else:
                if t>burstpeaks[-1]:
                    period = burstpeaks[-1]-burstpeaks[-2]
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360
            NeuronPhases_Pyr['ING'][int(N_p/2.):][n].append(phase%360)
        NeuronFRsModes_Pyr['ING'][int(N_p/2.):] += np.sum(spktrn_pyr2, axis=0)

        spktrn_int = np.copy(SpikeTrainSigs_Int[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_int, neurons_int = np.where(spktrn_int==1)
        for t,n in zip(times_int,neurons_int):
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360
            else:
                if t>burstpeaks[-1]:
                    period = burstpeaks[-1]-burstpeaks[-2]
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360
            NeuronPhases_Int['ING'][n].append(phase%360)
        NeuronFRsModes_Int['ING'] += np.sum(spktrn_int, axis=0)
        
    NeuronFRsModes_Pyr['ING'] /= np.float(W*len(HighModeInds_Int)*sim_dt)/1000.
    NeuronFRsModes_Int['ING'] /= np.float(W*len(HighModeInds_Int)*sim_dt)/1000.

    for n in range(N_p):
        phasesLM = np.copy(np.array(NeuronPhases_Pyr['PING'][n]))
        if CircPhase:
            phasesLMc = np.copy(phasesLM)
            phasesLMc[phasesLMc>180.0] -= 360
            NeuronPhasesC_Pyr['PING'][n] = phasesLMc
        if len(phasesLM)==0:
            PrefPhase_Pyr['PING'][n] = np.float('nan')
            if CircPhase:
                PrefPhaseC_Pyr['PING'][n] = np.float('nan')
        elif len(phasesLM)==1:
            PrefPhase_Pyr['PING'][n] = phasesLM[0]
            PrefPhasePr_Pyr['PING'][n] = 1.
            if CircPhase:
                PrefPhaseC_Pyr['PING'][n] = phasesLMc[0]
                PrefPhasePrC_Pyr['PING'][n] = 1.
        else:
            h, bins = histogram(phasesLM, bins=bins_vect)
            bins = bins[:-1] + np.diff(bins)[0]/2.
            PrefPhase_Pyr['PING'][n] = bins[np.argmax(h)]
            PrefPhasePr_Pyr['PING'][n] = h[np.argmax(h)]/np.sum(h).astype(np.float)
            if CircPhase:
                h, bins = histogram(phasesLMc, bins=bins_vectC)
                bins = bins[:-1] + np.diff(bins)[0]/2.
                PrefPhaseC_Pyr['PING'][n] = bins[np.argmax(h)]
                PrefPhasePrC_Pyr['PING'][n] = h[np.argmax(h)]/np.sum(h).astype(np.float)
        phasesHM = np.copy(np.array(NeuronPhases_Pyr['ING'][n]))
        if CircPhase:
            phasesHMc = np.copy(phasesHM)
            phasesHMc[phasesHMc>180.0] -= 360
            NeuronPhasesC_Pyr['ING'][n] = phasesHMc
        if len(phasesHM)==0:
            PrefPhase_Pyr['ING'][n] = np.float('nan')
            if CircPhase:
                PrefPhaseC_Pyr['ING'][n] = np.float('nan')
        elif len(phasesHM)==1:
            PrefPhase_Pyr['ING'][n] = phasesHM[0]
            PrefPhasePr_Pyr['ING'][n] = 1.
            if CircPhase:
                PrefPhaseC_Pyr['ING'][n] = phasesHMc[0]
                PrefPhasePrC_Pyr['ING'][n] = 1.
        else:
            h, bins = histogram(phasesHM, bins=bins_vect)
            bins = bins[:-1] + np.diff(bins)[0]/2.
            PrefPhase_Pyr['ING'][n] = bins[np.argmax(h)]
            PrefPhasePr_Pyr['ING'][n] = h[np.argmax(h)]/np.sum(h).astype(np.float)
            if CircPhase:
                h, bins = histogram(phasesHMc, bins=bins_vectC)
                bins = bins[:-1] + np.diff(bins)[0]/2.
                PrefPhaseC_Pyr['ING'][n] = bins[np.argmax(h)]
                PrefPhasePrC_Pyr['ING'][n] = h[np.argmax(h)]/np.sum(h).astype(np.float)

        if n<N_i:
            phasesLM = np.copy(np.array(NeuronPhases_Int['PING'][n]))
            if CircPhase:
                phasesLMc = np.copy(phasesLM)
                phasesLMc[phasesLMc>180.0] -= 360
                NeuronPhasesC_Int['PING'][n] = phasesLMc
            if len(phasesLM)==0:
                PrefPhase_Int['PING'][n] = np.float('nan')
                if CircPhase:
                    PrefPhaseC_Int['PING'][n] = np.float('nan')
            elif len(phasesLM)==1:
                PrefPhase_Int['PING'][n] = phasesLM[0]
                PrefPhasePr_Int['PING'][n] = 1.
                if CircPhase:
                    PrefPhaseC_Int['PING'][n] = phasesLMc[0]
                    PrefPhasePrC_Int['PING'][n] = 1.
            else:
                h, bins = histogram(phasesLM, bins=bins_vect)
                bins = bins[:-1] + np.diff(bins)[0]/2.
                PrefPhase_Int['PING'][n] = bins[np.argmax(h)]
                PrefPhasePr_Int['PING'][n] = h[np.argmax(h)]/np.sum(h).astype(np.float)
                if CircPhase:
                    h, bins = histogram(phasesLMc, bins=bins_vectC)
                    bins = bins[:-1] + np.diff(bins)[0]/2.
                    PrefPhaseC_Int['PING'][n] = bins[np.argmax(h)]
                    PrefPhasePrC_Int['PING'][n] = h[np.argmax(h)]/np.sum(h).astype(np.float)
            phasesHM = np.copy(np.array(NeuronPhases_Int['ING'][n]))
            if CircPhase:
                phasesHMc = np.copy(phasesHM)
                phasesHMc[phasesHMc>180.0] -= 360
                NeuronPhasesC_Int['ING'][n] = phasesHMc
            if len(phasesHM)==0:
                PrefPhase_Int['ING'][n] = np.float('nan')
                if CircPhase:
                    PrefPhaseC_Int['ING'][n] = np.float('nan')
            elif len(phasesHM)==1:
                PrefPhase_Int['ING'][n] = phasesHM[0]
                PrefPhasePr_Int['ING'][n] = 1.
                if CircPhase:
                    PrefPhaseC_Int['ING'][n] = phasesHMc[0]
                    PrefPhasePrC_Int['ING'][n] = 1.
            else:
                h, bins = histogram(phasesHM, bins=bins_vect)
                bins = bins[:-1] + np.diff(bins)[0]/2.
                PrefPhase_Int['ING'][n] = bins[np.argmax(h)]
                PrefPhasePr_Int['ING'][n] = h[np.argmax(h)]/np.sum(h).astype(np.float)
                if CircPhase:
                    h, bins = histogram(phasesHMc, bins=bins_vectC)
                    bins = bins[:-1] + np.diff(bins)[0]/2.
                    PrefPhaseC_Int['ING'][n] = bins[np.argmax(h)]
                    PrefPhasePrC_Int['ING'][n] = h[np.argmax(h)]/np.sum(h).astype(np.float)
                
    MTS_Results = {'RateMTS_Pyr':MTS_Pyr,
                  'RateMTS_Int':MTS_Int,
                  'RateMTS_Full':MTS_Full,
                  'MaxInds_Pyr':MaxInds_Pyr,
                  'MaxInds_Int':MaxInds_Int,
                  'MaxInds_Full':MaxInds_Full,
                  'FreqMidPnt':midfreq,
                  'WinningFreq':winfreq,
                  'freq_vect':freq_vect,
                  'MTS_time':MTS_time,
                  'LowModeInds_Pyr':LowModeInds_Pyr,
                  'LowModeInds_Int':LowModeInds_Int,
                  'LowModeInds_Full':LowModeInds_Full,
                  'HighModeInds_Pyr':HighModeInds_Pyr,
                  'HighModeInds_Int':HighModeInds_Int,
                  'HighModeInds_Full':HighModeInds_Full}
    
    SpkTrains = {'SpikeTrains_Pyr':SpikeTrains_Pyr,
                 'SpikeTrains_Int':SpikeTrains_Int}
    
    NeuronsFeats = {'NeuronFRs_Pyr':NeuronFRs_Pyr,
                   'NeuronFRs_Int':NeuronFRs_Int,
                   'NeuronFRsModes_Pyr':NeuronFRsModes_Pyr,
                   'NeuronFRsModes_Int':NeuronFRsModes_Int,
                   'IsynP_Pyr':IsynP_Pyr,
                   'IsynI_Pyr':IsynI_Pyr,
                   'IsynP_Int':IsynP_Int,
                   'IsynI_Int':IsynI_Int}
    
    if CircPhase:
        NeuronsFeats['NeuronPhases_Pyr'] = NeuronPhasesC_Pyr
        NeuronsFeats['NeuronPhases_Int'] = NeuronPhasesC_Int
        NeuronsFeats['PrefPhase_Pyr'] = PrefPhaseC_Pyr
        NeuronsFeats['PrefPhase_Int'] = PrefPhaseC_Int
        NeuronsFeats['PrefPhasePr_Pyr'] = PrefPhasePrC_Pyr
        NeuronsFeats['PrefPhasePr_Int'] = PrefPhasePrC_Int
    else:
        NeuronsFeats['NeuronPhases_Pyr'] = NeuronPhases_Pyr
        NeuronsFeats['NeuronPhases_Int'] = NeuronPhases_Int
        NeuronsFeats['PrefPhase_Pyr'] = PrefPhase_Pyr
        NeuronsFeats['PrefPhase_Int'] = PrefPhase_Int
        NeuronsFeats['PrefPhasePr_Pyr'] = PrefPhasePr_Pyr
        NeuronsFeats['PrefPhasePr_Int'] = PrefPhasePr_Int
    
    if not save_filename is None:
        with open(save_filename, 'wb') as f:
            pickle.dump({'MTS_Results':MTS_Results,'SpkTrains':SpkTrains,'NeuronsFeats':NeuronsFeats}, f)
    
    return MTS_Results, SpkTrains, NeuronsFeats