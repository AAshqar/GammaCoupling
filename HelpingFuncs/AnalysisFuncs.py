from brian2 import *
import numpy as np
from collections import Counter
import operator
from scipy.signal import butter, lfilter
from scipy.stats import circmean
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


def ExtractNeuronsFeats(filename, N_p=4000, N_i=1000, W=2**13, sim_dt=0.02, montdt=0.2, start_time=200, ws=None, NFFT=None, freq_limit=300, NW=2.5, midfreq_def=105., MTSdict=None, CircPhase=False, save_filename=None):
    
    '''
    Function to extract frequency-domain and oscillations' features from the raw data in addition to individual neurons' features with regard to the detected oscillations; such as firing rates and preferred phases
    
    '''
    
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
    PopRateSig_Full =  np.copy(PopRateSigWhole_Full[int(start_time/sim_dt):])
    IsynP_Pyr = np.copy(f.root.IsynP_Pyr.read()[int(start_time/montdt):, :])
    IsynI_Pyr = np.copy(f.root.IsynI_Pyr.read()[int(start_time/montdt):, :])
    IsynP_Int = np.copy(f.root.IsynP_Int.read()[int(start_time/montdt):, :])
    IsynI_Int = np.copy(f.root.IsynI_Int.read()[int(start_time/montdt):, :])
    IsynPmean_Pyr = np.mean(IsynP_Pyr, axis=0)
    IsynImean_Pyr = np.mean(IsynI_Pyr, axis=0)
    IsynPmean_Int = np.mean(IsynP_Int, axis=0)
    IsynImean_Int = np.mean(IsynI_Int, axis=0)
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
    
    if MTSdict is None:
        # Pyr.
        MTS_Pyr, freq_vect = comp_mtspectrogram(PopRateSig_Pyr, fs=fs, freq_limit=freq_limit, W=W, PlotFlag=False)
        # Int.
        MTS_Int, freq_vect = comp_mtspectrogram(PopRateSig_Int, fs=fs, freq_limit=freq_limit, W=W, PlotFlag=False)
        # Full.
        MTS_Full, freq_vect = comp_mtspectrogram(PopRateSig_Full, fs=fs, freq_limit=freq_limit, W=W, PlotFlag=False)
    else:
        MTS_Pyr = MTSdict['RateMTS_Pyr']
        MTS_Int = MTSdict['RateMTS_Int']
        MTS_Full = MTSdict['RateMTS_Full']
        MTS_time = MTSdict['MTS_time']
        freq_vect = MTSdict['freq_vect']
        


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
    HighModeInds_Pyr = np.array([(f,t) for f,t in MaxInds_Pyr if freq_vect[f]>=midfreq])
    HighModeInds_Int = np.array([(f,t) for f,t in MaxInds_Int if freq_vect[f]>=midfreq])
    
    NeuronFRs_Pyr = np.zeros([N_p])
    NeuronFRs_Int = np.zeros([N_i])
    SpikeTrains_Pyr = []
    SpikeTrains_Int = []
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
    NeuronPreSpkIsynP_Pyr = {'PING':[], 'ING':[]}
    NeuronPreSpkIsynI_Pyr = {'PING':[], 'ING':[]}
    NeuronPreSpkIsynP_Int = {'PING':[], 'ING':[]}
    NeuronPreSpkIsynI_Int = {'PING':[], 'ING':[]}

    NeuronFRsModes_Pyr = {'PING':np.zeros([N_p]), 'ING':np.zeros([N_p])}
    NeuronFRsModes_Int = {'PING':np.zeros([N_i]), 'ING':np.zeros([N_i])}
    
    if CircPhase:
        NeuronPhasesC_Pyr = {'PING':[], 'ING':[]}
        NeuronPhasesC_Int = {'PING':[], 'ING':[]}
        PrefPhase_Pyr = {'PING':np.zeros([N_p]), 'ING':np.zeros([N_p])}
        PrefPhase_Int = {'PING':np.zeros([N_i]), 'ING':np.zeros([N_i])}

    
    for n in range(N_p):
        NeuronPhases_Pyr['PING'].append([])
        NeuronPhases_Pyr['ING'].append([])
        NeuronPreSpkIsynP_Pyr['PING'].append([])
        NeuronPreSpkIsynP_Pyr['ING'].append([])
        NeuronPreSpkIsynI_Pyr['PING'].append([])
        NeuronPreSpkIsynI_Pyr['ING'].append([])
        if n<N_i:
            NeuronPhases_Int['PING'].append([])
            NeuronPhases_Int['ING'].append([])
            NeuronPreSpkIsynP_Int['PING'].append([])
            NeuronPreSpkIsynP_Int['ING'].append([])
            NeuronPreSpkIsynI_Int['PING'].append([])
            NeuronPreSpkIsynI_Int['ING'].append([])
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
            spk_I_ind = np.argmin(np.abs(time_v[t]-timeI_v))
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360.
            else:
                if t>burstpeaks[-1]:
                    period = np.float(burstpeaks[-1]-burstpeaks[-2])
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360.
            if t-period >= 0:
                prespk_I_ind = np.argmin(np.abs(time_v[int(t-period)]-timeI_v))
            else:
                prespk_I_ind = 0
            NeuronPhases_Pyr['PING'][:int(N_p/2.)][n].append(phase%360.)
            NeuronPreSpkIsynP_Pyr['PING'][:int(N_p / 2.)][n].append(np.mean(IsynP_Pyr[prespk_I_ind:spk_I_ind, n]))
            NeuronPreSpkIsynI_Pyr['PING'][:int(N_p / 2.)][n].append(np.mean(IsynI_Pyr[prespk_I_ind:spk_I_ind, n]))
        NeuronFRsModes_Pyr['PING'][:int(N_p/2.)] += np.sum(spktrn_pyr1, axis=0)

        spktrn_pyr2 = np.copy(SpikeTrainSigs_Pyr2[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_pyr2, neurons_pyr2 = np.where(spktrn_pyr2==1)
        for t,n in zip(times_pyr2,neurons_pyr2):
            spk_I_ind = np.argmin(np.abs(time_v[t] - timeI_v))
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360.
            else:
                if t>burstpeaks[-1]:
                    period = np.float(burstpeaks[-1]-burstpeaks[-2])
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360.
            if t-period >= 0:
                prespk_I_ind = np.argmin(np.abs(time_v[int(t-period)]-timeI_v))
            else:
                prespk_I_ind = 0
            NeuronPhases_Pyr['PING'][int(N_p/2.):][n].append(phase%360.)
            NeuronPreSpkIsynP_Pyr['PING'][int(N_p / 2.):][n].append(np.mean(IsynP_Pyr[prespk_I_ind:spk_I_ind, int(n+(N_p/2))]))
            NeuronPreSpkIsynI_Pyr['PING'][int(N_p / 2.):][n].append(np.mean(IsynI_Pyr[prespk_I_ind:spk_I_ind, int(n+(N_p/2))]))
        NeuronFRsModes_Pyr['PING'][int(N_p/2.):] += np.sum(spktrn_pyr2, axis=0)

        spktrn_int = np.copy(SpikeTrainSigs_Int[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_int, neurons_int = np.where(spktrn_int==1)
        for t,n in zip(times_int,neurons_int):
            spk_I_ind = np.argmin(np.abs(time_v[t] - timeI_v))
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360.
            else:
                if t>burstpeaks[-1]:
                    period = np.float(burstpeaks[-1]-burstpeaks[-2])
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360.
            if t-period >= 0:
                prespk_I_ind = np.argmin(np.abs(time_v[int(t-period)]-timeI_v))
            else:
                prespk_I_ind = 0
            NeuronPhases_Int['PING'][n].append(phase%360.)
            NeuronPreSpkIsynP_Int['PING'][n].append(np.mean(IsynP_Int[prespk_I_ind:spk_I_ind, n]))
            NeuronPreSpkIsynI_Int['PING'][n].append(np.mean(IsynI_Int[prespk_I_ind:spk_I_ind, n]))
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
            spk_I_ind = np.argmin(np.abs(time_v[t]-timeI_v))
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360.
            else:
                if t>burstpeaks[-1]:
                    period = np.float(burstpeaks[-1]-burstpeaks[-2])
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360.
            if t-period >= 0:
                prespk_I_ind = np.argmin(np.abs(time_v[int(t-period)]-timeI_v))
            else:
                prespk_I_ind = 0
            NeuronPhases_Pyr['ING'][:int(N_p/2.)][n].append(phase%360.)
            NeuronPreSpkIsynP_Pyr['ING'][:int(N_p / 2.)][n].append(np.mean(IsynP_Pyr[prespk_I_ind:spk_I_ind, n]))
            NeuronPreSpkIsynI_Pyr['ING'][:int(N_p / 2.)][n].append(np.mean(IsynI_Pyr[prespk_I_ind:spk_I_ind, n]))
        NeuronFRsModes_Pyr['ING'][:int(N_p/2.)] += np.sum(spktrn_pyr1, axis=0)

        spktrn_pyr2 = np.copy(SpikeTrainSigs_Pyr2[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_pyr2, neurons_pyr2 = np.where(spktrn_pyr2==1)
        for t,n in zip(times_pyr2,neurons_pyr2):
            spk_I_ind = np.argmin(np.abs(time_v[t]-timeI_v))
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360.
            else:
                if t>burstpeaks[-1]:
                    period = np.float(burstpeaks[-1]-burstpeaks[-2])
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360.
            if t-period >= 0:
                prespk_I_ind = np.argmin(np.abs(time_v[int(t-period)]-timeI_v))
            else:
                prespk_I_ind = 0
            NeuronPhases_Pyr['ING'][int(N_p/2.):][n].append(phase%360.)
            NeuronPreSpkIsynP_Pyr['ING'][int(N_p / 2.):][n].append(np.mean(IsynP_Pyr[prespk_I_ind:spk_I_ind, int(n+(N_p/2))]))
            NeuronPreSpkIsynI_Pyr['ING'][int(N_p / 2.):][n].append(np.mean(IsynI_Pyr[prespk_I_ind:spk_I_ind, int(n+(N_p/2))]))
        NeuronFRsModes_Pyr['ING'][int(N_p/2.):] += np.sum(spktrn_pyr2, axis=0)

        spktrn_int = np.copy(SpikeTrainSigs_Int[int(raw_ind-W/2):int(raw_ind+W/2),:])
        times_int, neurons_int = np.where(spktrn_int==1)
        for t,n in zip(times_int,neurons_int):
            spk_I_ind = np.argmin(np.abs(time_v[t]-timeI_v))
            if t<=burstpeaks[0]:
                period = np.float(burstpeaks[1]-burstpeaks[0])
                phase = ((period-(burstpeaks[0]-t))/period)*360.
            else:
                if t>burstpeaks[-1]:
                    period = np.float(burstpeaks[-1]-burstpeaks[-2])
                else:
                    period = np.float(burstpeaks[burstpeaks>=t][0]-burstpeaks[burstpeaks<t][-1])
                phase = ((t-burstpeaks[burstpeaks<t][-1])/period)*360.
            if t-period >= 0:
                prespk_I_ind = np.argmin(np.abs(time_v[int(t-period)]-timeI_v))
            else:
                prespk_I_ind = 0
            NeuronPhases_Int['ING'][n].append(phase%360.)
            NeuronPreSpkIsynP_Int['ING'][n].append(np.mean(IsynP_Int[prespk_I_ind:spk_I_ind, n]))
            NeuronPreSpkIsynI_Int['ING'][n].append(np.mean(IsynI_Int[prespk_I_ind:spk_I_ind, n]))
        NeuronFRsModes_Int['ING'] += np.sum(spktrn_int, axis=0)
        
    NeuronFRsModes_Pyr['ING'] /= np.float(W*len(HighModeInds_Int)*sim_dt)/1000.
    NeuronFRsModes_Int['ING'] /= np.float(W*len(HighModeInds_Int)*sim_dt)/1000.
    
    for n in range(N_p):
        phasesLMc = np.copy(np.array(NeuronPhases_Pyr['PING'][n]))
        phasesLMc[phasesLMc>180.] -= 360.
        NeuronPhasesC_Pyr['PING'][n] = phasesLMc
        if len(phasesLMc)==0:
            PrefPhase_Pyr['PING'][n] = np.float('nan')
        else:
            PrefPhase_Pyr['PING'][n] = circmean(phasesLMc*(np.pi/180.))*(180./np.pi)
        phasesHMc = np.copy(np.array(NeuronPhases_Pyr['ING'][n]))
        phasesHMc[phasesHMc>180.] -= 360.
        NeuronPhasesC_Pyr['ING'][n] = phasesHMc
        if len(phasesHMc)==0:
            PrefPhase_Pyr['ING'][n] = np.float('nan')
        else:
            PrefPhase_Pyr['ING'][n] = circmean(phasesHMc*(np.pi/180.))*(180./np.pi)
        if n<N_i:
            phasesLMc = np.copy(np.array(NeuronPhases_Int['PING'][n]))
            phasesLMc[phasesLMc>180.] -= 360.
            NeuronPhasesC_Int['PING'][n] = phasesLMc
            if len(phasesLMc)==0:
                PrefPhase_Int['PING'][n] = np.float('nan')
            else:
                PrefPhase_Int['PING'][n] = circmean(phasesLMc*(np.pi/180.))*(180./np.pi)
            phasesHMc = np.copy(np.array(NeuronPhases_Int['ING'][n]))
            phasesHMc[phasesHMc>180.] -= 360.
            NeuronPhasesC_Int['ING'][n] = phasesHMc
            if len(phasesHMc)==0:
                PrefPhase_Int['ING'][n] = np.float('nan')
            else:
                PrefPhase_Int['ING'][n] = circmean(phasesHMc*(np.pi/180.))*(180./np.pi) 
                
    MTS_Results = {'RateMTS_Pyr':MTS_Pyr,
                  'RateMTS_Int':MTS_Int,
                  'MaxInds_Pyr':MaxInds_Pyr,
                  'MaxInds_Int':MaxInds_Int,
                  'FreqMidPnt':midfreq,
                  'WinningFreq':winfreq,
                  'freq_vect':freq_vect,
                  'MTS_time':MTS_time,
                  'LowModeInds_Pyr':LowModeInds_Pyr,
                  'LowModeInds_Int':LowModeInds_Int,
                  'HighModeInds_Pyr':HighModeInds_Pyr,
                  'HighModeInds_Int':HighModeInds_Int}
    
    SpkTrains = {'SpikeTrains_Pyr':SpikeTrains_Pyr,
                 'SpikeTrains_Int':SpikeTrains_Int}
    
    NeuronsFeats = {'NeuronFRs_Pyr': NeuronFRs_Pyr,
                    'NeuronFRs_Int': NeuronFRs_Int,
                    'NeuronFRsModes_Pyr': NeuronFRsModes_Pyr,
                    'NeuronFRsModes_Int': NeuronFRsModes_Int,
                    'NeuronPhases_Pyr': NeuronPhases_Pyr,
                    'NeuronPhases_Int': NeuronPhases_Int,
                    'NeuronPreSpkIsynP_Pyr': NeuronPreSpkIsynP_Pyr,
                    'NeuronPreSpkIsynI_Pyr': NeuronPreSpkIsynI_Pyr,
                    'NeuronPreSpkIsynP_Int': NeuronPreSpkIsynP_Int,
                    'NeuronPreSpkIsynI_Int': NeuronPreSpkIsynI_Int,
                    'IsynP_Pyr': IsynPmean_Pyr,
                    'IsynI_Pyr': IsynImean_Pyr,
                    'IsynP_Int': IsynPmean_Int,
                    'IsynI_Int': IsynImean_Int}
    
    if CircPhase:
        NeuronsFeats['NeuronPhasesC_Pyr'] = NeuronPhasesC_Pyr
        NeuronsFeats['NeuronPhasesC_Int'] = NeuronPhasesC_Int
        NeuronsFeats['PrefPhase_Pyr'] = PrefPhase_Pyr
        NeuronsFeats['PrefPhase_Int'] = PrefPhase_Int
    
    if not save_filename is None:
        with open(save_filename, 'wb') as f:
            pickle.dump({'MTS_Results':MTS_Results,'SpkTrains':SpkTrains,'NeuronsFeats':NeuronsFeats}, f)
    
    return MTS_Results, SpkTrains, NeuronsFeats

############################################################################

def StackAllPhasesHists(NeuronPhases_Pyr, NeuronPhases_Int, bins_vect, groups_pyr=None, groups_int=None, N_p=4000, N_i=1000):
    
    if groups_pyr is None:
        groups_pyr=[range(N_p)]
        
    NeuronPhasesHists_Pyr = {'ING': np.zeros([N_p, len(bins_vect)-1]), 'PING': np.zeros([N_p, len(bins_vect)-1])}

    for gi,group in enumerate(groups_pyr):
        for ni in range(len(group)):

            if gi>0:
                nind = ni+gi*len(groups_pyr[gi-1])
            else:
                nind = ni

            h,bins = histogram(NeuronPhases_Pyr['ING'][group[ni]], bins_vect)
            NeuronPhasesHists_Pyr['ING'][nind,:] = h/np.sum(h).astype(np.float)

            h,bins = histogram(NeuronPhases_Pyr['PING'][group[ni]], bins_vect)
            NeuronPhasesHists_Pyr['PING'][nind,:] = h/np.sum(h).astype(np.float)

    if groups_int is None:
        groups_int=[range(N_i)]
        
    NeuronPhasesHists_Int = {'ING': np.zeros([N_i, len(bins_vect)-1]), 'PING': np.zeros([N_i, len(bins_vect)-1])}

    for gi,group in enumerate(groups_int):
        for ni in range(len(group)):

            if gi>0:
                nind = ni+gi*len(groups_int[gi-1])
            else:
                nind = ni

            h,bins = histogram(NeuronPhases_Int['ING'][group[ni]], bins_vect)
            NeuronPhasesHists_Int['ING'][nind,:] = h/np.sum(h).astype(np.float)

            h,bins = histogram(NeuronPhases_Int['PING'][group[ni]], bins_vect)
            NeuronPhasesHists_Int['PING'][nind,:] = h/np.sum(h).astype(np.float)
                
    return NeuronPhasesHists_Pyr, NeuronPhasesHists_Int
    