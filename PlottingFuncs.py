import numpy as np
import matplotlib.pyplot as plt
from random import choice
import tables
from brian2 import *
from spectrum import pmtm
from HelpingFuncs.peakdetect import peakdet

#####################################################################################

def plot_results(Sims_feats, IterArray1, IterArray2, mode, out_file=None):
    
    SimsFeats_Pyr = [Sims_feats['SynchFreq_Pyr'], np.log(Sims_feats['SynchFreqPow_Pyr']), Sims_feats['AvgCellRate_Pyr'], Sims_feats['PkWidth_Pyr'], Sims_feats['Harmonics_Pyr'], Sims_feats['SynchMeasure_Pyr']]
    
    SimsFeats_Int = [Sims_feats['SynchFreq_Int'], np.log(Sims_feats['SynchFreqPow_Int']), Sims_feats['AvgCellRate_Int'], Sims_feats['PkWidth_Int'], Sims_feats['Harmonics_Int'], Sims_feats['SynchMeasure_Int']]
    
    SimsFeats_Full = [Sims_feats['SynchFreq_Full'], np.log(Sims_feats['SynchFreqPow_Full']), Sims_feats['AvgCellRate_Full'], Sims_feats['PkWidth_Full'], Sims_feats['Harmonics_Full'], Sims_feats['SynchMeasure_Full'], Sims_feats['PhaseShift_PyrInt']]
    
    labels = ['Synch. Freq. (Hz)', 'Log Power', 'Avg. Cell Rate (Hz)', 'Pk Width (Hz)', '# of Harmonics', 'Synch. Measure', 'PR Phase Shift']
    
    if Sims_feats.has_key('PhaseShiftCurr_Pyr'):
        SimsFeats_Pyr.append(Sims_feats['PhaseShiftCurr_Pyr'])
        SimsFeats_Int.append(Sims_feats['PhaseShiftCurr_Int'])
        SimsFeats_Full.append(Sims_feats['PhaseShiftCurr_Full'])
        labels.append('Syn. Currents Phase Shift')
        
    nrows = len(labels)
    ncolumns = 3
    
    figure(figsize=[6*ncolumns,6*nrows])
    
    if mode is 'Homogenous':
        extent_entries = [IterArray1[0], IterArray1[-1], IterArray2[0], IterArray2[-1]]
        xlabel_txt = 'Pyr. Input (kHz)'
        ylabel_txt = 'Int. Input (kHz)'
    else:
        extent_entries = [IterArray1[0], IterArray1[-1], IterArray2[0]*Hz, IterArray2[-1]*Hz]
        xlabel_txt = 'Inh. Pois. Freq. (Hz)'
        ylabel_txt = 'Inh. Pois. Amplitude'
    
    spind = 1
    for fi in range(len(labels)):
        
        if labels[fi]=='PR Phase Shift':
            feat_nonnans = SimsFeats_Full[fi][~np.isnan(SimsFeats_Full[fi])]
        
        else:
            
            feat_nonnans = np.concatenate([SimsFeats_Pyr[fi][~np.isnan(SimsFeats_Pyr[fi])], SimsFeats_Int[fi][~np.isnan(SimsFeats_Int[fi])], SimsFeats_Full[fi][~np.isnan(SimsFeats_Full[fi])]])
            
            subplot(nrows,ncolumns,spind)
            imshow(SimsFeats_Pyr[fi].T, origin='lower', cmap='jet',
                   vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
                   extent=extent_entries)
            xlabel(xlabel_txt)
            ylabel(ylabel_txt)
            title('AMPA_dl=1.5ms, GABA_dl=0.5ms (from Pyr.)')
            cb = colorbar()
            cb.set_label(labels[fi])

            subplot(nrows,ncolumns,spind+1)
            imshow(SimsFeats_Int[fi].T, origin='lower', cmap='jet',
                   vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
                   extent=extent_entries)
            xlabel(xlabel_txt)
            ylabel(ylabel_txt)
            title('AMPA_dl=1.5ms, GABA_dl=0.5ms (from Int.)')
            cb = colorbar()
            cb.set_label(labels[fi])
        
        subplot(nrows,ncolumns,spind+2)
        imshow(SimsFeats_Full[fi].T, origin='lower', cmap='jet',
               vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
               extent=extent_entries)
        xlabel(xlabel_txt)
        ylabel(ylabel_txt)
        title('AMPA_dl=1.5ms, GABA_dl=0.5ms (from Full)')
        cb = colorbar()
        cb.set_label(labels[fi])
        
        spind += 3

    if not (out_file is None):
        savefig(out_file+'.png')
    show()
    
#####################################################################################

def plot_results_modes(Sims_feats, IterArray1, IterArray2, mode='Homogenous', out_file=None, CurrPhSh=False):
    
    if type(Sims_feats) is str:
        featsf = tables.open_file(Sims_feats, 'r')
        SynchFreq_Pyr = {'PING': featsf.root.SynchFreqPING_Pyr.read(),
                         'ING': featsf.root.SynchFreqING_Pyr.read()}
        SynchFreqPow_Pyr = {'PING': np.log(featsf.root.SynchFreqPowPING_Pyr.read()),
                            'ING': np.log(featsf.root.SynchFreqPowING_Pyr.read())}
        PhaseShift_Pyr = {'PING': featsf.root.PhaseShiftPING_Pyr.read(),
                          'ING': featsf.root.PhaseShiftING_Pyr.read()}
        AvgCellRate_Pyr = featsf.root.AvgCellRate_Pyr.read()
        SynchMeasure_Pyr = featsf.root.SynchMeasure_Pyr.read()
        
        SimsFeats_Pyr = [SynchFreq_Pyr, SynchFreqPow_Pyr, PhaseShift_Pyr, AvgCellRate_Pyr, SynchMeasure_Pyr]
        
        SynchFreq_Int = {'PING': featsf.root.SynchFreqPING_Int.read(),
                         'ING': featsf.root.SynchFreqING_Int.read()}
        SynchFreqPow_Int = {'PING': np.log(featsf.root.SynchFreqPowPING_Int.read()),
                            'ING': np.log(featsf.root.SynchFreqPowING_Int.read())}
        PhaseShift_Int = {'PING': featsf.root.PhaseShiftPING_Int.read(),
                          'ING': featsf.root.PhaseShiftING_Int.read()}
        AvgCellRate_Int = featsf.root.AvgCellRate_Int.read()
        SynchMeasure_Int = featsf.root.SynchMeasure_Int.read()

        SimsFeats_Int = [SynchFreq_Int, SynchFreqPow_Int, PhaseShift_Int, AvgCellRate_Int, SynchMeasure_Int]
        
        labels = ['Synch. Freq. (Hz)', 'Log Power', 'PR Phase Shift', 'Avg. Cell Rate (Hz)', 'Synch. Measure']
        
        if CurrPhSh:
            PhaseShiftCurr_Pyr = {'PING': featsf.root.PhaseShiftCurrPING_Pyr.read(),
                                  'ING': featsf.root.PhaseShiftCurrING_Pyr.read()}
            PhaseShiftCurr_Int = {'PING': featsf.root.PhaseShiftCurrPING_Int.read(),
                                  'ING': featsf.root.PhaseShiftCurrING_Int.read()}
            SimsFeats_Pyr.append(PhaseShiftCurr_Pyr)
            SimsFeats_Int.append(PhaseShiftCurr_Int)
            labels.append('Syn. Currents Phase Shift')
    else:
        
        SynchFreqPow_Pyr = {'PING': np.log(Sims_feats['SynchFreqPow_Pyr']['PING']),
                            'ING': np.log(Sims_feats['SynchFreqPow_Pyr']['ING'])}
        SynchFreqPow_Int = {'PING': np.log(Sims_feats['SynchFreqPow_Int']['PING']),
                            'ING': np.log(Sims_feats['SynchFreqPow_Int']['ING'])}
        
        SimsFeats_Pyr = [Sims_feats['SynchFreq_Pyr'], SynchFreqPow_Pyr, Sims_feats['PhaseShift_Pyr'], Sims_feats['AvgCellRate_Pyr'], Sims_feats['SynchMeasure_Pyr']]

        SimsFeats_Int = [Sims_feats['SynchFreq_Int'], SynchFreqPow_Int, Sims_feats['PhaseShift_Int'], Sims_feats['AvgCellRate_Int'], Sims_feats['SynchMeasure_Int']]
        
        labels = ['Synch. Freq. (Hz)', 'Log Power', 'PR Phase Shift', 'Avg. Cell Rate (Hz)', 'Synch. Measure']
        
        if CurrPhSh:
            SimsFeats_Pyr.append(Sims_feats['PhaseShiftCurr_Pyr'])
            SimsFeats_Int.append(Sims_feats['PhaseShiftCurr_Int'])
            labels.append('Syn. Currents Phase Shift')
    

    nrows = len(labels)
    ncolumns = 4
    
    figure(figsize=[6*ncolumns,6*nrows])
    
    if mode is 'Homogenous':
        extent_entries = [IterArray1[0], IterArray1[-1], IterArray2[0], IterArray2[-1]]
        xlabel_txt = 'Pyr. Input (kHz)'
        ylabel_txt = 'Int. Input (kHz)'
    else:
        extent_entries = [IterArray1[0], IterArray1[-1], IterArray2[0]*Hz, IterArray2[-1]*Hz]
        xlabel_txt = 'Inh. Pois. Freq. (Hz)'
        ylabel_txt = 'Inh. Pois. Amplitude'
    
    spind = 1
    for fi in range(len(labels)):
        
        if fi<=2:
            
            feat_nonnans = np.concatenate([SimsFeats_Pyr[fi]['PING'][~np.isnan(SimsFeats_Pyr[fi]['PING'])],
                                           SimsFeats_Pyr[fi]['ING'][~np.isnan(SimsFeats_Pyr[fi]['ING'])],
                                           SimsFeats_Int[fi]['PING'][~np.isnan(SimsFeats_Int[fi]['PING'])],
                                           SimsFeats_Int[fi]['ING'][~np.isnan(SimsFeats_Int[fi]['ING'])]])
            
            feat_nonnans = np.sort(feat_nonnans)
            feat_nonnans = feat_nonnans[int(0.025*len(feat_nonnans)):int(0.975*len(feat_nonnans))]

            subplot(nrows,ncolumns,spind)
            imshow(SimsFeats_Pyr[fi]['ING'].T, origin='lower', cmap='jet',
                   vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
                   extent=extent_entries)
            xlabel(xlabel_txt)
            ylabel(ylabel_txt)
            title('Pyramidals Population (ING Bursts)')
            cb = colorbar()
            cb.set_label(labels[fi])

            subplot(nrows,ncolumns,spind+1)
            imshow(SimsFeats_Pyr[fi]['PING'].T, origin='lower', cmap='jet',
                   vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
                   extent=extent_entries)
            xlabel(xlabel_txt)
            ylabel(ylabel_txt)
            title('Pyramidals Population (PING Bursts)')
            cb = colorbar()
            cb.set_label(labels[fi])

            subplot(nrows,ncolumns,spind+2)
            imshow(SimsFeats_Int[fi]['ING'].T, origin='lower', cmap='jet',
                   vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
                   extent=extent_entries)
            xlabel(xlabel_txt)
            ylabel(ylabel_txt)
            title('Interneurons Population (ING Bursts)')
            cb = colorbar()
            cb.set_label(labels[fi])

            subplot(nrows,ncolumns,spind+3)
            imshow(SimsFeats_Int[fi]['PING'].T, origin='lower', cmap='jet',
                   vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
                   extent=extent_entries)
            xlabel(xlabel_txt)
            ylabel(ylabel_txt)
            title('Interneurons Population (PING Bursts)')
            cb = colorbar()
            cb.set_label(labels[fi])
            
        else:
            
            feat_nonnans = np.concatenate([SimsFeats_Pyr[fi][~np.isnan(SimsFeats_Pyr[fi])], SimsFeats_Int[fi][~np.isnan(SimsFeats_Int[fi])]])
            
            feat_nonnans = np.sort(feat_nonnans)
            feat_nonnans = feat_nonnans[int(0.025*len(feat_nonnans)):int(0.975*len(feat_nonnans))]
            
            subplot(nrows,ncolumns,spind+1)
            imshow(SimsFeats_Pyr[fi].T, origin='lower', cmap='jet',
                   vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
                   extent=extent_entries)
            xlabel(xlabel_txt)
            ylabel(ylabel_txt)
            title('Pyramidals Population')
            cb = colorbar()
            cb.set_label(labels[fi])
            
            subplot(nrows,ncolumns,spind+3)
            imshow(SimsFeats_Int[fi].T, origin='lower', cmap='jet',
                   vmin = np.min(feat_nonnans), vmax = np.max(feat_nonnans),
                   extent=extent_entries)
            xlabel(xlabel_txt)
            ylabel(ylabel_txt)
            title('Interneurons Population')
            cb = colorbar()
            cb.set_label(labels[fi])

        spind += 4

    if not (out_file is None):
        savefig(out_file+'.png')
    show()
    
#####################################################################################

def plot_mts_grid(rawfile, mode, start_time=None, end_time=None, mts_win='whole', W=2**12, ws=None, sim_dt=0.02, out_file=None):
    
    sim_dt *= ms
    
    if ws is None:
        ws = W/10
    
    if mode is 'Homogenous':
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        PyrInps = (rawfile.root.PyrInps.read()/1000)*kHz
        IntInps = (rawfile.root.IntInps.read()/1000)*kHz

        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int.read()

        rawfile.close()

        figure(1, figsize=[6*len(PyrInps),5*len(IntInps)])
        figure(2, figsize=[6*len(PyrInps),5*len(IntInps)])

        fmax = (1/(sim_dt))/2
        
        runtime = len(PopRateSig_Pyr_list[0])*sim_dt/ms
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = runtime

        if start_time > runtime or end_time > runtime:
            raise ValueError('Please provide start time and end time within the simulation time window!')
    
        for ii in range(len(IntInps)):
            ii2 = len(IntInps)-ii-1

            for pi in range(len(PyrInps)):

                idx = pi*len(IntInps)+ii2
                sp_idx = ii*len(PyrInps)+pi

                RateSig_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Pyr -= np.mean(RateSig_Pyr)

                RateSig_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Int -= np.mean(RateSig_Int)

                if mts_win is 'whole':

                    N = RateSig_Pyr.shape[0]
                    NFFT = 2**(N-1).bit_length()
                    freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                    freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

                    a = pmtm(RateSig_Pyr, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    RateMTS_Pyr = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                    RateMTS_Pyr = RateMTS_Pyr[np.where(freq_vect/Hz<=300)]

                    a = pmtm(RateSig_Int, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    RateMTS_Int = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                    RateMTS_Int = RateMTS_Int[np.where(freq_vect/Hz<=300)]

                else:

                    NFFT=W*2
                    freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                    freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

                    N_segs = int((len(RateSig_Pyr) / ws)-(W-ws)/ws +1)
                    result = np.zeros((NFFT/2))
                    for i in range(N_segs):
                        data = RateSig_Pyr[i*ws:i*ws+W]
                        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                        result += Sks[:NFFT/2]/W
                    RateMTS_Pyr = result[np.where(freq_vect/Hz<=300)]/N_segs

                    N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                    result = np.zeros((NFFT/2))
                    for i in range(N_segs):
                        data = RateSig_Int[i*ws:i*ws+W]
                        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                        result += Sks[:NFFT/2]/W
                    RateMTS_Int = result[np.where(freq_vect/Hz<=300)]/N_segs
            
                figure(1)
                subplot(len(IntInps), len(PyrInps), sp_idx+1)
                plot(freq_vect, RateMTS_Pyr)
                xlim(0,300)
                xlabel('Frequency (Hz)')
                ylabel('Power')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))

                figure(2)
                subplot(len(IntInps), len(PyrInps), sp_idx+1)
                plot(freq_vect, RateMTS_Int)
                xlim(0,300)
                xlabel('Frequency (Hz)')
                ylabel('Power')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
                
    else:
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        IPois_As = rawfile.root.IPois_As.read()
        IPois_fs = (rawfile.root.IPois_fs.read())*Hz
        
        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int.read()

        rawfile.close()

        figure(1, figsize=[6*len(IPois_fs),5*len(IPois_As)])
        figure(2, figsize=[6*len(IPois_fs),5*len(IPois_As)])
    
        fmax = (1/(sim_dt))/2
        
        runtime = len(PopRateSig_Pyr_list[0])*sim_dt/ms
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = runtime

        if start_time > runtime or end_time > runtime:
            raise ValueError('Please provide start time and end time within the simulation time window!')
        
        for pi,IP_A in enumerate(IPois_As):
        
            for ii,IP_f in enumerate(IPois_fs):
                
                idx = pi*len(IPois_fs)+ii
                
                RateSig_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Pyr -= np.mean(RateSig_Pyr)

                RateSig_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Int -= np.mean(RateSig_Int)
                
                if mts_win is 'whole':

                    N = RateSig_Pyr.shape[0]
                    NFFT = 2**(N-1).bit_length()
                    freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                    freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

                    a = pmtm(RateSig_Pyr, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    RateMTS_Pyr = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                    RateMTS_Pyr = RateMTS_Pyr[np.where(freq_vect/Hz<=300)]

                    a = pmtm(RateSig_Int, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    RateMTS_Int = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                    RateMTS_Int = RateMTS_Int[np.where(freq_vect/Hz<=300)]

                else:

                    NFFT=W*2
                    freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                    freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

                    N_segs = int((len(RateSig_Pyr) / ws)-(W-ws)/ws +1)
                    result = np.zeros((NFFT/2))
                    for i in range(N_segs):
                        data = RateSig_Pyr[i*ws:i*ws+W]
                        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                        result += Sks[:NFFT/2]/W
                    RateMTS_Pyr = result[np.where(freq_vect/Hz<=300)]/N_segs

                    N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                    result = np.zeros((NFFT/2))
                    for i in range(N_segs):
                        data = RateSig_Int[i*ws:i*ws+W]
                        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                        result += Sks[:NFFT/2]/W
                    RateMTS_Int = result[np.where(freq_vect/Hz<=300)]/N_segs
                
                figure(1)
                subplot(len(IPois_As), len(IPois_fs), idx+1)
                plot(freq_vect, RateMTS_Pyr)
                xlim(0,300)
                xlabel('Frequency (Hz)')
                ylabel('Power')
                title('IP Amp.: %s, IP Freq.:%s' % (IP_A, IP_f))

                figure(2)
                subplot(len(IPois_As), len(IPois_fs), idx+1)
                plot(freq_vect, RateMTS_Int)
                xlim(0,300)
                xlabel('Frequency (Hz)')
                ylabel('Power')
                title('IP Amp.: %s, IP Freq.:%s' % (IP_A, IP_f))
            
    if not (out_file is None):
        figure(1)
        savefig(out_file+'_Pyr.png')
        figure(2)
        savefig(out_file+'_Int.png')
    
    show()
    
#####################################################################################


def plot_spikes_grid(rawfile, mode, start_time=None, end_time=None, sim_dt=0.02, out_file=None):
    
    sim_dt *= ms
    
    if mode is 'Homogenous':
    
        rawfile = tables.open_file(rawfile, mode='r')
    
        PyrInps = (rawfile.root.PyrInps.read()/1000)*kHz
        IntInps = (rawfile.root.IntInps.read()/1000)*kHz
    
        Spike_t_Pyr_list = rawfile.root.SpikeM_t_Pyr.read()
        Spike_i_Pyr_list = rawfile.root.SpikeM_i_Pyr.read()
        Spike_t_Int_list = rawfile.root.SpikeM_t_Int.read()
        Spike_i_Int_list = rawfile.root.SpikeM_i_Int.read()
    
        rawfile.close()
    
        figure(figsize=[8*len(PyrInps),5*len(IntInps)])

        fmax = (1/(sim_dt))/2

    
        for ii in range(len(IntInps)):
            ii2 = len(IntInps)-ii-1

            for pi in range(len(PyrInps)):

                idx = pi*len(IntInps)+ii2
                sp_idx = ii*len(PyrInps)+pi
            
                Spike_t_Pyr = Spike_t_Pyr_list[idx]
                Spike_i_Pyr = Spike_i_Pyr_list[idx]
                Spike_t_Int = Spike_t_Int_list[idx]
                Spike_i_Int = Spike_i_Int_list[idx]
            
                subplot(len(PyrInps), len(IntInps), sp_idx+1)
                plot(Spike_t_Pyr*1000, Spike_i_Pyr, '.', Spike_t_Int*1000, Spike_i_Int+4000, '.')
                if all((not start_time is None, not end_time is None)):
                    xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Neuron Index')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
                
    else:
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        IPois_As = rawfile.root.IPois_As.read()
        IPois_fs = (rawfile.root.IPois_fs.read())*Hz
    
        Spike_t_Pyr_list = rawfile.root.SpikeM_t_Pyr.read()
        Spike_i_Pyr_list = rawfile.root.SpikeM_i_Pyr.read()
        Spike_t_Int_list = rawfile.root.SpikeM_t_Int.read()
        Spike_i_Int_list = rawfile.root.SpikeM_i_Int.read()
    
        rawfile.close()
        
        figure(figsize=[8*len(IPois_fs),5*len(IPois_As)])
        
        fmax = (1/(sim_dt))/2
        
        for pi,IP_A in enumerate(IPois_As):
        
            for ii,IP_f in enumerate(IPois_fs):
                
                idx = pi*len(IPois_fs)+ii
                
                Spike_t_Pyr = Spike_t_Pyr_list[idx]
                Spike_i_Pyr = Spike_i_Pyr_list[idx]
                Spike_t_Int = Spike_t_Int_list[idx]
                Spike_i_Int = Spike_i_Int_list[idx]
            
                subplot(len(IPois_As), len(IPois_fs), idx+1)
                plot(Spike_t_Pyr*1000, Spike_i_Pyr, '.', Spike_t_Int*1000, Spike_i_Int+4000, '.')
                if all((not start_time is None, not end_time is None)):
                    xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Neuron Index')
                title('IP Amp.: %s, IP Freq.:%s' % (IP_A, IP_f))
            
    if not (out_file is None):
        savefig(out_file+'.png')
    
    show()
    
#####################################################################################


def plot_poprate_grid(rawfile, mode, start_time=None, end_time=None, sim_dt=0.02, out_file=None):
    
    sim_dt *= ms
    
    if mode is 'Homogenous':
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        PyrInps = (rawfile.root.PyrInps.read()/1000)*kHz
        IntInps = (rawfile.root.IntInps.read()/1000)*kHz

        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int.read()

        rawfile.close()

        figure(1, figsize=[8*len(PyrInps),5*len(IntInps)])
        figure(2, figsize=[8*len(PyrInps),5*len(IntInps)])
    
        fmax = (1/(sim_dt))/2
        
        runtime = len(PopRateSig_Pyr_list[0])*sim_dt/ms
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = runtime

        if start_time > runtime or end_time > runtime:
            raise ValueError('Please provide start time and end time within the simulation time window!')
    
        for ii in range(len(IntInps)):
            ii2 = len(IntInps)-ii-1

            for pi in range(len(PyrInps)):

                idx = pi*len(IntInps)+ii2
                sp_idx = ii*len(PyrInps)+pi
            
                RateSig_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
            
                time_v = np.linspace(start_time, end_time, len(RateSig_Pyr))
            
                figure(1)
                subplot(len(IntInps), len(PyrInps), sp_idx+1)
                plot(time_v, RateSig_Pyr)
                xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Inst. Pop. Rate')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
            
                figure(2)
                subplot(len(IntInps), len(PyrInps), sp_idx+1)
                plot(time_v, RateSig_Int)
                xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Inst. Pop. Rate')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
                
    else:
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        IPois_As = rawfile.root.IPois_As.read()
        IPois_fs = (rawfile.root.IPois_fs.read())*Hz
        
        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int.read()

        rawfile.close()

        figure(1, figsize=[6*len(IPois_fs),5*len(IPois_As)])
        figure(2, figsize=[6*len(IPois_fs),5*len(IPois_As)])
    
        fmax = (1/(sim_dt))/2
        
        runtime = len(PopRateSig_Pyr_list[0])*sim_dt/ms
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = runtime

        if start_time > runtime or end_time > runtime:
            raise ValueError('Please provide start time and end time within the simulation time window!')
        
        for pi,IP_A in enumerate(IPois_As):
        
            for ii,IP_f in enumerate(IPois_fs):
                
                idx = pi*len(IPois_fs)+ii
                
                RateSig_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
            
                time_v = np.linspace(start_time, end_time, len(RateSig_Pyr))
            
                figure(1)
                subplot(len(IPois_As), len(IPois_fs), idx+1)
                plot(time_v, RateSig_Pyr)
                xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Inst. Pop. Rate')
                title('IP Amp.: %s, IP Freq.:%s' % (IP_A, IP_f))
            
                figure(2)
                subplot(len(IPois_As), len(IPois_fs), idx+1)
                plot(time_v, RateSig_Int)
                xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Inst. Pop. Rate')
                title('IP Amp.: %s, IP Freq.:%s' % (IP_A, IP_f))
            
    if not (out_file is None):
        figure(1)
        savefig(out_file+'_Pyr.png')
        figure(2)
        savefig(out_file+'_Int.png')
    show()
    
#####################################################################################


def plot_spcgram_grid(rawfile, mode, start_time=None, end_time=None, W=2**12, ws=None, sim_dt=0.02, out_file=None):
    
    sim_dt *= ms
    
    if ws is None:
        ws = W/10
    
    if mode is 'Homogenous':
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        PyrInps = (rawfile.root.PyrInps.read()/1000)*kHz
        IntInps = (rawfile.root.IntInps.read()/1000)*kHz

        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int.read()

        rawfile.close()

        figure(1, figsize=[6*len(PyrInps),5*len(IntInps)])
        figure(2, figsize=[6*len(PyrInps),5*len(IntInps)])
    
        fmax = (1/(sim_dt))/2
        
        runtime = len(PopRateSig_Pyr_list[0])*sim_dt/ms
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = runtime

        if start_time > runtime or end_time > runtime:
            raise ValueError('Please provide start time and end time within the simulation time window!')
    
        for ii in range(len(IntInps)):
            ii2 = len(IntInps)-ii-1

            for pi in range(len(PyrInps)):

                idx = pi*len(IntInps)+ii2
                sp_idx = ii*len(PyrInps)+pi

                RateSig_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Pyr -= np.mean(RateSig_Pyr)

                RateSig_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Int -= np.mean(RateSig_Int)

                NFFT=W*2
                freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

                N_segs = int((len(RateSig_Pyr) / ws)-(W-ws)/ws +1)
                RateMTS_Pyr = np.zeros((NFFT/2, N_segs))
                for i in range(N_segs):
                    data = RateSig_Pyr[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    RateMTS_Pyr[:,i] = Sks[:NFFT/2]/W
                RateMTS_Pyr = np.squeeze(RateMTS_Pyr[np.where(freq_vect/Hz<=300),:])

                N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                RateMTS_Int = np.zeros((NFFT/2, N_segs))
                for i in range(N_segs):
                    data = RateSig_Int[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    RateMTS_Int[:,i] = Sks[:NFFT/2]/W
                RateMTS_Int = np.squeeze(RateMTS_Int[np.where(freq_vect/Hz<=300),:])
            
                figure(1)
                subplot(len(IntInps), len(PyrInps), sp_idx+1)
                imshow(RateMTS_Pyr, origin="lower",extent=[start_time, end_time, freq_vect[0]/Hz, freq_vect[-1]/Hz], aspect="auto", cmap='jet')
                xlabel('Time (ms)')
                ylabel('Frequency (Hz)')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))

                figure(2)
                subplot(len(IntInps), len(PyrInps), sp_idx+1)
                imshow(RateMTS_Int, origin="lower",extent=[start_time, end_time, freq_vect[0]/Hz, freq_vect[-1]/Hz], aspect="auto", cmap='jet')
                xlabel('Time (ms)')
                ylabel('Frequency (Hz)')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
        
    else:
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        IPois_As = rawfile.root.IPois_As.read()
        IPois_fs = (rawfile.root.IPois_fs.read())*Hz
        
        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int.read()

        rawfile.close()

        figure(1, figsize=[6*len(IPois_fs),5*len(IPois_As)])
        figure(2, figsize=[6*len(IPois_fs),5*len(IPois_As)])
    
        fmax = (1/(sim_dt))/2
        
        runtime = len(PopRateSig_Pyr_list[0])*sim_dt/ms
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = runtime

        if start_time > runtime or end_time > runtime:
            raise ValueError('Please provide start time and end time within the simulation time window!')
        
        for pi,IP_A in enumerate(IPois_As):
        
            for ii,IP_f in enumerate(IPois_fs):
                
                idx = pi*len(IPois_fs)+ii
                
                RateSig_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Pyr -= np.mean(RateSig_Pyr)

                RateSig_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Int -= np.mean(RateSig_Int)
                
                NFFT=W*2
                freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

                N_segs = int((len(RateSig_Pyr) / ws)-(W-ws)/ws +1)
                RateMTS_Pyr = np.zeros((NFFT/2, N_segs))
                for i in range(N_segs):
                    data = RateSig_Pyr[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    RateMTS_Pyr[:,i] = Sks[:NFFT/2]/W
                RateMTS_Pyr = np.squeeze(RateMTS_Pyr[np.where(freq_vect/Hz<=300),:])

                N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                RateMTS_Int = np.zeros((NFFT/2, N_segs))
                for i in range(N_segs):
                    data = RateSig_Int[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    RateMTS_Int[:,i] = Sks[:NFFT/2]/W
                RateMTS_Int = np.squeeze(RateMTS_Int[np.where(freq_vect/Hz<=300),:])
                
                figure(1)
                subplot(len(IPois_As), len(IPois_fs), idx+1)
                imshow(RateMTS_Pyr, origin="lower",extent=[start_time, end_time, freq_vect[0]/Hz, freq_vect[-1]/Hz], aspect="auto", cmap='jet')
                xlabel('Time (ms)')
                ylabel('Frequency (Hz)')
                title('IP Amp.: %s, IP Freq.:%s' % (IP_A, IP_f))

                figure(2)
                subplot(len(IPois_As), len(IPois_fs), idx+1)
                imshow(RateMTS_Int, origin="lower",extent=[start_time, end_time, freq_vect[0]/Hz, freq_vect[-1]/Hz], aspect="auto", cmap='jet')
                xlabel('Time (ms)')
                ylabel('Frequency (Hz)')
                title('IP Amp.: %s, IP Freq.:%s' % (IP_A, IP_f))

    if not (out_file is None):
        figure(1)
        savefig(out_file+'_Pyr.png')
        figure(2)
        savefig(out_file+'_Int.png')
    
    show()