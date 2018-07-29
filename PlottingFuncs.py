import numpy as np
import matplotlib.pyplot as plt
from random import choice
import tables
from brian2 import *
from spectrum import pmtm
from HelpingFuncs.peakdetect import peakdet

#####################################################################################

def plot_results(IterArray1, IterArray2, mode, Sims_feats, out_file=None):
    
    AvgCellRate_Pyr, SynchFreq_Pyr, SynchFreqPow_Pyr, PkWidth_Pyr, Harmonics_Pyr, SynchMeasure_Pyr, AvgCellRate_Int, SynchFreq_Int, SynchFreqPow_Int, PkWidth_Int, Harmonics_Int, SynchMeasure_Int = Sims_feats['AvgCellRate_Pyr'], Sims_feats['SynchFreq_Pyr'], Sims_feats['SynchFreqPow_Pyr'], Sims_feats['PkWidth_Pyr'], Sims_feats['Harmonics_Pyr'], Sims_feats['SynchMeasure_Pyr'], Sims_feats['AvgCellRate_Int'], Sims_feats['SynchFreq_Int'], Sims_feats['SynchFreqPow_Int'], Sims_feats['PkWidth_Int'], Sims_feats['Harmonics_Int'], Sims_feats['SynchMeasure_Int']
    
    if Sims_feats.has_key('PhaseShift_Pyr') and Sims_feats.has_key('PhaseShift_Int'):
        nrows = 7
        PhaseShift_Pyr, PhaseShift_Int = Sims_feats['PhaseShift_Pyr'], Sims_feats['PhaseShift_Int']    
    else:
        nrows = 6
    ncolumns = 2
    
    figure(figsize=[5*ncolumns,4*nrows])
    
    if mode is 'Homogenous':
        extent_entries = [IterArray1[0]/kHz, IterArray1[-1]/kHz, IterArray2[0]/kHz, IterArray2[-1]/kHz]
        xlabel_txt = 'Pyr. Input (kHz)'
        ylabel_txt = 'Int. Input (kHz)'
    else:
        extent_entries = [IterArray1[0], IterArray1[-1], IterArray2[0]*Hz, IterArray2[-1]*Hz]
        xlabel_txt = 'Inh. Pois. Freq. (Hz)'
        ylabel_txt = 'Inh. Pois. Amplitude'

    SynchFreq_NonNans = np.concatenate([SynchFreq_Pyr[~np.isnan(SynchFreq_Pyr)], SynchFreq_Int[~np.isnan(SynchFreq_Int)]])
    subplot(nrows,ncolumns,1)
    imshow(SynchFreq_Pyr.T, origin='lower', cmap='jet',
           vmin = np.min(SynchFreq_NonNans), vmax = np.max(SynchFreq_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    title('AMPA_dl=1.5ms, GABA_dl=0.5ms (from Pyr.)')
    cb = colorbar()
    cb.set_label('Synch. Freq. (Hz)')

    subplot(nrows,ncolumns,2)
    imshow(SynchFreq_Int.T, origin='lower', cmap='jet',
           vmin = np.min(SynchFreq_NonNans), vmax = np.max(SynchFreq_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    title('AMPA_dl=1.5ms, GABA_dl=0.5ms (from Int.)')
    cb = colorbar()
    cb.set_label('Synch. Freq. (Hz)')

    SynchFreqPow_NonNans = np.concatenate([np.log(SynchFreqPow_Pyr[~np.isnan(SynchFreqPow_Pyr)]), np.log(SynchFreqPow_Int[~np.isnan(SynchFreqPow_Int)])])
    subplot(nrows,ncolumns,3)
    imshow(np.log(SynchFreqPow_Pyr.T), origin='lower', cmap='jet',
           vmin = np.min(SynchFreqPow_NonNans), vmax = np.max(SynchFreqPow_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('Log Power')

    subplot(nrows,ncolumns,4)
    imshow(np.log(SynchFreqPow_Int.T), origin='lower', cmap='jet',
           vmin = np.min(SynchFreqPow_NonNans), vmax = np.max(SynchFreqPow_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('Log Power')

    AvgCellRate_NonNans = np.concatenate([AvgCellRate_Pyr[~np.isnan(AvgCellRate_Pyr)], AvgCellRate_Int[~np.isnan(AvgCellRate_Int)]])
    subplot(nrows,ncolumns,5)
    imshow(AvgCellRate_Pyr.T, origin='lower', cmap='jet',
           vmin = np.min(AvgCellRate_NonNans), vmax = np.max(AvgCellRate_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('Avg. Cell Rate (Hz)')

    subplot(nrows,ncolumns,6)
    imshow(AvgCellRate_Int.T, origin='lower', cmap='jet',
           vmin = np.min(AvgCellRate_NonNans), vmax = np.max(AvgCellRate_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('Avg. Cell Rate (Hz)')

    Harmonics_NonNans = np.concatenate([Harmonics_Pyr[~np.isnan(Harmonics_Pyr)], Harmonics_Int[~np.isnan(Harmonics_Int)]])
    subplot(nrows,ncolumns,7)
    imshow(Harmonics_Pyr.T, origin='lower', cmap='jet',
           vmin = np.min(Harmonics_NonNans), vmax = np.max(Harmonics_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('# of Harmonics')

    subplot(nrows,ncolumns,8)
    imshow(Harmonics_Int.T, origin='lower', cmap='jet',
           vmin = np.min(Harmonics_NonNans), vmax = np.max(Harmonics_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('# of Harmonics')

    PkWidth_NonNans = np.concatenate([PkWidth_Pyr[~np.isnan(PkWidth_Pyr)], PkWidth_Int[~np.isnan(PkWidth_Int)]])
    subplot(nrows,ncolumns,9)
    imshow(PkWidth_Pyr.T, origin='lower', cmap='jet',
           vmin = np.min(PkWidth_NonNans), vmax = np.max(PkWidth_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('Pk Width (Hz)')

    subplot(nrows,ncolumns,10)
    imshow(PkWidth_Int.T, origin='lower', cmap='jet',
           vmin = np.min(PkWidth_NonNans), vmax = np.max(PkWidth_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('Pk Width (Hz)')

    SynchMeasure_NonNans = np.concatenate([SynchMeasure_Pyr[~np.isnan(SynchMeasure_Pyr)], SynchMeasure_Int[~np.isnan(SynchMeasure_Int)]])
    subplot(nrows,ncolumns,11)
    imshow(SynchMeasure_Pyr.T, origin='lower', cmap='jet',
           vmin = np.min(SynchMeasure_NonNans), vmax = np.max(SynchMeasure_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('Synch. Measure')

    subplot(nrows,ncolumns,12)
    imshow(SynchMeasure_Int.T, origin='lower', cmap='jet',
           vmin = np.min(SynchMeasure_NonNans), vmax = np.max(SynchMeasure_NonNans),
           extent=extent_entries)
    xlabel(xlabel_txt)
    ylabel(ylabel_txt)
    cb = colorbar()
    cb.set_label('Synch. Measure')
    
    if not (PhaseShift_Pyr is None):
        PhaseShift_NonNans = np.concatenate([PhaseShift_Pyr[~np.isnan(PhaseShift_Pyr)], PhaseShift_Int[~np.isnan(PhaseShift_Int)]])
        subplot(nrows,ncolumns,13)
        imshow(PhaseShift_Pyr.T, origin='lower', cmap='jet',
               vmin = np.min(PhaseShift_NonNans), vmax = np.max(PhaseShift_NonNans),
               extent=extent_entries)
        xlabel(xlabel_txt)
        ylabel(ylabel_txt)
        cb = colorbar()
        cb.set_label('Phase Shift')
        subplot(nrows,ncolumns,14)
        imshow(PhaseShift_Int.T, origin='lower', cmap='jet',
               vmin = np.min(PhaseShift_NonNans), vmax = np.max(PhaseShift_NonNans),
               extent=extent_entries)
        xlabel(xlabel_txt)
        ylabel(ylabel_txt)
        cb = colorbar()
        cb.set_label('Phase Shift')
        

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

        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr_raw.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int_raw.read()

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
            ii2 = len(PyrInps)-ii-1

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
                subplot(len(PyrInps), len(IntInps), sp_idx+1)
                plot(freq_vect, RateMTS_Pyr)
                xlim(0,300)
                xlabel('Frequency (Hz)')
                ylabel('Power')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))

                figure(2)
                subplot(len(PyrInps), len(IntInps), sp_idx+1)
                plot(freq_vect, RateMTS_Int)
                xlim(0,300)
                xlabel('Frequency (Hz)')
                ylabel('Power')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
                
    else:
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        IPois_As = rawfile.root.IPois_As.read()
        IPois_fs = (rawfile.root.IPois_fs.read())*Hz
        
        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr_raw.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int_raw.read()

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
    
        Spike_t_Pyr_list = rawfile.root.SpikeM_t_Pyr_raw.read()
        Spike_i_Pyr_list = rawfile.root.SpikeM_i_Pyr_raw.read()
        Spike_t_Int_list = rawfile.root.SpikeM_t_Int_raw.read()
        Spike_i_Int_list = rawfile.root.SpikeM_i_Int_raw.read()
    
        rawfile.close()
    
        figure(figsize=[8*len(PyrInps),5*len(IntInps)])

        fmax = (1/(sim_dt))/2

    
        for ii in range(len(IntInps)):
            ii2 = len(PyrInps)-ii-1

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
    
        Spike_t_Pyr_list = rawfile.root.SpikeM_t_Pyr_raw.read()
        Spike_i_Pyr_list = rawfile.root.SpikeM_i_Pyr_raw.read()
        Spike_t_Int_list = rawfile.root.SpikeM_t_Int_raw.read()
        Spike_i_Int_list = rawfile.root.SpikeM_i_Int_raw.read()
    
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
                    xlim(start_time, end_time)                xlabel('Time (ms)')
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

        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr_raw.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int_raw.read()

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
            ii2 = len(PyrInps)-ii-1

            for pi in range(len(PyrInps)):

                idx = pi*len(IntInps)+ii2
                sp_idx = ii*len(PyrInps)+pi
            
                RateSig_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                RateSig_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
            
                time_v = np.linspace(start_time, end_time, len(RateSig_Pyr))
            
                figure(1)
                subplot(len(PyrInps), len(IntInps), sp_idx+1)
                plot(time_v, RateSig_Pyr)
                xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Inst. Pop. Rate')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
            
                figure(2)
                subplot(len(PyrInps), len(IntInps), sp_idx+1)
                plot(time_v, RateSig_Int)
                xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Inst. Pop. Rate')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
                
    else:
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        IPois_As = rawfile.root.IPois_As.read()
        IPois_fs = (rawfile.root.IPois_fs.read())*Hz
        
        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr_raw.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int_raw.read()

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

        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr_raw.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int_raw.read()

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
            ii2 = len(PyrInps)-ii-1

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
                subplot(len(PyrInps), len(IntInps), sp_idx+1)
                imshow(RateMTS_Pyr, origin="lower",extent=[start_time, end_time, freq_vect[0]/Hz, freq_vect[-1]/Hz], aspect="auto", cmap='jet')
                xlabel('Time (ms)')
                ylabel('Frequency (Hz)')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))

                figure(2)
                subplot(len(PyrInps), len(IntInps), sp_idx+1)
                imshow(RateMTS_Int, origin="lower",extent=[start_time, end_time, freq_vect[0]/Hz, freq_vect[-1]/Hz], aspect="auto", cmap='jet')
                xlabel('Time (ms)')
                ylabel('Frequency (Hz)')
                title('Pyr. Inp.: %s, Int. Inp.:%s' % (PyrInps[pi], IntInps[ii2]))
        
    else:
        
        rawfile = tables.open_file(rawfile, mode='r')
    
        IPois_As = rawfile.root.IPois_As.read()
        IPois_fs = (rawfile.root.IPois_fs.read())*Hz
        
        PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr_raw.read()
        PopRateSig_Int_list = rawfile.root.PopRateSig_Int_raw.read()

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