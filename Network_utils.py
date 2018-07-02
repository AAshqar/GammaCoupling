import numpy as np
import matplotlib.pyplot as plt
from random import choice
import tables
from brian2 import *
from spectrum import pmtm
from HelpingFuncs.peakdetect import peakdet
import time
import gc

from NeuronsSpecs.NeuronParams import *
from NeuronsSpecs.NeuronsEqs import *

#####################################################################################


def run_network(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInp=1, IntInp=1, PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, sim_dt=0.02, monitored=[], record_vm=False, verbose=True, PlotFlag=False):
    
    runtime *= ms
    sim_dt *= ms
    
    PyrInp *= kHz
    IntInp *= kHz
    
    if verbose:
        print('Setting up the network...')
    
    monitored_Pyr = list(monitored)
    monitored_Int = list(monitored)
    if record_vm:
        monitored_Pyr.append('v_s')
        monitored_Int.append('v')
    
    defaultclock.dt = sim_dt
    
    Pyr_pop = NeuronGroup(N_p, PyrEqs, threshold='v_s>-30*mV', refractory=1.3*ms, method='rk4')
    Int_pop = NeuronGroup(N_i, IntEqs, threshold='v>-30*mV', refractory=1.3*ms, method='rk4')

    SynPP = Synapses(Pyr_pop, Pyr_pop, on_pre=PreEq_AMPA, delay=delay_AMPA)
    SynPP.connect(p=PP_C)
    SynIP = Synapses(Int_pop, Pyr_pop, on_pre=PreEq_GABA, delay=delay_GABA)
    SynIP.connect(p=IP_C)

    SynII = Synapses(Int_pop, Int_pop, on_pre=PreEq_GABA, delay=delay_GABA)
    SynII.connect(p=II_C)
    SynPI = Synapses(Pyr_pop, Int_pop, on_pre=PreEq_AMPA, delay=delay_AMPA)
    SynPI.connect(p=PI_C)

    voltRange = np.arange(-100, -30, 0.1)
    Pyr_pop.v_s = choice(voltRange, N_p)*mV
    Int_pop.v = choice(voltRange, N_i)*mV

    Poiss_AMPA_Pyr = PoissonGroup(N_p, PyrInp)
    SynPoiss_AMPA_Pyr = Synapses(Poiss_AMPA_Pyr, Pyr_pop, on_pre=PreEq_AMPA_pois, delay=delay_AMPA)
    SynPoiss_AMPA_Pyr.connect(j='i')

    Poiss_AMPA_Int = PoissonGroup(N_i, IntInp)
    SynPoiss_AMPA_Int = Synapses(Poiss_AMPA_Int, Int_pop, on_pre=PreEq_AMPA_pois, delay=delay_AMPA)
    SynPoiss_AMPA_Int.connect(j='i')

    SpikeM_Pyr = SpikeMonitor(Pyr_pop)
    PopRateM_Pyr = PopulationRateMonitor(Pyr_pop)

    SpikeM_Int = SpikeMonitor(Int_pop)
    PopRateM_Int = PopulationRateMonitor(Int_pop)
    
    if monitored_Pyr==[]:
        net = Network(Pyr_pop, Int_pop, Poiss_AMPA_Pyr, Poiss_AMPA_Int,
                      SynIP, SynPI, SynII, SynPoiss_AMPA_Pyr,
                      SynPoiss_AMPA_Int, SpikeM_Pyr, PopRateM_Pyr,
                      SpikeM_Int, PopRateM_Int)
    else:
        StateM_Pyr = StateMonitor(Pyr_pop, monitored_Pyr, record=True)
        StateM_Int = StateMonitor(Int_pop, monitored_Int, record=True)
        net = Network(Pyr_pop, Int_pop, Poiss_AMPA_Pyr, Poiss_AMPA_Int,
                      SynIP, SynPI, SynII, SynPoiss_AMPA_Pyr,
                      SynPoiss_AMPA_Int, SpikeM_Pyr, PopRateM_Pyr,
                      SpikeM_Int, PopRateM_Int, StateM_Pyr, StateM_Int)
    
    if verbose:
        print('Running the network...')

    t1 = time.time()
    net.run(runtime)
    t2 = time.time()
    
    if verbose:
        print('Simulating %s took %s...' %(runtime, (t2-t1)*second))
        
    if PlotFlag:
        figure()
        subplot(2,1,1)
        plot(SpikeM_Pyr.t/ms, SpikeM_Pyr.i, '.', SpikeM_Int.t/ms, SpikeM_Int.i+4000, '.')
        xlim(PopRateM_Pyr.t[0]/ms, PopRateM_Pyr.t[-1]/ms)
        subplot(2,1,2)
        plot(PopRateM_Pyr.t/ms, PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms), PopRateM_Int.t/ms, PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
        xlabel('Time (ms)')
        xlim(PopRateM_Pyr.t[0]/ms, PopRateM_Pyr.t[-1]/ms)
        show()
    
    if monitored_Pyr==[]:
        return SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int
    else:
        return SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, StateM_Pyr, StateM_Int
    
#####################################################################################


def run_network_IP(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInp=1, IntInp=1, IPois_A=1., IPois_Atype='ramp', IPois_f=70, PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, sim_dt=0.02, monitored=[], record_vm=False, verbose=True, PlotFlag=False):
    
    runtime *= ms
    sim_dt *= ms
    
    PyrInp *= kHz
    IntInp *= kHz
    
    IPois_f *= Hz
    
    if verbose:
        print('Setting up the network...')
    
    monitored_Pyr = list(monitored)
    monitored_Int = list(monitored)
    if record_vm:
        monitored_Pyr.append('v_s')
        monitored_Int.append('v')
    
    defaultclock.dt = sim_dt
    
    t_vector = np.arange(0, runtime, defaultclock.dt)
    if IPois_Atype is 'ramp':
        IPois_A = np.linspace(0, IPois_A, len(t_vector))
    
    PoissRate_Pyr = TimedArray((1+IPois_A*np.cos(2*np.pi*IPois_f*t_vector))*PyrInp, dt=defaultclock.dt)
    PoissRate_Int = TimedArray((1+IPois_A*np.cos(2*np.pi*IPois_f*t_vector))*IntInp, dt=defaultclock.dt)
    
    Pyr_pop = NeuronGroup(N_p, PyrEqs, threshold='v_s>-30*mV', refractory=1.3*ms, method='rk4')
    Int_pop = NeuronGroup(N_i, IntEqs, threshold='v>-30*mV', refractory=1.3*ms, method='rk4')

    SynPP = Synapses(Pyr_pop, Pyr_pop, on_pre=PreEq_AMPA, delay=delay_AMPA)
    SynPP.connect(p=PP_C)
    SynIP = Synapses(Int_pop, Pyr_pop, on_pre=PreEq_GABA, delay=delay_GABA)
    SynIP.connect(p=IP_C)

    SynII = Synapses(Int_pop, Int_pop, on_pre=PreEq_GABA, delay=delay_GABA)
    SynII.connect(p=II_C)
    SynPI = Synapses(Pyr_pop, Int_pop, on_pre=PreEq_AMPA, delay=delay_AMPA)
    SynPI.connect(p=PI_C)

    voltRange = np.arange(-100, -30, 0.1)
    Pyr_pop.v_s = choice(voltRange, N_p)*mV
    Int_pop.v = choice(voltRange, N_i)*mV

    Poiss_AMPA_Pyr = PoissonGroup(N_p, rates='PoissRate_Pyr(t)')
    SynPoiss_AMPA_Pyr = Synapses(Poiss_AMPA_Pyr, Pyr_pop, on_pre=PreEq_AMPA_pois, delay=delay_AMPA)
    SynPoiss_AMPA_Pyr.connect(j='i')

    Poiss_AMPA_Int = PoissonGroup(N_i, rates='PoissRate_Int(t)')
    SynPoiss_AMPA_Int = Synapses(Poiss_AMPA_Int, Int_pop, on_pre=PreEq_AMPA_pois, delay=delay_AMPA)
    SynPoiss_AMPA_Int.connect(j='i')

    SpikeM_Pyr = SpikeMonitor(Pyr_pop)
    PopRateM_Pyr = PopulationRateMonitor(Pyr_pop)

    SpikeM_Int = SpikeMonitor(Int_pop)
    PopRateM_Int = PopulationRateMonitor(Int_pop)
    
    if monitored_Pyr==[]:
        net = Network(Pyr_pop, Int_pop, Poiss_AMPA_Pyr, Poiss_AMPA_Int,
                      SynIP, SynPI, SynII, SynPoiss_AMPA_Pyr,
                      SynPoiss_AMPA_Int, SpikeM_Pyr, PopRateM_Pyr,
                      SpikeM_Int, PopRateM_Int)
    else:
        StateM_Pyr = StateMonitor(Pyr_pop, monitored_Pyr, record=True)
        StateM_Int = StateMonitor(Int_pop, monitored_Int, record=True)
        net = Network(Pyr_pop, Int_pop, Poiss_AMPA_Pyr, Poiss_AMPA_Int,
                      SynIP, SynPI, SynII, SynPoiss_AMPA_Pyr,
                      SynPoiss_AMPA_Int, SpikeM_Pyr, PopRateM_Pyr,
                      SpikeM_Int, PopRateM_Int, StateM_Pyr, StateM_Int)
    
    if verbose:
        print('Running the network...')

    t1 = time.time()
    net.run(runtime)
    t2 = time.time()
    
    if verbose:
        print('Simulating %s took %s...' %(runtime, (t2-t1)*second))
        
    if PlotFlag:
        figure()
        subplot(2,1,1)
        plot(SpikeM_Pyr.t/ms, SpikeM_Pyr.i, '.', SpikeM_Int.t/ms, SpikeM_Int.i+4000, '.')
        xlim(PopRateM_Pyr.t[0]/ms, PopRateM_Pyr.t[-1]/ms)
        subplot(2,1,2)
        plot(PopRateM_Pyr.t/ms, PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms), PopRateM_Int.t/ms, PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
        xlabel('Time (ms)')
        xlim(PopRateM_Pyr.t[0]/ms, PopRateM_Pyr.t[-1]/ms)
        show()
    
    if monitored_Pyr==[]:
        return SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int
    else:
        return SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, StateM_Pyr, StateM_Int
    
#####################################################################################


def analyze_network(SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, StateM_Pyr=None, StateM_Int=None, comp_phase=False, N_p=4000, N_i=1000, start_time=200, end_time=1000, sim_dt=0.02, mts_win='whole', W=2**12, ws=(2**12)/10, PlotFlag=False):

    sim_dt *= ms
    
    if start_time > PopRateM_Pyr.t[-1]/ms:
        raise ValueError('Please provide start time and end time within the simulation time window!')

    rates_Pyr = np.zeros(N_p)
    spikes_Pyr = np.asarray([n for j,n in enumerate(SpikeM_Pyr.i) if SpikeM_Pyr.t[j]/ms >= start_time and SpikeM_Pyr.t[j]/ms <= end_time])
    for j in range(N_p):
        rates_Pyr[j] = sum(spikes_Pyr==j)/((end_time-start_time)*ms)

    rates_Int = np.zeros(N_i)
    spikes_Int = np.asarray([n for j,n in enumerate(SpikeM_Int.i) if SpikeM_Int.t[j]/ms >= start_time and SpikeM_Int.t[j]/ms <= end_time])
    for j in range(N_i):
        rates_Int[j] = sum(spikes_Int==j)/((end_time-start_time)*ms)
        
    AvgCellRate_Pyr = np.mean(rates_Pyr*Hz)
    AvgCellRate_Int = np.mean(rates_Int*Hz)

    fs = 1/(sim_dt)
    fmax = fs/2
        
    RateSig_Pyr = PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms)[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
    RateSig_Pyr -= np.mean(RateSig_Pyr)
    
    RateSig_Int = PopRateM_Int.smooth_rate(window='gaussian', width=1*ms)[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
    RateSig_Int -= np.mean(RateSig_Int)
    
    if mts_win is 'whole':
        
        N = RateSig_Pyr.shape[0]
        NFFT = 2**(N-1).bit_length()
        freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
        freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

        a = pmtm(RateSig_Pyr, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        RateMTS_Pyr = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
        RateMTS_Pyr = RateMTS_Pyr[np.where(freq_vect/Hz<=300)]

        a = pmtm(RateSig_Int, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        RateMTS_Int = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
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
            result += Sks[:NFFT/2]
        RateMTS_Pyr = result[np.where(freq_vect/Hz<=300)]/N_segs
        
        N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
        result = np.zeros((NFFT/2))
        for i in range(N_segs):
            data = RateSig_Int[i*ws:i*ws+W]
            a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
            Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
            result += Sks[:NFFT/2]    
        RateMTS_Int = result[np.where(freq_vect/Hz<=300)]/N_segs

    if np.max(RateMTS_Pyr)==0:
        SynchFreq_Pyr = float('nan')
        SynchFreqPow_Pyr = float('nan')
        PkWidth_Pyr = float('nan')
        Harmonics_Pyr = float('nan')
        SynchMeasure_Pyr = float('nan')
            
    else:
        SynchFreq_Pyr = freq_vect[np.argmax(RateMTS_Pyr)]
        SynchFreqPow_Pyr = np.max(RateMTS_Pyr)
        
        N_sm = 5
        smoothed = np.convolve(RateMTS_Pyr, np.ones([N_sm])/N_sm, mode='same')
        freq_vect_sm = freq_vect[:len(smoothed)]

        maxInd = np.argmax(RateMTS_Pyr)
        if np.argmax(RateMTS_Pyr) < 5:
            maxInd = np.argmax(RateMTS_Pyr[10:])+10
        
        maxInd_sm = np.argmax(smoothed)
        if np.argmax(smoothed) < 5:
            maxInd_sm = np.argmax(smoothed[10:])+10

        bline = np.mean(RateMTS_Pyr[freq_vect<300*Hz])
        bline_drop = np.max(RateMTS_Pyr)/6
        pk_offset = freq_vect_sm[(np.where(smoothed[maxInd_sm:]<bline_drop)[0][0]+maxInd_sm)]
        if not (np.where(smoothed[:maxInd_sm]<bline)[0]).any():
            maxtab, mintab = peakdet(smoothed, bline/2.)
            if not [minma[0] for minma in mintab if minma[0] < maxInd_sm]:
                pk_onset = freq_vect_sm[0]
            else:
                pk_onset = freq_vect_sm[int([minma[0] for minma in mintab if minma[0] < maxInd_sm][-1])]
        else:
            pk_onset = freq_vect_sm[np.where(smoothed[:maxInd_sm]<bline)[0][-1]]
        PkWidth_Pyr = pk_offset-pk_onset

        maxtab, mintab = peakdet(smoothed, np.max(smoothed)/3.)
        harms = [mxma for mxma in maxtab if (mxma[0]-maxInd_sm > 20 or mxma[0]-maxInd_sm < -20) and mxma[1] >= np.max(smoothed)/2.0]
        harms = np.array(harms)
        rem_inds = []
        for jj in range(len(harms))[:-1]:
            if harms[jj][0] - harms[jj+1][0] > -10:
                rem_inds.append(jj)
        harms = np.delete(harms, rem_inds, axis=0)
        Harmonics_Pyr = len(harms)
            
        corr_sig = np.correlate(RateSig_Pyr/Hz, RateSig_Pyr/Hz, mode='full')
        corr_sig = corr_sig[int(len(corr_sig)/2):]/(np.mean(rates_Pyr)**2)
        maxtab, mintab = peakdet(corr_sig, 0.1)
        if not mintab.any():
            SynchMeasure_Pyr = float('nan')
        else:
            SynchMeasure_Pyr = (corr_sig[0] - mintab[0,1])
                
    ##### Int.:
    
    if np.max(RateMTS_Int)==0:
        SynchFreq_Int = float('nan')
        SynchFreqPow_Int = float('nan')
        PkWidth_Int = float('nan')
        Harmonics_Int = float('nan')
        SynchMeasure_Int = float('nan')
            
    else:
        SynchFreq_Int = freq_vect[np.argmax(RateMTS_Int)]
        SynchFreqPow_Int = np.max(RateMTS_Int)
        
        N_sm = 5
        smoothed = np.convolve(RateMTS_Int, np.ones([N_sm])/N_sm, mode='same')
        freq_vect_sm = freq_vect[:len(smoothed)]

        maxInd = np.argmax(RateMTS_Int)
        if np.argmax(RateMTS_Int) < 5:
            maxInd = np.argmax(RateMTS_Int[10:])+10
        
        maxInd_sm = np.argmax(smoothed)
        if np.argmax(smoothed) < 5:
            maxInd_sm = np.argmax(smoothed[10:])+10

        bline = np.mean(RateMTS_Int[freq_vect<300*Hz])
        bline_drop = np.max(RateMTS_Int)/6
        pk_offset = freq_vect_sm[(np.where(smoothed[maxInd_sm:]<bline_drop)[0][0]+maxInd_sm)]
        if not (np.where(smoothed[:maxInd_sm]<bline)[0]).any():
            maxtab, mintab = peakdet(smoothed, bline/2.)
            if not [minma[0] for minma in mintab if minma[0] < maxInd_sm]:
                pk_onset = freq_vect_sm[0]
            else:
                pk_onset = freq_vect_sm[int([minma[0] for minma in mintab if minma[0] < maxInd_sm][-1])]
        else:
            pk_onset = freq_vect_sm[np.where(smoothed[:maxInd_sm]<bline)[0][-1]]
        PkWidth_Int = pk_offset-pk_onset

        maxtab, mintab = peakdet(smoothed, np.max(smoothed)/3.)
        harms = [mxma for mxma in maxtab if (mxma[0]-maxInd_sm > 20 or mxma[0]-maxInd_sm < -20) and mxma[1] >= np.max(smoothed)/2.0]
        harms = np.array(harms)
        rem_inds = []
        for jj in range(len(harms))[:-1]:
            if harms[jj][0] - harms[jj+1][0] > -10:
                rem_inds.append(jj)
        harms = np.delete(harms, rem_inds, axis=0)
        Harmonics_Int = len(harms)
            
        corr_sig = np.correlate(RateSig_Int/Hz, RateSig_Int/Hz, mode='full')
        corr_sig = corr_sig[int(len(corr_sig)/2):]/(np.mean(rates_Int)**2)
        maxtab, mintab = peakdet(corr_sig, 0.1)
        if not mintab.any():
            SynchMeasure_Int = float('nan')
        else:
            SynchMeasure_Int = (corr_sig[0] - mintab[0,1])
    
    if comp_phase:
        
        # Pyr.:
        I_AMPA = np.mean(StateM_Pyr.IsynP, axis=1)/namp
        I_GABA = np.mean(StateM_Pyr.IsynI, axis=1)/namp
        
        N = I_AMPA.shape[0]
        NFFT = 2**(N-1).bit_length()
        freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
        freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
        a = pmtm(RateSig_Pyr, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
        I_MTS = I_MTS[np.where(freq_vect/Hz<=300)]
        fpeak = freq_vect[np.argmax(I_MTS)]
        
        corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
        phases = np.arange(1-N, N)
        
        PhaseShift_Pyr = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
        
        # Int.:
        I_AMPA = np.mean(StateM_Int.IsynP, axis=1)/namp
        I_GABA = np.mean(StateM_Int.IsynI, axis=1)/namp
        
        N = I_AMPA.shape[0]
        NFFT = 2**(N-1).bit_length()
        freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
        freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
        a = pmtm(RateSig_Pyr, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
        I_MTS = I_MTS[np.where(freq_vect/Hz<=300)]
        fpeak = freq_vect[np.argmax(I_MTS)]
        
        corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
        phases = np.arange(1-N, N)
        
        PhaseShift_Int = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
        
    if PlotFlag:
        figure(figsize=[8,7])
        subplot(2,2,1)
        hist(rates_Pyr, bins=20)
        title('Single Cell Rates Hist. (Pyr)')
        subplot(2,2,2)
        hist(rates_Int, bins=20)
        title('Single Cell Rates Hist. (Int)')
        subplot(2,2,3)
        plot(freq_vect, RateMTS_Pyr)
        xlim(0, 300)
        title('Pop. Spectrum (Pyr)')
        subplot(2,2,4)
        plot(freq_vect, RateMTS_Int)
        xlim(0, 300)
        title('Pop. Spectrum (Int)')
        show()

    if comp_phase:
        return AvgCellRate_Pyr, SynchFreq_Pyr, SynchFreqPow_Pyr, PkWidth_Pyr, Harmonics_Pyr, SynchMeasure_Pyr, PhaseShift_Pyr, AvgCellRate_Int, SynchFreq_Int, SynchFreqPow_Int, PkWidth_Int, Harmonics_Int, SynchMeasure_Int, PhaseShift_Int
    else:
        return AvgCellRate_Pyr, SynchFreq_Pyr, SynchFreqPow_Pyr, PkWidth_Pyr, Harmonics_Pyr, SynchMeasure_Pyr, AvgCellRate_Int, SynchFreq_Int, SynchFreqPow_Int, PkWidth_Int, Harmonics_Int, SynchMeasure_Int
    
#####################################################################################


def comp_mtspectrogram(PopRateM_Pyr, PopRateM_Int, W=2**12, ws=(2**12)/10, start_time=0, end_time=1000, sim_dt=0.02, PlotFlag=True):
    
    sim_dt *= ms
    
    if start_time > PopRateM_Pyr.t[-1]/ms:
        raise ValueError('Please provide start time and end time within the simulation time window!')
    
    RateSig_Pyr = PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms)[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
    RateSig_Pyr -= np.mean(RateSig_Pyr)
    
    RateSig_Int = PopRateM_Int.smooth_rate(window='gaussian', width=1*ms)[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
    RateSig_Int -= np.mean(RateSig_Int)
    
    NFFT = W*2
    fs = 1/(sim_dt)
    fmax = fs/2
    
    freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
    freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
    
    N_segs = int((len(RateSig_Pyr) / ws)-(W-ws)/ws +1)
    result = np.zeros((NFFT/2, N_segs))
    for i in range(N_segs):
        data = RateSig_Pyr[i*ws:i*ws+W]
        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
        result[:,i] = Sks[:NFFT/2]
    RateMTS_Pyr = np.squeeze(result[np.where(freq_vect/Hz<=300),:])
        
    N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
    result = np.zeros((NFFT/2, N_segs))
    for i in range(N_segs):
        data = RateSig_Int[i*ws:i*ws+W]
        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
        result[:,i] = Sks[:NFFT/2]    
    RateMTS_Int = np.squeeze(result[np.where(freq_vect/Hz<=300),:])
    
    if PlotFlag:
        figure(figsize=[10,10])
        subplot(2,1,1)
        imshow(RateMTS_Pyr, origin="lower", extent=[start_time, end_time, freq_vect[0]/Hz, freq_vect[-1]/Hz], aspect="auto", cmap='jet')    
        xlabel('Time (ms)')
        ylabel('Frequency (Hz)')
        title('Spectrogram (Pyr.)')
        subplot(2,1,2)
        imshow(RateMTS_Int, origin="lower", extent=[start_time, end_time, freq_vect[0]/Hz, freq_vect[-1]/Hz], aspect="auto", cmap='jet')    
        xlabel('Time (ms)')
        ylabel('Frequency (Hz)')
        title('Spectrogram (Int.)')
        show()
    
    return RateMTS_Pyr, RateMTS_Int
    
#####################################################################################


def run_multsim(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInps=[0.5,1], IntInps=[0.5,1], PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, start_time=200, end_time=1000, sim_dt=0.02, monitored=[], mon_avg=True, comp_phase=False, record_vm=True, mts_win='whole', W=2**12, ws=(2**12)/10, verbose=True, analyze=True, save_analyzed=False, save_raw=False, filename=None):
    
    N_samples = int((runtime/ms)/(sim_dt/ms))
    
    if filename is None:
        filename = 'new_experiment'
    
    if save_raw:
        rawfile = tables.open_file(filename+'_raw.h5', mode='w', title='RawData')
        root = rawfile.root
        Params = rawfile.create_vlarray(root, 'InpsParams', tables.StringAtom(itemsize=40, shape=()))
        SpikeM_t_Pyr_raw = rawfile.create_vlarray(root, 'SpikeM_t_Pyr_raw', tables.Float64Atom(shape=()))
        SpikeM_i_Pyr_raw = rawfile.create_vlarray(root, 'SpikeM_i_Pyr_raw', tables.Float64Atom(shape=()))
        SpikeM_t_Int_raw = rawfile.create_vlarray(root, 'SpikeM_t_Int_raw', tables.Float64Atom(shape=()))
        SpikeM_i_Int_raw = rawfile.create_vlarray(root, 'SpikeM_i_Int_raw', tables.Float64Atom(shape=()))
        PopRateSig_Pyr_raw = rawfile.create_vlarray(root, 'PopRateSig_Pyr_raw', tables.Float64Atom(shape=()))
        PopRateSig_Int_raw = rawfile.create_vlarray(root, 'PopRateSig_Int_raw', tables.Float64Atom(shape=()))
        rawfile.create_carray(root, "PyrInps", obj=PyrInps)
        rawfile.create_carray(root, "IntInps", obj=IntInps)
        if not monitored==[]:
            for i,var in enumerate(monitored):
                if mon_avg:
                    locals()[var+'_Pyr'] = rawfile.create_vlarray(root, var+'_Pyr', tables.Float64Atom(shape=()))
                    locals()[var+'_Int'] = rawfile.create_vlarray(root, var+'_Int', tables.Float64Atom(shape=()))
                else:
                    locals()[var+'_Pyr'] = rawfile.create_vlarray(root, var+'_Pyr', tables.Float64Atom(shape=(N_samples, N_p)))
                    locals()[var+'_Int'] = rawfile.create_vlarray(root, var+'_Int', tables.Float64Atom(shape=(N_samples, N_i)))
    
        if record_vm:
            Vm_Pyr = rawfile.create_vlarray(root, 'Vm_Pyr', tables.Float64Atom(shape=(N_samples)))
            Vm_Int = rawfile.create_vlarray(root, 'Vm_Int', tables.Float64Atom(shape=(N_samples)))
        
    if analyze:
        AvgCellRate_Pyr = np.zeros((len(PyrInps),len(IntInps)))
        SynchFreq_Pyr = np.zeros_like(AvgCellRate_Pyr)
        SynchFreqPow_Pyr = np.zeros_like(AvgCellRate_Pyr)
        PkWidth_Pyr = np.zeros_like(AvgCellRate_Pyr)
        Harmonics_Pyr = np.zeros_like(AvgCellRate_Pyr)
        SynchMeasure_Pyr = np.zeros_like(AvgCellRate_Pyr)
        PhaseShift_Pyr = np.zeros_like(AvgCellRate_Pyr)
        AvgCellRate_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchFreq_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchFreqPow_Int = np.zeros_like(AvgCellRate_Pyr)
        PkWidth_Int = np.zeros_like(AvgCellRate_Pyr)
        Harmonics_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchMeasure_Int = np.zeros_like(AvgCellRate_Pyr)
        PhaseShift_Int = np.zeros_like(AvgCellRate_Pyr)
    
    for pi,PyrInp in enumerate(PyrInps):
        
        for ii,IntInp in enumerate(IntInps):
            
            if verbose:
                print('[Starting simulation (%d/%d) for (Pyr. Input: %s, Int. Input: %s)...]' % (pi*len(IntInps)+ii+1, len(PyrInps)*len(IntInps), PyrInp, IntInp))
            
            gc.collect()
            
            if monitored==[] and not record_vm:
                SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int = run_network(N_p=N_p, N_i=N_i, PyrEqs=PyrEqs, IntEqs=IntEqs, PreEqAMPA=PreEqAMPA, PreEqGABA=PreEqGABA, PyrInp=PyrInp, IntInp=IntInp, PP_C=PP_C, IP_C=IP_C, II_C=II_C, PI_C=PI_C, runtime=runtime, sim_dt=sim_dt, monitored=[], verbose=verbose, PlotFlag=False)
            else:
                SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, StateM_Pyr, StateM_Int = run_network(N_p=N_p, N_i=N_i, PyrEqs=PyrEqs, IntEqs=IntEqs, PreEqAMPA=PreEqAMPA, PreEqGABA=PreEqGABA, PyrInp=PyrInp, IntInp=IntInp, PP_C=PP_C, IP_C=IP_C, II_C=II_C, PI_C=PI_C, runtime=runtime, sim_dt=sim_dt, monitored=monitored, record_vm=record_vm, verbose=verbose, PlotFlag=False)
            
            if analyze:
                if comp_phase:
                    AvgCellRate_Pyr[pi,ii], SynchFreq_Pyr[pi,ii], SynchFreqPow_Pyr[pi,ii], PkWidth_Pyr[pi,ii], Harmonics_Pyr[pi,ii], SynchMeasure_Pyr[pi,ii], PhaseShift_Pyr[pi,ii], AvgCellRate_Int[pi,ii], SynchFreq_Int[pi,ii], SynchFreqPow_Int[pi,ii], PkWidth_Int[pi,ii], Harmonics_Int[pi,ii], SynchMeasure_Int[pi,ii], PhaseShift_Int[pi,ii] = analyze_network(SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, StateM_Pyr=StateM_Pyr, StateM_Int=StateM_Int, comp_phase=True, N_p=N_p, N_i=N_i, start_time=start_time, end_time=end_time, sim_dt=sim_dt, mts_win=mts_win, W=W, ws=ws)
                else:
                    AvgCellRate_Pyr[pi,ii], SynchFreq_Pyr[pi,ii], SynchFreqPow_Pyr[pi,ii], PkWidth_Pyr[pi,ii], Harmonics_Pyr[pi,ii], SynchMeasure_Pyr[pi,ii], AvgCellRate_Int[pi,ii], SynchFreq_Int[pi,ii], SynchFreqPow_Int[pi,ii], PkWidth_Int[pi,ii], Harmonics_Int[pi,ii], SynchMeasure_Int[pi,ii] = analyze_network(SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, N_p=N_p, N_i=N_i, start_time=start_time, end_time=end_time, sim_dt=sim_dt, mts_win=mts_win, W=W, ws=ws)
                
            if save_raw:
                Params.append(str((PyrInp, IntInp)))
                SpikeM_t_Pyr_raw.append(np.array(SpikeM_Pyr.t/ms)*ms)
                SpikeM_i_Pyr_raw.append(np.array(SpikeM_Pyr.i))
                SpikeM_t_Int_raw.append(np.array(SpikeM_Int.t/ms)*ms)
                SpikeM_i_Int_raw.append(np.array(SpikeM_Int.i))
                PopRateSig_Pyr_raw.append(PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
                PopRateSig_Int_raw.append(PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
                if not monitored==[]:
                    for i,var in enumerate(monitored): 
                        if mon_avg:
                            locals()[var+'_Pyr'].append(np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                            locals()[var+'_Int'].append(np.array(StateM_Int.get_states()[var]).mean(axis=1))
                        else:
                        
                            locals()[var+'_Pyr'].append(np.array(StateM_Pyr.get_states()[var]))
                            locals()[var+'_Int'].append(np.array(StateM_Int.get_states()[var]))
                
                if record_vm:
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,0])
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,-1])
                    Vm_Int.append(StateM_Int.get_states()['v'][:,0])
                    Vm_Int.append(StateM_Int.get_states()['v'][:,-1])
                
            
    if verbose:
        print('***Finished all simulations successfully***')
            
    if save_analyzed:
        with tables.open_file(filename+'_analysis.h5', mode='w', title='Analysis') as h5file:
            root = h5file.root
            h5file.create_carray(root, "AvgCellRate_Pyr", obj=AvgCellRate_Pyr)
            h5file.create_carray(root, "SynchFreq_Pyr", obj=SynchFreq_Pyr)
            h5file.create_carray(root, "SynchFreqPow_Pyr", obj=SynchFreqPow_Pyr)
            h5file.create_carray(root, "PkWidth_Pyr", obj=PkWidth_Pyr)
            h5file.create_carray(root, "Harmonics_Pyr", obj=Harmonics_Pyr)
            h5file.create_carray(root, "SynchMeasure_Pyr", obj=SynchMeasure_Pyr)
            h5file.create_carray(root, "PhaseShift_Pyr", obj=PhaseShift_Pyr)
            h5file.create_carray(root, "AvgCellRate_Int", obj=AvgCellRate_Int)
            h5file.create_carray(root, "SynchFreq_Int", obj=SynchFreq_Int)
            h5file.create_carray(root, "SynchFreqPow_Int", obj=SynchFreqPow_Int)
            h5file.create_carray(root, "PkWidth_Int", obj=PkWidth_Int)
            h5file.create_carray(root, "Harmonics_Int", obj=Harmonics_Int)
            h5file.create_carray(root, "SynchMeasure_Int", obj=SynchMeasure_Int)
            h5file.create_carray(root, "PhaseShift_Int", obj=PhaseShift_Int)
            h5file.create_carray(root, "PyrInps", obj=PyrInps)
            h5file.create_carray(root, "IntInps", obj=IntInps)
            h5file.close()
        if verbose:
            print('Saved analysis results successfully!')
            
    if save_raw:
        if verbose:
            print('Saved raw data successfully!')
        rawfile.close()
    
    if analyze:
        return AvgCellRate_Pyr, SynchFreq_Pyr, SynchFreqPow_Pyr, PkWidth_Pyr, Harmonics_Pyr, SynchMeasure_Pyr, AvgCellRate_Int, SynchFreq_Int, SynchFreqPow_Int, PkWidth_Int, Harmonics_Int, SynchMeasure_Int
    
#####################################################################################


def run_multsim_IP(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInp=1, IntInp=1, IPois_As=[1.], IPois_Atype='ramp', IPois_fs=[70], PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, start_time=200, end_time=1000, sim_dt=0.02, monitored=[], mon_avg=True, record_vm=True, mts_win='whole', W=2**12, ws=(2**12)/10, verbose=True, analyze=True, save_analyzed=False, save_raw=False, filename=None):
    
    N_samples = int((runtime/ms)/(sim_dt/ms))
    
    if filename is None:
        filename = 'new_experiment'
    
    if save_raw:
        rawfile = tables.open_file(filename+'_raw.h5', mode='w', title='RawData')
        root = rawfile.root
        Params = rawfile.create_vlarray(root, 'InpsParams', tables.StringAtom(itemsize=40, shape=()))
        SpikeM_t_Pyr_raw = rawfile.create_vlarray(root, 'SpikeM_t_Pyr_raw', tables.Float64Atom(shape=()))
        SpikeM_i_Pyr_raw = rawfile.create_vlarray(root, 'SpikeM_i_Pyr_raw', tables.Float64Atom(shape=()))
        SpikeM_t_Int_raw = rawfile.create_vlarray(root, 'SpikeM_t_Int_raw', tables.Float64Atom(shape=()))
        SpikeM_i_Int_raw = rawfile.create_vlarray(root, 'SpikeM_i_Int_raw', tables.Float64Atom(shape=()))
        PopRateSig_Pyr_raw = rawfile.create_vlarray(root, 'PopRateSig_Pyr_raw', tables.Float64Atom(shape=()))
        PopRateSig_Int_raw = rawfile.create_vlarray(root, 'PopRateSig_Int_raw', tables.Float64Atom(shape=()))
        rawfile.create_carray(root, "IPois_As", obj=IPois_As)
        rawfile.create_carray(root, "IPois_fs", obj=IPois_fs)
        
        if not monitored==[]:
            for i,var in enumerate(monitored):
                if mon_avg:
                    locals()[var+'_Pyr'] = rawfile.create_vlarray(root, var+'_Pyr', tables.Float64Atom(shape=()))
                    locals()[var+'_Int'] = rawfile.create_vlarray(root, var+'_Int', tables.Float64Atom(shape=()))
                else:
                    locals()[var+'_Pyr'] = rawfile.create_vlarray(root, var+'_Pyr', tables.Float64Atom(shape=(N_samples, N_p)))
                    locals()[var+'_Int'] = rawfile.create_vlarray(root, var+'_Int', tables.Float64Atom(shape=(N_samples, N_i)))
    
        if record_vm:
            Vm_Pyr = rawfile.create_vlarray(root, 'Vm_Pyr', tables.Float64Atom(shape=(N_samples)))
            Vm_Int = rawfile.create_vlarray(root, 'Vm_Int', tables.Float64Atom(shape=(N_samples)))
        
    if analyze:
        AvgCellRate_Pyr = np.zeros((len(PyrInps),len(IntInps)))
        SynchFreq_Pyr = np.zeros_like(AvgCellRate_Pyr)
        SynchFreqPow_Pyr = np.zeros_like(AvgCellRate_Pyr)
        PkWidth_Pyr = np.zeros_like(AvgCellRate_Pyr)
        Harmonics_Pyr = np.zeros_like(AvgCellRate_Pyr)
        SynchMeasure_Pyr = np.zeros_like(AvgCellRate_Pyr)
        PhaseShift_Pyr = np.zeros_like(AvgCellRate_Pyr)
        AvgCellRate_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchFreq_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchFreqPow_Int = np.zeros_like(AvgCellRate_Pyr)
        PkWidth_Int = np.zeros_like(AvgCellRate_Pyr)
        Harmonics_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchMeasure_Int = np.zeros_like(AvgCellRate_Pyr)
        PhaseShift_Int = np.zeros_like(AvgCellRate_Pyr)
    
    for pi,IP_A in enumerate(IPois_As):
        
        for ii,IP_f in enumerate(IPois_fs):
            
            if verbose:
                print('[Starting simulation (%d/%d) for (IPois. Amp.: %s, IPois. Freq.: %s)...]' % (pi*len(IPois_fs)+ii+1, len(IPois_As)*len(IPois_fs), IP_A, IP_f))
            
            gc.collect()
            
            if monitored==[] and not record_vm:
                SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int = run_network_IP(N_p=N_p, N_i=N_i, PyrEqs=PyrEqs, IntEqs=IntEqs, PreEqAMPA=PreEqAMPA, PreEqGABA=PreEqGABA, PyrInp=PyrInp, IntInp=IntInp, IPois_A=IP_A, IPois_Atype=IPois_Atype, IPois_f=IP_f, PP_C=PP_C, IP_C=IP_C, II_C=II_C, PI_C=PI_C, runtime=runtime, sim_dt=sim_dt, monitored=[], verbose=verbose, PlotFlag=False)
            else:
                SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, StateM_Pyr, StateM_Int = run_network_IP(N_p=N_p, N_i=N_i, PyrEqs=PyrEqs, IntEqs=IntEqs, PreEqAMPA=PreEqAMPA, PreEqGABA=PreEqGABA, PyrInp=PyrInp, IntInp=IntInp, IPois_A=IP_A, IPois_Atype=IPois_Atype, IPois_f=IP_f, PP_C=PP_C, IP_C=IP_C, II_C=II_C, PI_C=PI_C, runtime=runtime, sim_dt=sim_dt, monitored=monitored, record_vm=record_vm, verbose=verbose, PlotFlag=False)
            
            if analyze:
                if comp_phase:
                    AvgCellRate_Pyr[pi,ii], SynchFreq_Pyr[pi,ii], SynchFreqPow_Pyr[pi,ii], PkWidth_Pyr[pi,ii], Harmonics_Pyr[pi,ii], SynchMeasure_Pyr[pi,ii], PhaseShift_Pyr[pi,ii], AvgCellRate_Int[pi,ii], SynchFreq_Int[pi,ii], SynchFreqPow_Int[pi,ii], PkWidth_Int[pi,ii], Harmonics_Int[pi,ii], SynchMeasure_Int[pi,ii], PhaseShift_Int[pi,ii] = analyze_network(SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, StateM_Pyr=StateM_Pyr, StateM_Int=StateM_Int, comp_phase=True, N_p=N_p, N_i=N_i, start_time=start_time, end_time=end_time, sim_dt=sim_dt, mts_win=mts_win, W=W, ws=ws)
                else:
                    AvgCellRate_Pyr[pi,ii], SynchFreq_Pyr[pi,ii], SynchFreqPow_Pyr[pi,ii], PkWidth_Pyr[pi,ii], Harmonics_Pyr[pi,ii], SynchMeasure_Pyr[pi,ii], AvgCellRate_Int[pi,ii], SynchFreq_Int[pi,ii], SynchFreqPow_Int[pi,ii], PkWidth_Int[pi,ii], Harmonics_Int[pi,ii], SynchMeasure_Int[pi,ii] = analyze_network(SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, N_p=N_p, N_i=N_i, start_time=start_time, end_time=end_time, sim_dt=sim_dt, mts_win=mts_win, W=W, ws=ws)
                
            if save_raw:
                Params.append(str((PyrInp, IntInp)))
                SpikeM_t_Pyr_raw.append(np.array(SpikeM_Pyr.t/ms)*ms)
                SpikeM_i_Pyr_raw.append(np.array(SpikeM_Pyr.i))
                SpikeM_t_Int_raw.append(np.array(SpikeM_Int.t/ms)*ms)
                SpikeM_i_Int_raw.append(np.array(SpikeM_Int.i))
                PopRateSig_Pyr_raw.append(PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
                PopRateSig_Int_raw.append(PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
                if not monitored==[]:
                    for i,var in enumerate(monitored): 
                        if mon_avg:
                            locals()[var+'_Pyr'].append(np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                            locals()[var+'_Int'].append(np.array(StateM_Int.get_states()[var]).mean(axis=1))
                        else:
                        
                            locals()[var+'_Pyr'].append(np.array(StateM_Pyr.get_states()[var]))
                            locals()[var+'_Int'].append(np.array(StateM_Int.get_states()[var]))
                
                if record_vm:
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,0])
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,-1])
                    Vm_Int.append(StateM_Int.get_states()['v'][:,0])
                    Vm_Int.append(StateM_Int.get_states()['v'][:,-1])
                
            
    if verbose:
        print('***Finished all simulations successfully***')
            
    if save_analyzed:
        with tables.open_file(filename+'_analysis.h5', mode='w', title='Analysis') as h5file:
            root = h5file.root
            h5file.create_carray(root, "AvgCellRate_Pyr", obj=AvgCellRate_Pyr)
            h5file.create_carray(root, "SynchFreq_Pyr", obj=SynchFreq_Pyr)
            h5file.create_carray(root, "SynchFreqPow_Pyr", obj=SynchFreqPow_Pyr)
            h5file.create_carray(root, "PkWidth_Pyr", obj=PkWidth_Pyr)
            h5file.create_carray(root, "Harmonics_Pyr", obj=Harmonics_Pyr)
            h5file.create_carray(root, "SynchMeasure_Pyr", obj=SynchMeasure_Pyr)
            h5file.create_carray(root, "AvgCellRate_Int", obj=AvgCellRate_Int)
            h5file.create_carray(root, "SynchFreq_Int", obj=SynchFreq_Int)
            h5file.create_carray(root, "SynchFreqPow_Int", obj=SynchFreqPow_Int)
            h5file.create_carray(root, "PkWidth_Int", obj=PkWidth_Int)
            h5file.create_carray(root, "Harmonics_Int", obj=Harmonics_Int)
            h5file.create_carray(root, "SynchMeasure_Int", obj=SynchMeasure_Int)
            h5file.create_carray(root, "IPois_As", obj=IPois_As)
            h5file.create_carray(root, "IPois_fs", obj=IPois_fs)
            h5file.close()
        if verbose:
            print('Saved analysis results successfully!')
            
    if save_raw:
        if verbose:
            print('Saved raw data successfully!')
        rawfile.close()
    
    if analyze:
        if comp_phase:
            return AvgCellRate_Pyr, SynchFreq_Pyr, SynchFreqPow_Pyr, PkWidth_Pyr, Harmonics_Pyr, SynchMeasure_Pyr, PhaseShift_Pyr, AvgCellRate_Int, SynchFreq_Int, SynchFreqPow_Int, PkWidth_Int, Harmonics_Int, SynchMeasure_Int, PhaseShift_Int
        else:
            return AvgCellRate_Pyr, SynchFreq_Pyr, SynchFreqPow_Pyr, PkWidth_Pyr, Harmonics_Pyr, SynchMeasure_Pyr, AvgCellRate_Int, SynchFreq_Int, SynchFreqPow_Int, PkWidth_Int, Harmonics_Int, SynchMeasure_Int
    
#####################################################################################


def analyze_raw(filename, mode, N_p=4000, N_i=1000, start_time=200, end_time=1000, sim_dt=0.02, comp_phase=False, mts_win='whole', W=2**12, ws=(2**12)/10, PlotFlag=False, out_file=None):
    
    sim_dt *= ms
    
    rawfile = tables.open_file(filename, mode='r')
    
    if mode is 'Homogenous':
        IterArray1 = (rawfile.root.PyrInps.read()/1000)*kHz
        IterArray2 = (rawfile.root.IntInps.read()/1000)*kHz
    else:
        IterArray1 = rawfile.root.IPois_As.read()
        IterArray2 = (rawfile.root.IPois_fs.read())*Hz
    
    PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr_raw.read()
    PopRateSig_Int_list = rawfile.root.PopRateSig_Int_raw.read()
    
    Spike_t_Pyr_list = rawfile.root.SpikeM_t_Pyr_raw.read()
    Spike_i_Pyr_list = rawfile.root.SpikeM_i_Pyr_raw.read()
    Spike_t_Int_list = rawfile.root.SpikeM_t_Int_raw.read()
    Spike_i_Int_list = rawfile.root.SpikeM_i_Int_raw.read()
    
    rawfile.close()
        
    AvgCellRate_Pyr = np.zeros((len(PyrInps),len(IntInps)))
    SynchFreq_Pyr = np.zeros_like(AvgCellRate_Pyr)
    SynchFreqPow_Pyr = np.zeros_like(AvgCellRate_Pyr)
    PkWidth_Pyr = np.zeros_like(AvgCellRate_Pyr)
    Harmonics_Pyr = np.zeros_like(AvgCellRate_Pyr)
    SynchMeasure_Pyr = np.zeros_like(AvgCellRate_Pyr)
    PhaseShift_Pyr = np.zeros_like(AvgCellRate_Pyr)
    AvgCellRate_Int = np.zeros_like(AvgCellRate_Pyr)
    SynchFreq_Int = np.zeros_like(AvgCellRate_Pyr)
    SynchFreqPow_Int = np.zeros_like(AvgCellRate_Pyr)
    PkWidth_Int = np.zeros_like(AvgCellRate_Pyr)
    Harmonics_Int = np.zeros_like(AvgCellRate_Pyr)
    SynchMeasure_Int = np.zeros_like(AvgCellRate_Pyr)
    PhaseShift_Int = np.zeros_like(AvgCellRate_Pyr)
    
    for pi,IterItem1 in enumerate(IterArray1):
        for ii,IterItem2 in enumerate(IterArray2):
            
            idx = pi*len(IterArray2)+ii
            
            RateSig_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
            RateSig_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]

            Spike_t_Pyr = Spike_t_Pyr_list[idx]
            Spike_i_Pyr = Spike_i_Pyr_list[idx]
            Spike_t_Int = Spike_t_Int_list[idx]
            Spike_i_Int = Spike_i_Int_list[idx]
            
            if int((start_time*ms)/sim_dt) > len(RateSig_Pyr):
                raise ValueError('Please provide start time and end time within the simulation time window!')

            rates_Pyr = np.zeros(N_p)
            spikes_Pyr = np.asarray([n for j,n in enumerate(Spike_i_Pyr) if Spike_t_Pyr[j]/(0.001) >= start_time and Spike_t_Pyr[j]/(0.001) <= end_time])
            for j in range(N_p):
                rates_Pyr[j] = sum(spikes_Pyr==j)/((end_time-start_time)*ms)

            rates_Int = np.zeros(N_i)
            spikes_Int = np.asarray([n for j,n in enumerate(Spike_i_Int) if Spike_t_Int[j]/(0.001) >= start_time and Spike_t_Int[j]/(0.001) <= end_time])
            for j in range(N_i):
                rates_Int[j] = sum(spikes_Int==j)/((end_time-start_time)*ms)
        
            AvgCellRate_Pyr[pi,ii] = np.mean(rates_Pyr*Hz)
            AvgCellRate_Int[pi,ii] = np.mean(rates_Int*Hz)

            fs = 1/(sim_dt)
            fmax = fs/2

            RateSig_Pyr -= np.mean(RateSig_Pyr)

            RateSig_Int -= np.mean(RateSig_Int)
    
            if mts_win is 'whole':

                N = RateSig_Pyr.shape[0]
                NFFT = 2**(N-1).bit_length()
                freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

                a = pmtm(RateSig_Pyr, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                RateMTS_Pyr = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
                RateMTS_Pyr = RateMTS_Pyr[np.where(freq_vect/Hz<=300)]

                a = pmtm(RateSig_Int, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                RateMTS_Int = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
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
                    result += Sks[:NFFT/2]
                RateMTS_Pyr = result[np.where(freq_vect/Hz<=300)]/N_segs

                N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                result = np.zeros((NFFT/2))
                for i in range(N_segs):
                    data = RateSig_Int[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    result += Sks[:NFFT/2]    
                RateMTS_Int = result[np.where(freq_vect/Hz<=300)]/N_segs

            if np.max(RateMTS_Pyr)==0:
                SynchFreq_Pyr[pi,ii] = float('nan')
                SynchFreqPow_Pyr[pi,ii] = float('nan')
                PkWidth_Pyr[pi,ii] = float('nan')
                Harmonics_Pyr[pi,ii] = float('nan')
                SynchMeasure_Pyr[pi,ii] = float('nan')
            
            else:
                SynchFreq_Pyr[pi,ii] = freq_vect[np.argmax(RateMTS_Pyr)]
                SynchFreqPow_Pyr[pi,ii] = np.max(RateMTS_Pyr)

                N_sm = 5
                smoothed = np.convolve(RateMTS_Pyr, np.ones([N_sm])/N_sm, mode='same')
                freq_vect_sm = freq_vect[:len(smoothed)]

                maxInd = np.argmax(RateMTS_Pyr)
                if np.argmax(RateMTS_Pyr) < 5:
                    maxInd = np.argmax(RateMTS_Pyr[10:])+10

                maxInd_sm = np.argmax(smoothed)
                if np.argmax(smoothed) < 5:
                    maxInd_sm = np.argmax(smoothed[10:])+10

                bline = np.mean(RateMTS_Pyr[freq_vect<300*Hz])
                bline_drop = np.max(RateMTS_Pyr)/6
                pk_offset = freq_vect_sm[(np.where(smoothed[maxInd_sm:]<bline_drop)[0][0]+maxInd_sm)]
                if not (np.where(smoothed[:maxInd_sm]<bline)[0]).any():
                    maxtab, mintab = peakdet(smoothed, bline/2.)
                    if not [minma[0] for minma in mintab if minma[0] < maxInd_sm]:
                        pk_onset = freq_vect_sm[0]
                    else:
                        pk_onset = freq_vect_sm[int([minma[0] for minma in mintab if minma[0] < maxInd_sm][-1])]
                else:
                    pk_onset = freq_vect_sm[np.where(smoothed[:maxInd_sm]<bline)[0][-1]]
                PkWidth_Pyr[pi,ii] = pk_offset-pk_onset

                maxtab, mintab = peakdet(smoothed, np.max(smoothed)/3.)
                harms = [mxma for mxma in maxtab if (mxma[0]-maxInd_sm > 20 or mxma[0]-maxInd_sm < -20) and mxma[1] >= np.max(smoothed)/2.0]
                harms = np.array(harms)
                rem_inds = []
                for jj in range(len(harms))[:-1]:
                    if harms[jj][0] - harms[jj+1][0] > -10:
                        rem_inds.append(jj)
                harms = np.delete(harms, rem_inds, axis=0)
                Harmonics_Pyr[pi,ii] = len(harms)

                corr_sig = np.correlate(RateSig_Pyr, RateSig_Pyr, mode='full')
                corr_sig = corr_sig[int(len(corr_sig)/2):]/(np.mean(rates_Pyr)**2)
                maxtab, mintab = peakdet(corr_sig, 0.1)
                if not mintab.any():
                    SynchMeasure_Pyr[pi,ii] = float('nan')
                else:
                    SynchMeasure_Pyr[pi,ii] = (corr_sig[0] - mintab[0,1])
                
            ##### Int.:

            if np.max(RateMTS_Int)==0:
                SynchFreq_Int[pi,ii] = float('nan')
                SynchFreqPow_Int[pi,ii] = float('nan')
                PkWidth_Int[pi,ii] = float('nan')
                Harmonics_Int[pi,ii] = float('nan')
                SynchMeasure_Int[pi,ii] = float('nan')
            
            else:
                SynchFreq_Int[pi,ii] = freq_vect[np.argmax(RateMTS_Int)]
                SynchFreqPow_Int[pi,ii] = np.max(RateMTS_Int)

                N_sm = 5
                smoothed = np.convolve(RateMTS_Int, np.ones([N_sm])/N_sm, mode='same')
                freq_vect_sm = freq_vect[:len(smoothed)]

                maxInd = np.argmax(RateMTS_Int)
                if np.argmax(RateMTS_Int) < 5:
                    maxInd = np.argmax(RateMTS_Int[10:])+10

                maxInd_sm = np.argmax(smoothed)
                if np.argmax(smoothed) < 5:
                    maxInd_sm = np.argmax(smoothed[10:])+10

                bline = np.mean(RateMTS_Int[freq_vect<300*Hz])
                bline_drop = np.max(RateMTS_Int)/6
                pk_offset = freq_vect_sm[(np.where(smoothed[maxInd_sm:]<bline_drop)[0][0]+maxInd_sm)]
                if not (np.where(smoothed[:maxInd_sm]<bline)[0]).any():
                    maxtab, mintab = peakdet(smoothed, bline/2.)
                    if not [minma[0] for minma in mintab if minma[0] < maxInd_sm]:
                        pk_onset = freq_vect_sm[0]
                    else:
                        pk_onset = freq_vect_sm[int([minma[0] for minma in mintab if minma[0] < maxInd_sm][-1])]
                else:
                    pk_onset = freq_vect_sm[np.where(smoothed[:maxInd_sm]<bline)[0][-1]]
                PkWidth_Int[pi,ii] = pk_offset-pk_onset

                maxtab, mintab = peakdet(smoothed, np.max(smoothed)/3.)
                harms = [mxma for mxma in maxtab if (mxma[0]-maxInd_sm > 20 or mxma[0]-maxInd_sm < -20) and mxma[1] >= np.max(smoothed)/2.0]
                harms = np.array(harms)
                rem_inds = []
                for jj in range(len(harms))[:-1]:
                    if harms[jj][0] - harms[jj+1][0] > -10:
                        rem_inds.append(jj)
                harms = np.delete(harms, rem_inds, axis=0)
                Harmonics_Int[pi,ii] = len(harms)

                corr_sig = np.correlate(RateSig_Int, RateSig_Int, mode='full')
                corr_sig = corr_sig[int(len(corr_sig)/2):]/(np.mean(rates_Int)**2)
                maxtab, mintab = peakdet(corr_sig, 0.1)
                if not mintab.any():
                    SynchMeasure_Int[pi,ii] = float('nan')
                else:
                    SynchMeasure_Int[pi,ii] = (corr_sig[0] - mintab[0,1])
                    
                    
                if comp_phase:
        
                    # Pyr.:
                    I_AMPA = rawfile.root.IsynP_Pyr/namp
                    I_GABA = rawfile.root.IsynP_Pyr/namp

                    N = I_AMPA.shape[0]
                    NFFT = 2**(N-1).bit_length()
                    freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                    freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
                    a = pmtm(RateSig_Pyr, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
                    I_MTS = I_MTS[np.where(freq_vect/Hz<=300)]
                    fpeak = freq_vect[np.argmax(I_MTS)]

                    corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
                    phases = np.arange(1-N, N)

                    PhaseShift_Pyr[pi,ii] = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)

                    # Int.:
                    I_AMPA = rawfile.root.IsynP_Int/namp
                    I_GABA = rawfile.root.IsynP_Int/namp

                    N = I_AMPA.shape[0]
                    NFFT = 2**(N-1).bit_length()
                    freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                    freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
                    a = pmtm(RateSig_Pyr, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
                    I_MTS = I_MTS[np.where(freq_vect/Hz<=300)]
                    fpeak = freq_vect[np.argmax(I_MTS)]

                    corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
                    phases = np.arange(1-N, N)

                    PhaseShift_Int[pi,ii] = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
    
    if PlotFlag:
        plot_results(AvgCellRate_Pyr, SynchFreq_Pyr, SynchFreqPow_Pyr, PkWidth_Pyr, Harmonics_Pyr, SynchMeasure_Pyr, PhaseShift_Pyr, AvgCellRate_Int, SynchFreq_Int, SynchFreqPow_Int, PkWidth_Int, Harmonics_Int, SynchMeasure_Int, PhaseShift_Int, IterArray1, IterArray2, mode, out_file)
    
#####################################################################################


def plot_results(IterArray1, IterArray2, mode, AvgCellRate_Pyr, SynchFreq_Pyr, SynchFreqPow_Pyr, PkWidth_Pyr, Harmonics_Pyr, SynchMeasure_Pyr, AvgCellRate_Int, SynchFreq_Int, SynchFreqPow_Int, PkWidth_Int, Harmonics_Int, SynchMeasure_Int, PhaseShift_Pyr=None, PhaseShift_Int=None, out_file=None):
    
    if PhaseShift_Pyr is None:
        nrows = 6
    else:
        nrows = 7
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


def plot_mts_grid(rawfile, mode, start_time=200, end_time=1000, mts_win='whole', W=2**12, ws=(2**12)/10, sim_dt=0.02, out_file=None):
    
    sim_dt *= ms
    
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
                    RateMTS_Pyr = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
                    RateMTS_Pyr = RateMTS_Pyr[np.where(freq_vect/Hz<=300)]

                    a = pmtm(RateSig_Int, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    RateMTS_Int = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
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
                        result += Sks[:NFFT/2]
                    RateMTS_Pyr = result[np.where(freq_vect/Hz<=300)]/N_segs

                    N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                    result = np.zeros((NFFT/2))
                    for i in range(N_segs):
                        data = RateSig_Int[i*ws:i*ws+W]
                        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                        result += Sks[:NFFT/2]    
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
                    RateMTS_Pyr = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
                    RateMTS_Pyr = RateMTS_Pyr[np.where(freq_vect/Hz<=300)]

                    a = pmtm(RateSig_Int, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    RateMTS_Int = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]
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
                        result += Sks[:NFFT/2]
                    RateMTS_Pyr = result[np.where(freq_vect/Hz<=300)]/N_segs

                    N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                    result = np.zeros((NFFT/2))
                    for i in range(N_segs):
                        data = RateSig_Int[i*ws:i*ws+W]
                        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                        result += Sks[:NFFT/2]    
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


def plot_spikes_grid(rawfile, mode, start_time=200, end_time=1000, sim_dt=0.02, out_file=None):
    
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
                xlim(start_time, end_time)
                xlabel('Time (ms)')
                ylabel('Neuron Index')
                title('IP Amp.: %s, IP Freq.:%s' % (IP_A, IP_f))
            
    if not (out_file is None):
        savefig(out_file+'.png')
    
    show()
    
#####################################################################################


def plot_poprate_grid(rawfile, mode, start_time=200, end_time=1000, sim_dt=0.02, out_file=None):
    
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


def plot_spcgram_grid(rawfile, mode, start_time=200, end_time=1000, W=2**12, ws=(2**12)/10, sim_dt=0.02, out_file=None):
    
    sim_dt *= ms
    
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
                    RateMTS_Pyr[:,i] = Sks[:NFFT/2]
                RateMTS_Pyr = np.squeeze(RateMTS_Pyr[np.where(freq_vect/Hz<=300),:])

                N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                RateMTS_Int = np.zeros((NFFT/2, N_segs))
                for i in range(N_segs):
                    data = RateSig_Int[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    RateMTS_Int[:,i] = Sks[:NFFT/2]
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
                    RateMTS_Pyr[:,i] = Sks[:NFFT/2]
                RateMTS_Pyr = np.squeeze(RateMTS_Pyr[np.where(freq_vect/Hz<=300),:])

                N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
                RateMTS_Int = np.zeros((NFFT/2, N_segs))
                for i in range(N_segs):
                    data = RateSig_Int[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    RateMTS_Int[:,i] = Sks[:NFFT/2]
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
        savefig(out_file+'_Pyr')
        figure(2)
        savefig(out_file+'_Int')
    
    show()
