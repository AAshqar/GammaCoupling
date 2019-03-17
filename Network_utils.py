import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from skimage.feature import peak_local_max
from random import choice
import tables
from brian2 import *
from spectrum import pmtm
from HelpingFuncs.peakdetect import peakdet
from HelpingFuncs.peakfinders import *
from HelpingFuncs.FreqAnalysis import comp_mtspectrogram
import time
import gc

from NeuronsSpecs.NeuronParams import *
from NeuronsSpecs.NeuronEqs_DFsepI import *

import copy

#####################################################################################


def sim_network(N_p=4000, N_i=1000, SilencePyrs=None, SilenceInts=None, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInp=1, IntInp=1, PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, sim_dt=0.02, SynConns=None, SaveSynns=False, monitored=[], monitors_dt=0.2, avgoneurons=True, avgotime=False, avgotimest=10, monitor_pois=False, record_vm=False, save_raw=False, filename=None, verbose=True, PlotFlag=False):
    '''
    Simulates a network consisting of the desired number of pyramidal neurons and interneurons with dynamics described by predefined set of equations.
    
    * Parameters:
    - N_p: Number of excitatory pyramidal neurons in the network
    - N_i: Number of inhibitory interneurons in the network
    - PyrEqs: Equations to use for the pyramidal population
    - IntEqs: Equations to use for the interneurons population
    - PreEqAMPA: Equations to use for AMPA (excitatory) synapses
    - PreEqGABA: Equations to use for GABA (inhibitory) synapses
    - PyrInp: The poisson input to the pyramidal population. Should be either a single value [kHz] to model a homogenous Poisson input, or a function that takes a time vector as an input and returns an inhomogenous Poisson signal
    - IntInp: The poisson input to the interneuron population. Should be either a single value [kHz] to model a homogenous Poisson input, or a function that takes a time vector as an input and returns an inhomogenous Poisson signal
    - PP_C: Pyramidal-pyramidal Connectivity
    - IP_C: Interneuron-pyramidal Connectivity
    - II_C: Interneuron-interneuron Connectivity
    - PI_C: Pyramidal-interneuron Connectivity
    - runtime: Time of the simulation [ms]
    - sim_dt: Simulation time step [ms]
    - monitored: List of monitored variables from the neurons
    - monitor_pois: Whether to have monitors for the poisson input populations
    - record_vm: Whether to record membrane voltage of the neurons (only first and last of each population)
    - save_raw: Whether to save raw data after the simulation
    - filename: File name under which raw data is saved
    - verbose: Whether to print progress texts while running
    - PlotFlag: Whether to plot some results after the simulation
    
    * Returns:
    - Monitors: A dictinary of the different monitors used in the simulation and recorded data stored within them
    '''
    
    runtime *= ms
    sim_dt *= ms
    monitors_dt *= ms
    
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
    SynIP = Synapses(Int_pop, Pyr_pop, on_pre=PreEq_GABA, delay=delay_GABA)
    SynII = Synapses(Int_pop, Int_pop, on_pre=PreEq_GABA, delay=delay_GABA)
    SynPI = Synapses(Pyr_pop, Int_pop, on_pre=PreEq_AMPA, delay=delay_AMPA)

    if SynConns is None:
        SynPP.connect(p=PP_C)
        SynIP.connect(p=IP_C)
        SynII.connect(p=II_C)
        SynPI.connect(p=PI_C)
    else:
        SynPP.connect(i=SynConns['SynPPij'][:,0], j=SynConns['SynPPij'][:,1])
        SynIP.connect(i=SynConns['SynIPij'][:,0], j=SynConns['SynIPij'][:,1])
        SynII.connect(i=SynConns['SynIIij'][:,0], j=SynConns['SynIIij'][:,1])
        SynPI.connect(i=SynConns['SynPIij'][:,0], j=SynConns['SynPIij'][:,1])

    if not SilencePyrs is None:
        Pyr_pop.Iext_s[SilencePyrs] = -500*namp
    if not SilenceInts is None:
        Int_pop.Iext[SilenceInts] = -500*namp
        
    Pyr_VoltRange = np.arange((eL_p/mV)-5., (eL_p/mV)+5.1, 0.1)
    Int_VoltRange = np.arange((eL_i/mV)-5., (eL_i/mV)+5.1, 0.1)
    Pyr_pop.v_s = choice(Pyr_VoltRange, N_p)*mV
    Pyr_pop.v_d = choice(Pyr_VoltRange, N_p)*mV
    Int_pop.v = choice(Int_VoltRange, N_i)*mV
    
    defaultclock.dt = sim_dt
    t_vector = np.arange(0, runtime, defaultclock.dt)
    
    if callable(PyrInp):
        PyrInpSignal, PyrIPois_params = PyrInp(t_vector)
        Poiss_AMPA_Pyr = PoissonGroup(N_p, rates='PyrInpSignal(t)')
    else:
        PyrInp *= kHz
        Poiss_AMPA_Pyr = PoissonGroup(N_p, PyrInp)
        
    if callable(IntInp):
        IntInpSignal, IntIPois_params = IntInp(t_vector)
        Poiss_AMPA_Int = PoissonGroup(N_i, rates='IntInpSignal(t)')
    else:
        IntInp *= kHz
        Poiss_AMPA_Int = PoissonGroup(N_i, IntInp)

    SynPoiss_AMPA_Pyr = Synapses(Poiss_AMPA_Pyr, Pyr_pop, on_pre=PreEq_AMPA_pois, delay=delay_AMPA)
    SynPoiss_AMPA_Pyr.connect(j='i')

    SynPoiss_AMPA_Int = Synapses(Poiss_AMPA_Int, Int_pop, on_pre=PreEq_AMPA_pois, delay=delay_AMPA)
    SynPoiss_AMPA_Int.connect(j='i')

    SpikeM_Pyr = SpikeMonitor(Pyr_pop)
    PopRateM_Pyr = PopulationRateMonitor(Pyr_pop)

    SpikeM_Int = SpikeMonitor(Int_pop)
    PopRateM_Int = PopulationRateMonitor(Int_pop)
    
    monitors = [SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int]
    if not monitored_Pyr==[]:
        StateM_Pyr = StateMonitor(Pyr_pop, monitored_Pyr, record=True, dt=monitors_dt)
        StateM_Int = StateMonitor(Int_pop, monitored_Int, record=True, dt=monitors_dt)
        monitors.append(StateM_Pyr)
        monitors.append(StateM_Int)
        
    if monitor_pois:
        SpikeM_PoisPyr = SpikeMonitor(Poiss_AMPA_Pyr)
        PopRateM_PoisPyr = PopulationRateMonitor(Poiss_AMPA_Pyr)
        monitors.append(SpikeM_PoisPyr)
        monitors.append(PopRateM_PoisPyr)

        SpikeM_PoisInt = SpikeMonitor(Poiss_AMPA_Int)
        PopRateM_PoisInt = PopulationRateMonitor(Poiss_AMPA_Int)
        monitors.append(SpikeM_PoisInt)
        monitors.append(PopRateM_PoisInt)
        
    net = Network(Pyr_pop, Int_pop, Poiss_AMPA_Pyr, Poiss_AMPA_Int,
                  SynPP, SynIP, SynPI, SynII, SynPoiss_AMPA_Pyr,
                  SynPoiss_AMPA_Int, monitors)
    if verbose:
        print('Running %s simulation of the network...' %(runtime))

    t1 = time.time()
    net.run(runtime)
    t2 = time.time()
    
    if verbose:
        print('Simulating %s took %s minutes...' %(runtime, (t2-t1)/60.))
    
    if save_raw:
        
        if filename is None:
            filename = 'new_experiment'
        
        rawfile = tables.open_file(filename+'_raw.h5', mode='w', title='RawData')
        root = rawfile.root
        rawfile.create_carray(root, 'SpikeM_t_Pyr', obj=np.array(SpikeM_Pyr.t/ms)*ms)
        rawfile.create_carray(root, 'SpikeM_i_Pyr', obj=np.array(SpikeM_Pyr.i))
        rawfile.create_carray(root, 'SpikeM_t_Int', obj=np.array(SpikeM_Int.t/ms)*ms)
        rawfile.create_carray(root, 'SpikeM_i_Int', obj=np.array(SpikeM_Int.i))
        rawfile.create_carray(root, 'PopRateSig_Pyr', obj=PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
        rawfile.create_carray(root, 'PopRateSig_Int', obj=PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
        if SaveSynns:
            rawfile.create_carray(root, 'SynPP_ij', obj=np.array(zip(SynPP.i,SynPP.j)))
            rawfile.create_carray(root, 'SynIP_ij', obj=np.array(zip(SynIP.i,SynIP.j)))
            rawfile.create_carray(root, 'SynII_ij', obj=np.array(zip(SynII.i,SynII.j)))
            rawfile.create_carray(root, 'SynPI_ij', obj=np.array(zip(SynPI.i,SynPI.j)))
        if monitor_pois:
            rawfile.create_carray(root, 'SpikeM_t_PoisPyr', obj=np.array(SpikeM_PoisPyr.t/ms)*ms)
            rawfile.create_carray(root, 'SpikeM_i_PoisPyr', obj=np.array(SpikeM_PoisPyr.i))
            rawfile.create_carray(root, 'SpikeM_t_PoisInt', obj=np.array(SpikeM_PoisInt.t/ms)*ms)
            rawfile.create_carray(root, 'SpikeM_i_PoisInt', obj=np.array(SpikeM_PoisInt.i))
            rawfile.create_carray(root, 'PopRateSig_PoisPyr', obj=PopRateM_PoisPyr.smooth_rate(window='gaussian', width=1*ms))
            rawfile.create_carray(root, 'PopRateSig_PoisInt', obj=PopRateM_PoisInt.smooth_rate(window='gaussian', width=1*ms))
            if callable(PyrInp):
                rawfile.create_array(root, 'PyrInpDC', obj=PyrIPois_params['DC_Inp'])
                rawfile.create_array(root, 'PyrIPois_A', obj=PyrIPois_params['IPois_A'])
                rawfile.create_array(root, 'PyrIPois_f', obj=PyrIPois_params['IPois_f'])
                rawfile.create_array(root, 'PyrIPois_ph', obj=PyrIPois_params['IPois_ph'])
                rawfile.create_array(root, 'PyrIPoisA_Type', obj=PyrIPois_params['IPoisA_Type'])
                rawfile.create_array(root, 'PyrIPois_phrad', obj=PyrIPois_params['IPois_phrad'])
            else:
                rawfile.create_array(root, 'PyrInp', obj=PyrInp)
            if callable(IntInp):
                rawfile.create_array(root, 'IntInpDC', obj=IntIPois_params['DC_Inp'])
                rawfile.create_array(root, 'IntIPois_A', obj=IntIPois_params['IPois_A'])
                rawfile.create_array(root, 'IntIPois_f', obj=IntIPois_params['IPois_f'])
                rawfile.create_array(root, 'IntIPois_ph', obj=IntIPois_params['IPois_ph'])
                rawfile.create_array(root, 'IntIPoisA_Type', obj=IntIPois_params['IPoisA_Type'])
                rawfile.create_array(root, 'IntIPois_phrad', obj=IntIPois_params['IPois_phrad'])
            else:
                rawfile.create_array(root, 'IntInp', obj=IntInp)

        if not monitored==[]:
            for i,var in enumerate(monitored):
                if avgoneurons:
                    rawfile.create_carray(root, var+'_Pyr', obj=np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                    rawfile.create_carray(root, var+'_Int', obj=np.array(StateM_Int.get_states()[var]).mean(axis=1))
                if avgotime:
                    rawfile.create_carray(root, var+'_Pyr', obj=np.array(StateM_Pyr.get_states()[var])[int(avgotimest/monitors_dt),:].mean(axis=0))
                    rawfile.create_carray(root, var+'_Int', obj=np.array(StateM_Int.get_states()[var])[int(avgotimest/monitors_dt),:].mean(axis=0))
                if not any((avgoneurons,avgotime)):
                    locals()[var+'_Pyr'] = rawfile.create_carray(root, var+'_Pyr', obj=np.array(StateM_Pyr.get_states()[var]))
                    locals()[var+'_Int'] = rawfile.create_carray(root, var+'_Int', obj=np.array(StateM_Int.get_states()[var]))
    
        if record_vm:
            rawfile.create_carray(root, 'Vm_Pyr', obj=StateM_Pyr.get_states()['v_s'])
            rawfile.create_carray(root, 'Vm_Int', obj=StateM_Pyr.get_states()['v'])
        
        rawfile.close()
        
        if verbose:
            print('Saved raw data successfullty!')
          
    if PlotFlag:
        figure()
        subplot(2,1,1)
        plot(SpikeM_Pyr.t/ms, SpikeM_Pyr.i, '.', SpikeM_Int.t/ms, SpikeM_Int.i+4000, '.')
        xlim(PopRateM_Pyr.t[0]/ms, PopRateM_Pyr.t[-1]/ms)
        ylabel('Neuron Index')
        subplot(2,1,2)
        plot(PopRateM_Pyr.t/ms, PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms), PopRateM_Int.t/ms, PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
        xlabel('Time (ms)')
        ylabel('Inst. Population Rate (Hz)')
        xlim(PopRateM_Pyr.t[0]/ms, PopRateM_Pyr.t[-1]/ms)
        show()
    
    Monitors = {'SpikeM_Pyr':SpikeM_Pyr,
                'PopRateM_Pyr':PopRateM_Pyr,
                'SpikeM_Int':SpikeM_Int,
                'PopRateM_Int':PopRateM_Int,
                'SynPP':SynPP, 'SynIP':SynIP,
                'SynII':SynII, 'SynPI':SynPI}
    if not monitored_Pyr==[]:
        Monitors['StateM_Pyr'] = StateM_Pyr
        Monitors['StateM_Int'] = StateM_Int
    if monitor_pois:
        Monitors['SpikeM_PoisPyr'] = SpikeM_PoisPyr
        Monitors['PopRateM_PoisPyr'] = PopRateM_PoisPyr
        Monitors['SpikeM_PoisInt'] = SpikeM_PoisInt
        Monitors['PopRateM_PoisInt'] = PopRateM_PoisInt
        
    return Monitors
    
#####################################################################################


def analyze_network(Monitors, comp_phase_curr=False, N_p=4000, N_i=1000, start_time=None, end_time=None, sim_dt=0.02, mts_win='whole', W=2**13, ws=None, PlotFlag=False):
    '''
    Analyzes a pre-simulated network and extracts various features using the provided monitors
    
    * Parameters:
    - Monitors: A dictionary of the monitors used to record raw data during the simulation
    - comp_phase_curr: Whether to include phase shift calculations between of AMPA and GABA currents
    - N_p: Number of pyramidal neurons in the simulated network
    - N_i: Number of interneurons in the simulated network
    - start_time: Beginning time of analysis within the simulation time [ms]
    - end_time: Ending time of analysis within the simulation time [ms]
    - sim_dt: Time step used in the simulation [ms]
    - mts_win: Whether to calculate the spectrum for the whole recording or as an average of moving window spectrums
    - W: Window length for the calculation of the multi-taper spectrum
    - ws: Sliding step of the moving window for muti-taper spectrum estimation
    - PlotFlag: Whether to plot some results after the simulation
    
    * Returns:
    - Network_feats: A dictinary of the different features calculated from the recorded data of the simulation
    '''

    SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int = Monitors['SpikeM_Pyr'], Monitors['PopRateM_Pyr'], Monitors['SpikeM_Int'], Monitors['PopRateM_Int']
    
    sim_dt *= ms
    
    if ws is None:
        ws = W/10
        
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = PopRateM_Pyr.t[-1]/ms
    
    if start_time > PopRateM_Pyr.t[-1]/ms or end_time > PopRateM_Pyr.t[-1]/ms:
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
        RateMTS_Pyr = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
        RateMTS_Pyr = RateMTS_Pyr[np.where(freq_vect/Hz<=300)]

        a = pmtm(RateSig_Int, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        RateMTS_Int = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
        RateMTS_Int = RateMTS_Int[np.where(freq_vect/Hz<=300)]
        
    else:
        
        NFFT=W*2
        freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
        freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
        
        N = len(RateSig_Pyr)
        N_segs = int((N-W)/ws+2)
        result = np.zeros((NFFT/2))
        for i in range(N_segs):
            data = RateSig_Pyr[i*ws:i*ws+W]
            a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
            Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
            result += Sks[:NFFT/2]/W
        RateMTS_Pyr = result[np.where(freq_vect/Hz<=300)]/N_segs
        
        N = len(RateSig_Int)
        N_segs = int((N-W)/ws+2)
        result = np.zeros((NFFT/2))
        for i in range(N_segs):
            data = RateSig_Int[i*ws:i*ws+W]
            a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
            Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
            result += Sks[:NFFT/2]/W
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
        corr_sig = corr_sig[int(len(corr_sig)/2):]/(N*np.mean(rates_Pyr)**2)
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
        corr_sig = corr_sig[int(len(corr_sig)/2):]/(N*np.mean(rates_Int)**2)
        maxtab, mintab = peakdet(corr_sig, 0.1)
        if not mintab.any():
            SynchMeasure_Int = float('nan')
        else:
            SynchMeasure_Int = (corr_sig[0] - mintab[0,1])
    
    if comp_phase_curr:
        
        StateM_Pyr, StateM_Int = Monitors['StateM_Pyr'], Monitors['StateM_Int']
        
        # Pyr.:
        I_AMPA = np.mean(StateM_Pyr.get_states()['IsynP'], axis=1)/namp
        I_AMPA = I_AMPA[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
        I_AMPA -= np.mean(I_AMPA)
        I_GABA = np.mean(StateM_Pyr.get_states()['IsynI'], axis=1)/namp
        I_GABA = I_GABA[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
        I_GABA -= np.mean(I_GABA)

        N = I_AMPA.shape[0]
        NFFT = 2**(N-1).bit_length()
        freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
        freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
        a = pmtm(I_GABA, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
        I_MTS = I_MTS[np.where(freq_vect/Hz<=300)]
        fpeak = freq_vect[np.argmax(I_MTS)]

        corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
        phases = np.arange(1-N, N)

        PhaseShiftCurr_Pyr = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
        PhaseShiftCurr_Pyr = np.sign(PhaseShiftCurr_Pyr)*(PhaseShiftCurr_Pyr%360)
        
        # Int.:
        I_AMPA = np.mean(StateM_Int.get_states()['IsynP'], axis=1)/namp
        I_AMPA = I_AMPA[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
        I_AMPA -= np.mean(I_AMPA)
        I_GABA = np.mean(StateM_Int.get_states()['IsynI'], axis=1)/namp
        I_GABA = I_GABA[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
        I_GABA -= np.mean(I_GABA)

        N = I_AMPA.shape[0]
        NFFT = 2**(N-1).bit_length()
        freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
        freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
        a = pmtm(I_GABA, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
        I_MTS = I_MTS[np.where(freq_vect/Hz<=300)]
        fpeak = freq_vect[np.argmax(I_MTS)]

        corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
        phases = np.arange(1-N, N)

        PhaseShiftCurr_Int = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
        PhaseShiftCurr_Int = np.sign(PhaseShiftCurr_Int)*(PhaseShiftCurr_Int%360)
        
    if PlotFlag:
        figure(figsize=[8,7])
        subplot(2,2,1)
        hist(rates_Pyr, bins=20)
        title('Single Cell Rates Hist. (Pyr)')
        xlabel('Frequency (Hz)')
        subplot(2,2,2)
        hist(rates_Int, bins=20)
        title('Single Cell Rates Hist. (Int)')
        xlabel('Frequency (Hz)')
        ylabel('Count')
        subplot(2,2,3)
        plot(freq_vect, RateMTS_Pyr)
        xlim(0, 300)
        title('Pop. Spectrum (Pyr)')
        xlabel('Frequency (Hz)')
        subplot(2,2,4)
        plot(freq_vect, RateMTS_Int)
        xlim(0, 300)
        title('Pop. Spectrum (Int)')
        xlabel('Frequency (Hz)')
        ylabel('Power')
        show()

    Network_feats = {'AvgCellRate_Pyr':AvgCellRate_Pyr,
                     'SynchFreq_Pyr':SynchFreq_Pyr,
                     'SynchFreqPow_Pyr':SynchFreqPow_Pyr,
                     'PkWidth_Pyr':PkWidth_Pyr,
                     'Harmonics_Pyr':Harmonics_Pyr,
                     'SynchMeasure_Pyr':SynchMeasure_Pyr,
                     'AvgCellRate_Int':AvgCellRate_Int,
                     'SynchFreq_Int':SynchFreq_Int,
                     'SynchFreqPow_Int':SynchFreqPow_Int,
                     'PkWidth_Int':PkWidth_Int,
                     'Harmonics_Int':Harmonics_Int,
                     'SynchMeasure_Int':SynchMeasure_Int}
    if comp_phase_curr:
        Network_feats['PhaseShiftCurr_Pyr'] = PhaseShiftCurr_Pyr
        Network_feats['PhaseShiftCurr_Int'] = PhaseShiftCurr_Int
            
    return Network_feats
    
#####################################################################################


def PopRateM_mtspectrogram(Monitors, W=2**13, ws=None, start_time=None, end_time=None, sim_dt=0.02, PlotFlag=True):
    '''
    Computes spectrograms of the instantaneuous population rates for the pyramidal population and interneuronal population respectively, using the provided monitors dictionary
    
    * Parameters:
    - Monitors: A dictionary of the monitors used to record raw data during the simulation
    - mts_win: Whether to calculate the spectrum for the whole recording or as an average of moving window spectrums
    - W: Window length for the calculation of the multi-taper spectrum
    - ws: Sliding step of the moving window for muti-taper spectrum estimation
    - start_time: Beginning time of analysis within the simulation time [ms]
    - end_time: Ending time of analysis within the simulation time [ms]
    - sim_dt: Time step used in the simulation [ms]
    - PlotFlag: Whether to plot some results after the simulation
    
    * Returns:
    - Rate_MTS: A dictinary containing spectrograms of the pyramidal population and interneuronal population respectively
    '''
    
    PopRateM_Pyr, PopRateM_Int = Monitors['PopRateM_Pyr'], Monitors['PopRateM_Int']
    
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = PopRateM_Pyr.t[-1]/ms
    
    if start_time > PopRateM_Pyr.t[-1]/ms or end_time > PopRateM_Pyr.t[-1]/ms:
        raise ValueError('Please provide start time and end time within the simulation time window!')
    
    sim_dt *= ms
    
    if ws is None:
        ws = W/10
    
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
    
    N_whole = len(PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
    T = N_whole*(sim_dt/ms)
    Time_whole = np.arange(0,T,(sim_dt/ms))
    
    N = len(RateSig_Pyr)
    N_segs = int((N-W)/ws+2)
    time_v = Time_whole[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
    MTS_time = time_v[np.arange(W/2, (N_segs)*ws+(W/2), ws).astype(int)]
    
    result = np.zeros((NFFT/2, N_segs))
    for i in range(N_segs):
        data = RateSig_Pyr[i*ws:i*ws+W]
        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
        result[:,i] = Sks[:NFFT/2]/W
    RateMTS_Pyr = np.squeeze(result[np.where(freq_vect/Hz<=300),:])
        
    N = len(RateSig_Int)
    N_segs = int((N-W)/ws+2)
    result = np.zeros((NFFT/2, N_segs))
    for i in range(N_segs):
        data = RateSig_Int[i*ws:i*ws+W]
        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
        result[:,i] = Sks[:NFFT/2]/W
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
    
    MTS_data = {'RateMTS_Pyr':RateMTS_Pyr, 'RateMTS_Int':RateMTS_Int, 'MTS_time':MTS_time, 'freq_vect':freq_vect}
    
    return MTS_data
    
#####################################################################################


def run_multsim(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInps=[0.5,1], IntInps=[0.5,1], PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, start_time=None, end_time=None, sim_dt=0.02, SynConns=None, SaveSynns=False, monitored=[], monitors_dt=0.2, avgoneurons=True, avgotime=False, avgotimest=10, monitor_pois=False, mon_avg=True, comp_phase_curr=False, record_vm=False, mts_win='whole', W=2**13, ws=None, verbose=True, analyze=True, save_analyzed=False, save_raw=False, filename=None, newfile=True):
    '''
    Runs simulations for all combinations of the provided lists of poisson input rates for the two populations of pyramidal neurons and interneurons.
    
    * Parameters:
    - analyze: Whether to analyze simulations and extract features
    - save_analyzed: Whether to sava analysis results in an output file
    - save_raw: Whether to save raw data of the simulations' recordings
    - filename: Name of the file underwhich raw/analysis data is saved (name will be appended by 'raw' or 'analysis')
    [Please refer to the documentation of 'run_network()' for other parameters]
    
    * Returns:
    - Sims_feats: (if 'analyze' is set to true) a dictinary of the different features calculated from the recorded data of all simulations in a matrix form
    '''
    
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = runtime
    
    if start_time > runtime or end_time > runtime:
        raise ValueError('Please provide start time and end time within the simulation time window!')
    
    if not any((type(PyrInps) is list, type(PyrInps) is np.ndarray)):
        PyrInps = [PyrInps]
    if not any((type(IntInps) is list, type(IntInps) is np.ndarray)):
        IntInps = [IntInps]
    
    if ws is None:
        ws = W/10
    
    N_samples = int((runtime/ms)/(monitors_dt/ms))
    
    if filename is None:
        filename = 'new_experiment'
    
    if save_raw and newfile:
        rawfile = tables.open_file(filename+'_raw.h5', mode='w', title='RawData')
        root = rawfile.root
        Params = rawfile.create_vlarray(root, 'InpsParams', tables.Float64Atom(shape=()))
        SpikeM_t_Pyr = rawfile.create_vlarray(root, 'SpikeM_t_Pyr', tables.Float64Atom(shape=()))
        SpikeM_i_Pyr = rawfile.create_vlarray(root, 'SpikeM_i_Pyr', tables.Float64Atom(shape=()))
        SpikeM_t_Int = rawfile.create_vlarray(root, 'SpikeM_t_Int', tables.Float64Atom(shape=()))
        SpikeM_i_Int = rawfile.create_vlarray(root, 'SpikeM_i_Int', tables.Float64Atom(shape=()))
        PopRateSig_Pyr = rawfile.create_vlarray(root, 'PopRateSig_Pyr', tables.Float64Atom(shape=()))
        PopRateSig_Int = rawfile.create_vlarray(root, 'PopRateSig_Int', tables.Float64Atom(shape=()))
        if SaveSynns:
            SynPP_i = rawfile.create_vlarray(root, 'SynPP_i', tables.Int32Atom(shape=()))
            SynPP_j = rawfile.create_vlarray(root, 'SynPP_j', tables.Int32Atom(shape=()))
            SynIP_i = rawfile.create_vlarray(root, 'SynIP_i', tables.Int32Atom(shape=()))
            SynIP_j = rawfile.create_vlarray(root, 'SynIP_j', tables.Int32Atom(shape=()))
            SynII_i = rawfile.create_vlarray(root, 'SynII_i', tables.Int32Atom(shape=()))
            SynII_j = rawfile.create_vlarray(root, 'SynII_j', tables.Int32Atom(shape=()))
            SynPI_i = rawfile.create_vlarray(root, 'SynPI_i', tables.Int32Atom(shape=()))
            SynPI_j = rawfile.create_vlarray(root, 'SynPI_j', tables.Int32Atom(shape=()))
        rawfile.create_carray(root, "PyrInps", obj=PyrInps*kHz)
        rawfile.create_carray(root, "IntInps", obj=IntInps*kHz)
        if not monitored==[]:
            for i,var in enumerate(monitored):
                if mon_avg:
                    locals()[var+'_Pyr'] = rawfile.create_vlarray(root, var+'_Pyr', tables.Float64Atom(shape=()))
                    locals()[var+'_Int'] = rawfile.create_vlarray(root, var+'_Int', tables.Float64Atom(shape=()))
                else:
                    locals()[var+'_Pyr'] = rawfile.create_vlarray(root, var+'_Pyr', tables.Float64Atom(shape=(N_samples, N_p)))
                    locals()[var+'_Int'] = rawfile.create_vlarray(root, var+'_Int', tables.Float64Atom(shape=(N_samples, N_i)))
        if monitor_pois:
            SpikeM_t_PoisPyr = rawfile.create_vlarray(root, 'SpikeM_t_PoisPyr', tables.Float64Atom(shape=()))
            SpikeM_i_PoisPyr = rawfile.create_vlarray(root, 'SpikeM_i_PoisPyr', tables.Float64Atom(shape=()))
            SpikeM_t_PoisInt = rawfile.create_vlarray(root, 'SpikeM_t_PoisInt', tables.Float64Atom(shape=()))
            SpikeM_i_PoisInt = rawfile.create_vlarray(root, 'SpikeM_i_PoisInt', tables.Float64Atom(shape=()))
            PopRateSig_PoisPyr = rawfile.create_vlarray(root, 'PopRateSig_PoisPyr', tables.Float64Atom(shape=()))
            PopRateSig_PoisInt = rawfile.create_vlarray(root, 'PopRateSig_PoisInt', tables.Float64Atom(shape=()))
        if record_vm:
            Vm_Pyr = rawfile.create_vlarray(root, 'Vm_Pyr', tables.Float64Atom(shape=(N_samples)))
            Vm_Int = rawfile.create_vlarray(root, 'Vm_Int', tables.Float64Atom(shape=(N_samples)))
        rawfile.close()

        
    if analyze:
        AvgCellRate_Pyr = np.zeros((len(PyrInps),len(IntInps)))
        SynchFreq_Pyr = np.zeros_like(AvgCellRate_Pyr)
        SynchFreqPow_Pyr = np.zeros_like(AvgCellRate_Pyr)
        PkWidth_Pyr = np.zeros_like(AvgCellRate_Pyr)
        Harmonics_Pyr = np.zeros_like(AvgCellRate_Pyr)
        SynchMeasure_Pyr = np.zeros_like(AvgCellRate_Pyr)
        PhaseShiftCurr_Pyr = np.zeros_like(AvgCellRate_Pyr)
        AvgCellRate_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchFreq_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchFreqPow_Int = np.zeros_like(AvgCellRate_Pyr)
        PkWidth_Int = np.zeros_like(AvgCellRate_Pyr)
        Harmonics_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchMeasure_Int = np.zeros_like(AvgCellRate_Pyr)
        PhaseShiftCurr_Int = np.zeros_like(AvgCellRate_Pyr)
    
    for pi,PyrInp in enumerate(PyrInps):
        
        for ii,IntInp in enumerate(IntInps):
            
            if verbose:
                print('[Starting simulation (%d/%d) for (Pyr. Input: %s, Int. Input: %s)...]' % (pi*len(IntInps)+ii+1, len(PyrInps)*len(IntInps), PyrInp, IntInp))
                
            if save_raw:
                rawfile = tables.open_file(filename+'_raw.h5', mode='a')
                InpsParamsList = rawfile.root.InpsParams.read()
                rawfile.close()
                if len(InpsParamsList)>0:
                    if any(all(np.array([PyrInp, IntInp]) == InpsParamsList, axis=1)):
                        print(' Already saved, skipping...')
                        continue
            
            gc.collect()
            
            Monitors = run_network(N_p=N_p, N_i=N_i, PyrEqs=PyrEqs, IntEqs=IntEqs, PreEqAMPA=PreEqAMPA, PreEqGABA=PreEqGABA, PyrInp=PyrInp, IntInp=IntInp, PP_C=PP_C, IP_C=IP_C, II_C=II_C, PI_C=PI_C, runtime=runtime, sim_dt=sim_dt, SynConns=SynConns, SaveSynns=SaveSynns, monitored=monitored, monitors_dt=monitors_dt, avgoneurons=avgoneurons, avgotime=avgotime, avgotimest=avgotimest, monitor_pois=monitor_pois, record_vm=record_vm, verbose=verbose, PlotFlag=False)
            
            if analyze:
                Network_feats = analyze_network(Monitors, comp_phase_curr=comp_phase_curr, N_p=N_p, N_i=N_i, start_time=start_time, end_time=end_time, sim_dt=sim_dt, mts_win=mts_win, W=W, ws=ws)
                
                AvgCellRate_Pyr[pi,ii], SynchFreq_Pyr[pi,ii], SynchFreqPow_Pyr[pi,ii], PkWidth_Pyr[pi,ii], Harmonics_Pyr[pi,ii], SynchMeasure_Pyr[pi,ii], AvgCellRate_Int[pi,ii], SynchFreq_Int[pi,ii], SynchFreqPow_Int[pi,ii], PkWidth_Int[pi,ii], Harmonics_Int[pi,ii], SynchMeasure_Int[pi,ii] = Network_feats['AvgCellRate_Pyr'], Network_feats['SynchFreq_Pyr'], Network_feats['SynchFreqPow_Pyr'], Network_feats['PkWidth_Pyr'], Network_feats['Harmonics_Pyr'], Network_feats['SynchMeasure_Pyr'], Network_feats['AvgCellRate_Int'], Network_feats['SynchFreq_Int'], Network_feats['SynchFreqPow_Int'], Network_feats['PkWidth_Int'], Network_feats['Harmonics_Int'], Network_feats['SynchMeasure_Int']
            
                if comp_phase_curr:
                    PhaseShiftCurr_Pyr[pi,ii], PhaseShiftCurr_Int[pi,ii] = Network_feats['PhaseShiftCurr_Pyr'], Network_feats['PhaseShiftCurr_Int']
                
            if save_raw:
                SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, SynPP, SynIP, SynII, SynPI = Monitors['SpikeM_Pyr'], Monitors['PopRateM_Pyr'], Monitors['SpikeM_Int'], Monitors['PopRateM_Int'], Monitors['SynPP'], Monitors['SynIP'], Monitors['SynII'], Monitors['SynPI']

                rawfile = tables.open_file(filename+'_raw.h5', mode='a')
                SpikeM_t_Pyr = rawfile.root.SpikeM_t_Pyr
                SpikeM_t_Pyr.append(np.array(SpikeM_Pyr.t/ms)*ms)
                SpikeM_i_Pyr = rawfile.root.SpikeM_i_Pyr
                SpikeM_i_Pyr.append(np.array(SpikeM_Pyr.i))
                SpikeM_t_Int = rawfile.root.SpikeM_t_Int
                SpikeM_t_Int.append(np.array(SpikeM_Int.t/ms)*ms)
                SpikeM_i_Int = rawfile.root.SpikeM_i_Int
                SpikeM_i_Int.append(np.array(SpikeM_Int.i))
                
                PopRateSig_Pyr = rawfile.root.PopRateSig_Pyr
                PopRateSig_Pyr.append(PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
                PopRateSig_Int = rawfile.root.PopRateSig_Int
                PopRateSig_Int.append(PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
                if SaveSynns:
                    SynPP_i = rawfile.root.SynPP_i
                    SynPP_i.append(np.array(SynPP.i))
                    SynPP_j = rawfile.root.SynPP_j
                    SynPP_j.append(np.array(SynPP.j))
                    SynIP_i = rawfile.root.SynIP_i
                    SynIP_i.append(np.array(SynIP.i))
                    SynIP_j = rawfile.root.SynIP_j
                    SynIP_j.append(np.array(SynIP.j))
                    SynII_i = rawfile.root.SynII_i
                    SynII_i.append(np.array(SynII.i))
                    SynII_j = rawfile.root.SynII_j
                    SynII_j.append(np.array(SynII.j))
                    SynPI_i = rawfile.root.SynPI_i
                    SynPI_i.append(np.array(SynPI.i))
                    SynPI_j = rawfile.root.SynPI_j
                    SynPI_j.append(np.array(SynPI.j))
                                        
                if not monitored==[]:
                    StateM_Pyr, StateM_Int = Monitors['StateM_Pyr'], Monitors['StateM_Int']
                    for i,var in enumerate(monitored): 
                        if avgoneurons:
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                            rawfile.get_node('/', name=var+'_Int').append(np.array(StateM_Int.get_states()[var]).mean(axis=1))
                        if avgotime:
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var])[int(avgotimest/monitors_dt),:].mean(axis=0))
                            rawfile.get_node('/', name=var+'_Int').append(np.array(StateM_Int.get_states()[var])[int(avgotimest/monitors_dt),:].mean(axis=0))
                        if not any((avgoneurons,avgotime)):
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var]))
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Int.get_states()[var]))
                            
                if monitor_pois:
                    SpikeM_PoisPyr, PopRateM_PoisPyr, SpikeM_PoisInt, PopRateM_PoisInt = Monitors['SpikeM_PoisPyr'], Monitors['PopRateM_PoisPyr'], Monitors['SpikeM_PoisInt'], Monitors['PopRateM_PoisInt']
                    SpikeM_t_PoisPyr = rawfile.root.SpikeM_t_PoisPyr
                    SpikeM_t_PoisPyr.append(np.array(SpikeM_PoisPyr.t/ms)*ms)
                    SpikeM_i_PoisPyr = rawfile.root.SpikeM_i_PoisPyr
                    SpikeM_i_PoisPyr.append(np.array(SpikeM_PoisPyr.i))
                    SpikeM_t_PoisInt = rawfile.root.SpikeM_t_PoisInt
                    SpikeM_t_PoisInt.append(np.array(SpikeM_PoisInt.t/ms)*ms)
                    SpikeM_i_PoisInt = rawfile.root.SpikeM_i_PoisInt
                    SpikeM_i_PoisInt.append(np.array(SpikeM_PoisInt.i))
                    PopRateSig_PoisPyr = rawfile.root.PopRateSig_PoisPyr
                    PRPois_Pyr = PopRateM_PoisPyr.smooth_rate(window='gaussian', width=1*ms)
                    PopRateSig_PoisPyr.append(PRPois_Pyr)
                    PopRateSig_PoisInt = rawfile.root.PopRateSig_PoisInt
                    PRPois_Int = PopRateM_PoisInt.smooth_rate(window='gaussian', width=1*ms)
                    PopRateSig_PoisInt.append(PRPois_Int)
                if record_vm:
                    StateM_Pyr, StateM_Int = Monitors['StateM_Pyr'], Monitors['StateM_Int']
                    Vm_Pyr = rawfile.root.Vm_Pyr
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,0])
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,-1])
                    Vm_Int = rawfile.root.Vm_Int
                    Vm_Int.append(StateM_Int.get_states()['v'][:,0])
                    Vm_Int.append(StateM_Int.get_states()['v'][:,-1])
                    
                Params = rawfile.root.InpsParams
                Params.append((PyrInp, IntInp))
                rawfile.close()
      
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
            h5file.create_carray(root, "PhaseShiftCurr_Pyr", obj=PhaseShiftCurr_Pyr)
            h5file.create_carray(root, "AvgCellRate_Int", obj=AvgCellRate_Int)
            h5file.create_carray(root, "SynchFreq_Int", obj=SynchFreq_Int)
            h5file.create_carray(root, "SynchFreqPow_Int", obj=SynchFreqPow_Int)
            h5file.create_carray(root, "PkWidth_Int", obj=PkWidth_Int)
            h5file.create_carray(root, "Harmonics_Int", obj=Harmonics_Int)
            h5file.create_carray(root, "SynchMeasure_Int", obj=SynchMeasure_Int)
            h5file.create_carray(root, "PhaseShiftCurr_Int", obj=PhaseShiftCurr_Int)
            h5file.create_carray(root, "PyrInps", obj=PyrInps)
            h5file.create_carray(root, "IntInps", obj=IntInps)
            h5file.close()
        if verbose:
            print('Saved analysis results successfully!')
            
    if save_raw:
        if verbose:
            print('Saved raw data successfully!')
    
    if analyze:
        Sims_feats = {'AvgCellRate_Pyr':AvgCellRate_Pyr,
                      'SynchFreq_Pyr':SynchFreq_Pyr,
                      'SynchFreqPow_Pyr':SynchFreqPow_Pyr,
                      'PkWidth_Pyr':PkWidth_Pyr,
                      'Harmonics_Pyr':Harmonics_Pyr,
                      'SynchMeasure_Pyr':SynchMeasure_Pyr,
                      'AvgCellRate_Int':AvgCellRate_Int,
                      'SynchFreq_Int':SynchFreq_Int,
                      'SynchFreqPow_Int':SynchFreqPow_Int,
                      'PkWidth_Int':PkWidth_Int,
                      'Harmonics_Int':Harmonics_Int,
                      'SynchMeasure_Int':SynchMeasure_Int}
        if comp_phase_curr:
            Sims_feats['PhaseShiftCurr_Pyr'] = PhaseShiftCurr_Pyr
            Sims_feats['PhaseShiftCurr_Int'] = PhaseShiftCurr_Int
            
        return Sims_feats
    
#####################################################################################


def run_multsim_IP(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInp=1, IntInp=1, IPois_As=[1.], IPois_phs=[0.], IPois_Atype='ramp', IPois_fs=[70], PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, start_time=None, end_time=None, sim_dt=0.02, SynConns=None, SaveSynns=False, monitored=[], monitors_dt=0.2, monitor_pois=False, avgoneurons=True, avgotime=False, avgotimest=10, record_vm=False, mts_win='whole', W=2**13, ws=None, verbose=True, analyze=True, save_analyzed=False, save_raw=False, filename=None, newfile=True):
    '''
    Runs simulations for all combinations of the provided lists of inhomogenous poisson amplitudes and frequencies and the provided values of inputs to the two populations (pyramidal neurons and interneurons).
    
    * Parameters:
    - analyze: Whether to analyze simulations and extract features
    - save_analyzed: Whether to sava analysis results in an output file
    - save_raw: Whether to save raw data of the simulations' recordings
    - filename: Name of the file underwhich raw/analysis data is saved (name will be appended by 'raw' or 'analysis')
    [Please refer to the documentation of 'run_network_IP()' for other parameters]
    
    * Returns:
    - Sims_feats: (if 'analyze' is set to true) a dictinary of the different features calculated from the recorded data of all simulations in a matrix form
    '''
    
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = runtime
    
    if start_time > runtime or end_time > runtime:
        raise ValueError('Please provide start time and end time within the simulation time window!')
    
    if not any((type(IPois_As) is list, type(IPois_As) is np.ndarray)):
        IPois_As = [IPois_As]
    if not any((type(IPois_fs) is list, type(IPois_fs) is np.ndarray)):
        IPois_fs = [IPois_fs]
    if not any((type(IPois_phs) is list, type(IPois_phs) is np.ndarray)):
        IPois_phs = np.ones([len(IPois_fs)])*IPois_phs
    
    if ws is None:
        ws = W/10
    
    N_samples = int((runtime/ms)/(monitors_dt/ms))
    
    if filename is None:
        filename = 'new_experiment'
    
    if save_raw and newfile:
        rawfile = tables.open_file(filename+'_raw.h5', mode='w', title='RawData')
        root = rawfile.root
        Params = rawfile.create_vlarray(root, 'InpsParams', tables.StringAtom(itemsize=40, shape=()))
        Params.append(str((PyrInp, IntInp)))
        IPoisParams = rawfile.create_vlarray(root, 'IPoisParams', tables.Float64Atom(shape=()))
        SpikeM_t_Pyr = rawfile.create_vlarray(root, 'SpikeM_t_Pyr', tables.Float64Atom(shape=()))
        SpikeM_i_Pyr = rawfile.create_vlarray(root, 'SpikeM_i_Pyr', tables.Float64Atom(shape=()))
        SpikeM_t_Int = rawfile.create_vlarray(root, 'SpikeM_t_Int', tables.Float64Atom(shape=()))
        SpikeM_i_Int = rawfile.create_vlarray(root, 'SpikeM_i_Int', tables.Float64Atom(shape=()))
        PopRateSig_Pyr = rawfile.create_vlarray(root, 'PopRateSig_Pyr', tables.Float64Atom(shape=()))
        PopRateSig_Int = rawfile.create_vlarray(root, 'PopRateSig_Int', tables.Float64Atom(shape=()))
        if SaveSynns:
            SynPP_i = rawfile.create_vlarray(root, 'SynPP_i', tables.Int32Atom(shape=()))
            SynPP_j = rawfile.create_vlarray(root, 'SynPP_j', tables.Int32Atom(shape=()))
            SynIP_i = rawfile.create_vlarray(root, 'SynIP_i', tables.Int32Atom(shape=()))
            SynIP_j = rawfile.create_vlarray(root, 'SynIP_j', tables.Int32Atom(shape=()))
            SynII_i = rawfile.create_vlarray(root, 'SynII_i', tables.Int32Atom(shape=()))
            SynII_j = rawfile.create_vlarray(root, 'SynII_j', tables.Int32Atom(shape=()))
            SynPI_i = rawfile.create_vlarray(root, 'SynPI_i', tables.Int32Atom(shape=()))
            SynPI_j = rawfile.create_vlarray(root, 'SynPI_j', tables.Int32Atom(shape=()))
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
        if monitor_pois:
            SpikeM_t_PoisPyr = rawfile.create_vlarray(root, 'SpikeM_t_PoisPyr', tables.Float64Atom(shape=()))
            SpikeM_i_PoisPyr = rawfile.create_vlarray(root, 'SpikeM_i_PoisPyr', tables.Float64Atom(shape=()))
            SpikeM_t_PoisInt = rawfile.create_vlarray(root, 'SpikeM_t_PoisInt', tables.Float64Atom(shape=()))
            SpikeM_i_PoisInt = rawfile.create_vlarray(root, 'SpikeM_i_PoisInt', tables.Float64Atom(shape=()))
            PopRateSig_PoisPyr = rawfile.create_vlarray(root, 'PopRateSig_PoisPyr', tables.Float64Atom(shape=()))
            PopRateSig_PoisInt = rawfile.create_vlarray(root, 'PopRateSig_PoisInt', tables.Float64Atom(shape=()))
    
        if record_vm:
            Vm_Pyr = rawfile.create_vlarray(root, 'Vm_Pyr', tables.Float64Atom(shape=(N_samples)))
            Vm_Int = rawfile.create_vlarray(root, 'Vm_Int', tables.Float64Atom(shape=(N_samples)))
        rawfile.close()
        
    if analyze:
        AvgCellRate_Pyr = np.zeros((len(PyrInps),len(IntInps)))
        SynchFreq_Pyr = np.zeros_like(AvgCellRate_Pyr)
        SynchFreqPow_Pyr = np.zeros_like(AvgCellRate_Pyr)
        PkWidth_Pyr = np.zeros_like(AvgCellRate_Pyr)
        Harmonics_Pyr = np.zeros_like(AvgCellRate_Pyr)
        SynchMeasure_Pyr = np.zeros_like(AvgCellRate_Pyr)
        PhaseShiftCurr_Pyr = np.zeros_like(AvgCellRate_Pyr)
        AvgCellRate_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchFreq_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchFreqPow_Int = np.zeros_like(AvgCellRate_Pyr)
        PkWidth_Int = np.zeros_like(AvgCellRate_Pyr)
        Harmonics_Int = np.zeros_like(AvgCellRate_Pyr)
        SynchMeasure_Int = np.zeros_like(AvgCellRate_Pyr)
        PhaseShiftCurr_Int = np.zeros_like(AvgCellRate_Pyr)
    
    if len(IPois_phs)<len(IPois_fs):
        IPois_phs = np.ones([len(IPois_fs)])*IPois_phs[0]
        
    for pi,IP_A in enumerate(IPois_As):
        
        for ii,IP_f in enumerate(IPois_fs):
            
            if verbose:
                print('[Starting simulation (%d/%d) for (IPois. Amp.: %s, IPois. Freq.: %s)...]' % (pi*len(IPois_fs)+ii+1, len(IPois_As)*len(IPois_fs), IP_A, IP_f))
            
            if save_raw:
                
                if not newfile:
                    rawfile = tables.open_file(filename+'_raw.h5', mode='a')
                    IPoisParamsList = rawfile.root.IPoisParams.read()
                    rawfile.close()
                    if len(IPoisParamsList)>0:
                        if any(all(np.array([IP_A, IP_f]) == IPoisParamsList, axis=1)):
                            print(' Already saved, skipping...')
                            continue
            
            gc.collect()
            
            Monitors = run_network_IP(N_p=N_p, N_i=N_i, PyrEqs=PyrEqs, IntEqs=IntEqs, PreEqAMPA=PreEqAMPA, PreEqGABA=PreEqGABA, PyrInp=PyrInp, IntInp=IntInp, IPois_A=IP_A, IPois_ph=IPois_phs[ii], IPois_Atype=IPois_Atype, IPois_f=IP_f, PP_C=PP_C, IP_C=IP_C, II_C=II_C, PI_C=PI_C, runtime=runtime, sim_dt=sim_dt, SynConns=SynConns, SaveSynns=SaveSynns, monitored=monitored, monitors_dt=monitors_dt, avgoneurons=avgoneurons, avgotime=avgotime, avgotimest=avgotimest, monitor_pois=monitor_pois, record_vm=record_vm, verbose=verbose, PlotFlag=False)
            
            if analyze:
                Network_feats = analyze_network(Monitors, comp_phase_curr=comp_phase_curr, N_p=N_p, N_i=N_i, start_time=start_time, end_time=end_time, sim_dt=sim_dt, mts_win=mts_win, W=W, ws=ws)
                
                AvgCellRate_Pyr[pi,ii], SynchFreq_Pyr[pi,ii], SynchFreqPow_Pyr[pi,ii], PkWidth_Pyr[pi,ii], Harmonics_Pyr[pi,ii], SynchMeasure_Pyr[pi,ii], AvgCellRate_Int[pi,ii], SynchFreq_Int[pi,ii], SynchFreqPow_Int[pi,ii], PkWidth_Int[pi,ii], Harmonics_Int[pi,ii], SynchMeasure_Int[pi,ii] = Network_feats['AvgCellRate_Pyr'], Network_feats['SynchFreq_Pyr'], Network_feats['SynchFreqPow_Pyr'], Network_feats['PkWidth_Pyr'], Network_feats['Harmonics_Pyr'], Network_feats['SynchMeasure_Pyr'], Network_feats['AvgCellRate_Int'], Network_feats['SynchFreq_Int'], Network_feats['SynchFreqPow_Int'], Network_feats['PkWidth_Int'], Network_feats['Harmonics_Int'], Network_feats['SynchMeasure_Int']
            
                if comp_phase_curr:
                    PhaseShiftCurr_Pyr[pi,ii], PhaseShiftCurr_Int[pi,ii] = Network_feats['PhaseShiftCurr_Pyr'], Network_feats['PhaseShiftCurr_Int']
                
            if save_raw:
                SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int, SynPP, SynIP, SynII, SynPI = Monitors['SpikeM_Pyr'], Monitors['PopRateM_Pyr'], Monitors['SpikeM_Int'], Monitors['PopRateM_Int'], Monitors['SynPP'], Monitors['SynIP'], Monitors['SynII'], Monitors['SynPI']

                rawfile = tables.open_file(filename+'_raw.h5', mode='a')
                SpikeM_t_Pyr = rawfile.root.SpikeM_t_Pyr
                SpikeM_t_Pyr.append(np.array(SpikeM_Pyr.t/ms)*ms)
                SpikeM_i_Pyr = rawfile.root.SpikeM_i_Pyr
                SpikeM_i_Pyr.append(np.array(SpikeM_Pyr.i))
                SpikeM_t_Int = rawfile.root.SpikeM_t_Int
                SpikeM_t_Int.append(np.array(SpikeM_Int.t/ms)*ms)
                SpikeM_i_Int = rawfile.root.SpikeM_i_Int
                SpikeM_i_Int.append(np.array(SpikeM_Int.i))
                
                PopRateSig_Pyr = rawfile.root.PopRateSig_Pyr
                PopRateSig_Pyr.append(PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
                PopRateSig_Int = rawfile.root.PopRateSig_Int
                PopRateSig_Int.append(PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
                if SaveSynns:
                    SynPP_i = rawfile.root.SynPP_i
                    SynPP_i.append(np.array(SynPP.i))
                    SynPP_j = rawfile.root.SynPP_j
                    SynPP_j.append(np.array(SynPP.j))
                    SynIP_i = rawfile.root.SynIP_i
                    SynIP_i.append(np.array(SynIP.i))
                    SynIP_j = rawfile.root.SynIP_j
                    SynIP_j.append(np.array(SynIP.j))
                    SynII_i = rawfile.root.SynII_i
                    SynII_i.append(np.array(SynII.i))
                    SynII_j = rawfile.root.SynII_j
                    SynII_j.append(np.array(SynII.j))
                    SynPI_i = rawfile.root.SynPI_i
                    SynPI_i.append(np.array(SynPI.i))
                    SynPI_j = rawfile.root.SynPI_j
                    SynPI_j.append(np.array(SynPI.j))
                    
                if not monitored==[]:
                    StateM_Pyr, StateM_Int = Monitors['StateM_Pyr'], Monitors['StateM_Int']
                    for i,var in enumerate(monitored): 
                        if avgoneurons:
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                            rawfile.get_node('/', name=var+'_Int').append(np.array(StateM_Int.get_states()[var]).mean(axis=1))
                        if avgotime:
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var])[int(avgotimest/monitors_dt),:].mean(axis=0))
                            rawfile.get_node('/', name=var+'_Int').append(np.array(StateM_Int.get_states()[var])[int(avgotimest/monitors_dt),:].mean(axis=0))
                        if not any((avgoneurons,avgotime)):
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var]))
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Int.get_states()[var]))
                            
                if monitor_pois:
                    SpikeM_PoisPyr, PopRateM_PoisPyr, SpikeM_PoisInt, PopRateM_PoisInt = Monitors['SpikeM_PoisPyr'], Monitors['PopRateM_PoisPyr'], Monitors['SpikeM_PoisInt'], Monitors['PopRateM_PoisInt']
                    SpikeM_t_PoisPyr = rawfile.root.SpikeM_t_PoisPyr
                    SpikeM_t_PoisPyr.append(np.array(SpikeM_PoisPyr.t/ms)*ms)
                    SpikeM_i_PoisPyr = rawfile.root.SpikeM_i_PoisPyr
                    SpikeM_i_PoisPyr.append(np.array(SpikeM_PoisPyr.i))
                    SpikeM_t_PoisInt = rawfile.root.SpikeM_t_PoisInt
                    SpikeM_t_PoisInt.append(np.array(SpikeM_PoisInt.t/ms)*ms)
                    SpikeM_i_PoisInt = rawfile.root.SpikeM_i_PoisInt
                    SpikeM_i_PoisInt.append(np.array(SpikeM_PoisInt.i))
                    PopRateSig_PoisPyr = rawfile.root.PopRateSig_PoisPyr
                    PRPois_Pyr = PopRateM_PoisPyr.smooth_rate(window='gaussian', width=1*ms)
                    PopRateSig_PoisPyr.append(PRPois_Pyr)
                    PopRateSig_PoisInt = rawfile.root.PopRateSig_PoisInt
                    PRPois_Int = PopRateM_PoisInt.smooth_rate(window='gaussian', width=1*ms)
                    PopRateSig_PoisInt.append(PRPois_Int)
                if record_vm:
                    StateM_Pyr, StateM_Int = Monitors['StateM_Pyr'], Monitors['StateM_Int']
                    Vm_Pyr = rawfile.root.Vm_Pyr
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,0])
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,-1])
                    Vm_Int = rawfile.root.Vm_Int
                    Vm_Int.append(StateM_Int.get_states()['v'][:,0])
                    Vm_Int.append(StateM_Int.get_states()['v'][:,-1])
                
                IPoisParams = rawfile.root.IPoisParams
                IPoisParams.append((IP_A, IP_f))
                
                rawfile.close()
        
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
        Sims_feats = {'AvgCellRate_Pyr':AvgCellRate_Pyr,
                      'SynchFreq_Pyr':SynchFreq_Pyr,
                      'SynchFreqPow_Pyr':SynchFreqPow_Pyr,
                      'PkWidth_Pyr':PkWidth_Pyr,
                      'Harmonics_Pyr':Harmonics_Pyr,
                      'SynchMeasure_Pyr':SynchMeasure_Pyr,
                      'AvgCellRate_Int':AvgCellRate_Int,
                      'SynchFreq_Int':SynchFreq_Int,
                      'SynchFreqPow_Int':SynchFreqPow_Int,
                      'PkWidth_Int':PkWidth_Int,
                      'Harmonics_Int':Harmonics_Int,
                      'SynchMeasure_Int':SynchMeasure_Int}
        if comp_phase_curr:
            Sims_feats['PhaseShiftCurr_Pyr'] = PhaseShiftCurr_Pyr
            Sims_feats['PhaseShiftCurr_Int'] = PhaseShiftCurr_Int
            
        return Sims_feats
    
#####################################################################################


def analyze_raw(filename, mode, N_p=4000, N_i=1000, start_time=None, end_time=None, sim_dt=0.02, comp_phase_curr=False, mts_win='whole', W=2**13, ws=None, verbose=False, printevery=None, PlotFlag=False, plot_file=None, out_file=None):
    '''
    Analyzes a pre-simulated network and extracts various features using the raw data saved under the provided file name
    
    * Parameters:
    - filename: The directory of the file under which raw data of the simulations are saved
    - mode: type of the inputs used in the simulations ('Homogenous' or 'Inhomogenous')
    - N_p: Number of pyramidal neurons in the simulated network
    - N_i: Number of interneurons in the simulated network
    - start_time: Beginning time of analysis within the simulation time [ms]
    - end_time: Ending time of analysis within the simulation time [ms]
    - sim_dt: Time step used in the simulation [ms]
    - comp_phase_curr: Whether to include phase shift calculations between of AMPA and GABA currents
    - mts_win: Whether to calculate the spectrum for the whole recording or as an average of moving window spectrums
    - W: Window length for the calculation of the multi-taper spectrum
    - ws: Sliding step of the moving window for muti-taper spectrum estimation
    - verbose: Whether to print progress texts during the analysis
    - PlotFlag: Whether to plot the results of the analysis
    - plot_file: Directory of the image file under which the results' plot is to be saved
    - out_file: Directory of the output file under which the analysis results are to be saved
    
    * Returns:
    - Sims_feats: A dictinary of the different features calculated from the recorded data of all simulations
    '''
    
    sim_dt *= ms
    
    if ws is None:
        ws = W/10
    
    rawfile = tables.open_file(filename, mode='r')
    
    if mode is 'Homogenous':
        IterArray1 = (rawfile.root.PyrInps.read()/1000)*kHz
        IterArray2 = (rawfile.root.IntInps.read()/1000)*kHz
    else:
        IterArray1 = rawfile.root.IPois_As.read()
        IterArray2 = (rawfile.root.IPois_fs.read())*Hz
    
    PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr.read()
    PopRateSig_Int_list = rawfile.root.PopRateSig_Int.read()
    
    Spike_t_Pyr_list = rawfile.root.SpikeM_t_Pyr.read()
    Spike_i_Pyr_list = rawfile.root.SpikeM_i_Pyr.read()
    Spike_t_Int_list = rawfile.root.SpikeM_t_Int.read()
    Spike_i_Int_list = rawfile.root.SpikeM_i_Int.read()
    
    if comp_phase_curr:
        I_AMPA_Pyr_list = rawfile.root.IsynP_Pyr.read()
        I_GABA_Pyr_list = rawfile.root.IsynI_Pyr.read()
        I_AMPA_Int_list = rawfile.root.IsynP_Int.read()
        I_GABA_Int_list = rawfile.root.IsynI_Int.read()
        
    rawfile.close()
    
    runtime = len(PopRateSig_Pyr_list[0])*sim_dt/ms
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = runtime
    
    if start_time > runtime or end_time > runtime:
        raise ValueError('Please provide start time and end time within the simulation time window!')
        
    AvgCellRate_Pyr = np.zeros((len(IterArray1),len(IterArray2)))
    SynchFreq_Pyr = np.zeros_like(AvgCellRate_Pyr)
    SynchFreqPow_Pyr = np.zeros_like(AvgCellRate_Pyr)
    PkWidth_Pyr = np.zeros_like(AvgCellRate_Pyr)
    Harmonics_Pyr = np.zeros_like(AvgCellRate_Pyr)
    SynchMeasure_Pyr = np.zeros_like(AvgCellRate_Pyr)
    PhaseShiftCurr_Pyr = np.zeros_like(AvgCellRate_Pyr)
    
    AvgCellRate_Int = np.zeros_like(AvgCellRate_Pyr)
    SynchFreq_Int = np.zeros_like(AvgCellRate_Pyr)
    SynchFreqPow_Int = np.zeros_like(AvgCellRate_Pyr)
    PkWidth_Int = np.zeros_like(AvgCellRate_Pyr)
    Harmonics_Int = np.zeros_like(AvgCellRate_Pyr)
    SynchMeasure_Int = np.zeros_like(AvgCellRate_Pyr)
    PhaseShiftCurr_Int = np.zeros_like(AvgCellRate_Pyr)
    
    AvgCellRate_Full = np.zeros_like(AvgCellRate_Pyr)
    SynchFreq_Full = np.zeros_like(AvgCellRate_Pyr)
    SynchFreqPow_Full = np.zeros_like(AvgCellRate_Pyr)
    PkWidth_Full = np.zeros_like(AvgCellRate_Pyr)
    Harmonics_Full = np.zeros_like(AvgCellRate_Pyr)
    SynchMeasure_Full = np.zeros_like(AvgCellRate_Pyr)
    PhaseShift_PyrInt = np.zeros_like(AvgCellRate_Pyr)
    
    for pi,IterItem1 in enumerate(IterArray1):
        for ii,IterItem2 in enumerate(IterArray2):
            
            if verbose:
                if not printevery is None:
                    if (pi*len(IterArray2)+ii)%printevery==0:
                        print('Analyzing network %d/%d...' %(pi*len(IterArray2)+ii+1, len(IterArray1)*len(IterArray2)))
            
            idx = pi*len(IterArray2)+ii
            
            if int((start_time*ms)/sim_dt) > len(PopRateSig_Pyr_list[idx]):
                raise ValueError('Please provide start time and end time within the simulation time window!')
            
            RateSigRaw_Pyr = PopRateSig_Pyr_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
            RateSigRaw_Int = PopRateSig_Int_list[idx][int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
            RateSigRaw_Full = (N_p*RateSigRaw_Pyr+N_i*RateSigRaw_Int)/np.float(N_p+N_i)

            Spike_t_Pyr = Spike_t_Pyr_list[idx]
            Spike_i_Pyr = Spike_i_Pyr_list[idx]
            Spike_t_Int = Spike_t_Int_list[idx]
            Spike_i_Int = Spike_i_Int_list[idx]

            rates_Pyr = np.zeros(N_p)
            spikes_Pyr = np.asarray([n for j,n in enumerate(Spike_i_Pyr) if Spike_t_Pyr[j]/(0.001) >= start_time and Spike_t_Pyr[j]/(0.001) <= end_time])
            for j in range(N_p):
                rates_Pyr[j] = sum(spikes_Pyr==j)/((end_time-start_time)*ms)

            rates_Int = np.zeros(N_i)
            spikes_Int = np.asarray([n for j,n in enumerate(Spike_i_Int) if Spike_t_Int[j]/(0.001) >= start_time and Spike_t_Int[j]/(0.001) <= end_time])
            for j in range(N_i):
                rates_Int[j] = sum(spikes_Int==j)/((end_time-start_time)*ms)
        
            rates_Full = np.append(rates_Pyr, rates_Int)
            AvgCellRate_Pyr[pi,ii] = np.mean(rates_Pyr*Hz)
            AvgCellRate_Int[pi,ii] = np.mean(rates_Int*Hz)
            AvgCellRate_Full[pi,ii] = np.mean(rates_Full*Hz)

            fs = 1/(sim_dt)
            fmax = fs/2

            RateSig_Pyr = RateSigRaw_Pyr-np.mean(RateSigRaw_Pyr)

            RateSig_Int = RateSigRaw_Int-np.mean(RateSigRaw_Int)
            
            RateSig_Full = RateSigRaw_Full-np.mean(RateSigRaw_Full)
    
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
                
                a = pmtm(RateSig_Full, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                RateMTS_Full = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                RateMTS_Full = RateMTS_Full[np.where(freq_vect/Hz<=300)]
        
            else:

                NFFT=W*2
                freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]

                N = len(RateSig_Pyr)
                N_segs = int((N-W)/ws+2)
                result = np.zeros((NFFT/2))
                for i in range(N_segs):
                    data = RateSig_Pyr[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    result += Sks[:NFFT/2]/W
                RateMTS_Pyr = result[np.where(freq_vect/Hz<=300)]/N_segs

                N = len(RateSig_Int)
                N_segs = int((N-W)/ws+2)
                result = np.zeros((NFFT/2))
                for i in range(N_segs):
                    data = RateSig_Int[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    result += Sks[:NFFT/2]/W  
                RateMTS_Int = result[np.where(freq_vect/Hz<=300)]/N_segs
                
                N = len(RateSig_Full)
                N_segs = int((N-W)/ws+2)
                result = np.zeros((NFFT/2))
                for i in range(N_segs):
                    data = RateSig_Full[i*ws:i*ws+W]
                    a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                    Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
                    result += Sks[:NFFT/2]/W  
                RateMTS_Full = result[np.where(freq_vect/Hz<=300)]/N_segs

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

                corr_sig = np.correlate(RateSigRaw_Pyr, RateSigRaw_Pyr, mode='full')
                corr_sig = corr_sig[int(len(corr_sig)/2):]/(N*np.mean(RateSigRaw_Pyr)**2)
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

                corr_sig = np.correlate(RateSigRaw_Int, RateSigRaw_Int, mode='full')
                corr_sig = corr_sig[int(len(corr_sig)/2):]/(N*np.mean(RateSigRaw_Int)**2)
                maxtab, mintab = peakdet(corr_sig, 0.1)
                if not mintab.any():
                    SynchMeasure_Int[pi,ii] = float('nan')
                else:
                    SynchMeasure_Int[pi,ii] = (corr_sig[0] - mintab[0,1])
            
            ### Full:
            
            if np.max(RateMTS_Full)==0:
                SynchFreq_Full[pi,ii] = float('nan')
                SynchFreqPow_Full[pi,ii] = float('nan')
                PkWidth_Full[pi,ii] = float('nan')
                Harmonics_Full[pi,ii] = float('nan')
                SynchMeasure_Full[pi,ii] = float('nan')
                PhaseShift_PyrInt[pi,ii] = float('nan')
            else:
                SynchFreq_Full[pi,ii] = freq_vect[np.argmax(RateMTS_Full)]
                SynchFreqPow_Full[pi,ii] = np.max(RateMTS_Full)

                N_sm = 5
                smoothed = np.convolve(RateMTS_Full, np.ones([N_sm])/N_sm, mode='same')
                freq_vect_sm = freq_vect[:len(smoothed)]

                maxInd = np.argmax(RateMTS_Full)
                if np.argmax(RateMTS_Full) < 5:
                    maxInd = np.argmax(RateMTS_Full[10:])+10

                maxInd_sm = np.argmax(smoothed)
                if np.argmax(smoothed) < 5:
                    maxInd_sm = np.argmax(smoothed[10:])+10

                bline = np.mean(RateMTS_Full[freq_vect<300*Hz])
                bline_drop = np.max(RateMTS_Full)/6
                pk_offset = freq_vect_sm[(np.where(smoothed[maxInd_sm:]<bline_drop)[0][0]+maxInd_sm)]
                if not (np.where(smoothed[:maxInd_sm]<bline)[0]).any():
                    maxtab, mintab = peakdet(smoothed, bline/2.)
                    if not [minma[0] for minma in mintab if minma[0] < maxInd_sm]:
                        pk_onset = freq_vect_sm[0]
                    else:
                        pk_onset = freq_vect_sm[int([minma[0] for minma in mintab if minma[0] < maxInd_sm][-1])]
                else:
                    pk_onset = freq_vect_sm[np.where(smoothed[:maxInd_sm]<bline)[0][-1]]
                PkWidth_Full[pi,ii] = pk_offset-pk_onset

                maxtab, mintab = peakdet(smoothed, np.max(smoothed)/3.)
                harms = [mxma for mxma in maxtab if (mxma[0]-maxInd_sm > 20 or mxma[0]-maxInd_sm < -20) and mxma[1] >= np.max(smoothed)/2.0]
                harms = np.array(harms)
                rem_inds = []
                for jj in range(len(harms))[:-1]:
                    if harms[jj][0] - harms[jj+1][0] > -10:
                        rem_inds.append(jj)
                harms = np.delete(harms, rem_inds, axis=0)
                Harmonics_Full[pi,ii] = len(harms)

                corr_sig = np.correlate(RateSigRaw_Full, RateSigRaw_Full, mode='full')
                corr_sig = corr_sig[int(len(corr_sig)/2):]/(N*np.mean(RateSigRaw_Full)**2)
                maxtab, mintab = peakdet(corr_sig, 0.1)
                if not mintab.any():
                    SynchMeasure_Full[pi,ii] = float('nan')
                else:
                    SynchMeasure_Full[pi,ii] = (corr_sig[0] - mintab[0,1])
                
#                 RateSigReg_Pyr = (RateSigRaw_Pyr-np.mean(RateSigRaw_Pyr))/np.std(RateSigRaw_Pyr)
#                 RateSigReg_Int = (RateSigRaw_Int-np.mean(RateSigRaw_Int))/np.std(RateSigRaw_Int)
#                 corr_sig = np.correlate(RateSigReg_Pyr, RateSigReg_Int, mode='full')
                fpeak = freq_vect[np.argmax(RateMTS_Full)]
                f, Pxx = signal.csd(RateSigRaw_Pyr, RateSigRaw_Pyr, fs=fs/Hz, nperseg=W)
                f, Pyy = signal.csd(RateSigRaw_Int, RateSigRaw_Int, fs=fs/Hz, nperseg=W)
                f, Pxy = signal.csd(RateSigRaw_Pyr, RateSigRaw_Int, fs=fs/Hz, nperseg=W)
                Cxy_phs = np.angle(Pxy/(Pxx*Pyy), deg=True)
                Cxy_ph = Cxy_phs[np.argmin(abs(f-fpeak/Hz))]
                PhaseShift_PyrInt[pi,ii] = Cxy_ph
            
                    
            if comp_phase_curr:
                # Pyr.:
                I_AMPA = I_AMPA_Pyr_list[idx]/(1e-9)
                I_AMPA = I_AMPA[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                I_AMPA -= np.mean(I_AMPA)
                I_GABA = I_GABA_Pyr_list[idx]/(1e-9)
                I_GABA = I_GABA[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                I_GABA -= np.mean(I_GABA)

                N = I_AMPA.shape[0]
                NFFT = 2**(N-1).bit_length()
                freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
                a = pmtm(I_GABA, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                I_MTS = I_MTS[np.where(freq_vect/Hz<=300)]
                fpeak = freq_vect[np.argmax(I_MTS)]

                corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
                phases = np.arange(1-N, N)

                PhaseShift = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
                PhaseShiftCurr_Pyr[pi,ii] = np.sign(PhaseShift)*(abs(PhaseShift)%360)

                # Int.:
                I_AMPA = I_AMPA_Int_list[idx]/(1e-9)
                I_AMPA = I_AMPA[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                I_AMPA -= np.mean(I_AMPA)
                I_GABA = I_GABA_Int_list[idx]/(1e-9)
                I_GABA = I_GABA[int(start_time*ms/sim_dt):int(end_time*ms/sim_dt)]
                I_GABA -= np.mean(I_GABA)

                N = I_AMPA.shape[0]
                NFFT = 2**(N-1).bit_length()
                freq_vect = np.linspace(0, fmax/Hz, NFFT/2)*Hz
                freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
                a = pmtm(I_GABA, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                I_MTS = I_MTS[np.where(freq_vect/Hz<=300)]
                fpeak = freq_vect[np.argmax(I_MTS)]

                corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
                phases = np.arange(1-N, N)

                PhaseShift = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
                PhaseShiftCurr_Int[pi,ii] = np.sign(PhaseShift)*(abs(PhaseShift)%360)

    if not (out_file is None):
        h5file = tables.open_file(out_file+'.hf5', mode='w', title='Analysis')
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
        h5file.create_carray(root, "AvgCellRate_Full", obj=AvgCellRate_Full)
        h5file.create_carray(root, "SynchFreq_Full", obj=SynchFreq_Full)
        h5file.create_carray(root, "SynchFreqPow_Full", obj=SynchFreqPow_Full)
        h5file.create_carray(root, "PkWidth_Full", obj=PkWidth_Full)
        h5file.create_carray(root, "Harmonics_Full", obj=Harmonics_Full)
        h5file.create_carray(root, "SynchMeasure_Full", obj=SynchMeasure_Full)
        h5file.create_carray(root, "PhaseShift_PyrInt", obj=PhaseShift_PyrInt)
        h5file.create_carray(root, "IterArray1", obj=IterArray1)
        h5file.create_carray(root, "IterArray2", obj=IterArray2)
        if comp_phase_curr:
            h5file.create_carray(root, "PhaseShiftCurr_Pyr", obj=PhaseShiftCurr_Pyr)
            h5file.create_carray(root, "PhaseShiftCurr_Int", obj=PhaseShiftCurr_Int)
        h5file.close()
        if verbose:
            print('Saved analysis results successfully!')
    
    Sims_feats = {'AvgCellRate_Pyr':AvgCellRate_Pyr,
                  'SynchFreq_Pyr':SynchFreq_Pyr,
                  'SynchFreqPow_Pyr':SynchFreqPow_Pyr,
                  'PkWidth_Pyr':PkWidth_Pyr,
                  'Harmonics_Pyr':Harmonics_Pyr,
                  'SynchMeasure_Pyr':SynchMeasure_Pyr,
                  'AvgCellRate_Int':AvgCellRate_Int,
                  'SynchFreq_Int':SynchFreq_Int,
                  'SynchFreqPow_Int':SynchFreqPow_Int,
                  'PkWidth_Int':PkWidth_Int,
                  'Harmonics_Int':Harmonics_Int,
                  'SynchMeasure_Int':SynchMeasure_Int,
                  'AvgCellRate_Full':AvgCellRate_Full,
                  'SynchFreq_Full':SynchFreq_Full,
                  'SynchFreqPow_Full':SynchFreqPow_Full,
                  'PkWidth_Full':PkWidth_Full,
                  'Harmonics_Full':Harmonics_Full,
                  'SynchMeasure_Full':SynchMeasure_Full,
                  'PhaseShift_PyrInt':PhaseShift_PyrInt}
    if comp_phase_curr:
        Sims_feats['PhaseShiftCurr_Pyr'] = PhaseShiftCurr_Pyr
        Sims_feats['PhaseShiftCurr_Int'] = PhaseShiftCurr_Int
            
    if PlotFlag:
        plot_results(IterArray1, IterArray2, mode, Sims_feats, plot_file)
   
    return Sims_feats


#################################################################################

def analyze_raw_modes(filename, mode, N_p=4000, N_i=1000, start_time=None, end_time=None, sim_dt=0.02, comp_phase_curr=False, mts_win='whole', W=2**12, ws=None, verbose=False, printevery=None, PlotFlag=False, plot_file=None, out_file=None):
    '''
    Analyzes a pre-simulated network and extracts various features using the raw data saved under the provided file name
    
    * Parameters:
    - filename: The directory of the file under which raw data of the simulations are saved
    - mode: type of the inputs used in the simulations ('Homogenous' or 'Inhomogenous')
    - N_p: Number of pyramidal neurons in the simulated network
    - N_i: Number of interneurons in the simulated network
    - start_time: Beginning time of analysis within the simulation time [ms]
    - end_time: Ending time of analysis within the simulation time [ms]
    - sim_dt: Time step used in the simulation [ms]
    - comp_phase_curr: Whether to include phase shift calculations between of AMPA and GABA currents
    - mts_win: Whether to calculate the spectrum for the whole recording or as an average of moving window spectrums
    - W: Window length for the calculation of the multi-taper spectrum
    - ws: Sliding step of the moving window for muti-taper spectrum estimation
    - verbose: Whether to print progress texts during the analysis
    - PlotFlag: Whether to plot the results of the analysis
    - plot_file: Directory of the image file under which the results' plot is to be saved
    - out_file: Directory of the output file under which the analysis results are to be saved
    
    * Returns:
    - Sims_feats: A dictinary of the different features calculated from the recorded data of all simulations
    '''
    
    if ws is None:
        ws = W/10
    
    rawfile = tables.open_file(filename, mode='r')
    
    if mode is 'Homogenous':
        IterArray1 = rawfile.root.PyrInps.read()*Hz
        IterArray2 = rawfile.root.IntInps.read()*Hz
    else:
        IterArray1 = rawfile.root.IPois_As.read()
        IterArray2 = (rawfile.root.IPois_fs.read())*Hz
    
    PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr.read()
    PopRateSig_Int_list = rawfile.root.PopRateSig_Int.read()
    
    Spike_t_Pyr_list = rawfile.root.SpikeM_t_Pyr.read()
    Spike_i_Pyr_list = rawfile.root.SpikeM_i_Pyr.read()
    Spike_t_Int_list = rawfile.root.SpikeM_t_Int.read()
    Spike_i_Int_list = rawfile.root.SpikeM_i_Int.read()
    
    if comp_phase_curr:
        I_AMPA_Pyr_list = rawfile.root.IsynP_Pyr.read()
        I_GABA_Pyr_list = rawfile.root.IsynI_Pyr.read()
        I_AMPA_Int_list = rawfile.root.IsynP_Int.read()
        I_GABA_Int_list = rawfile.root.IsynI_Int.read()
        
    rawfile.close()
    
    runtime = len(PopRateSig_Pyr_list[0])*sim_dt
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = runtime
    
    if start_time > runtime or end_time > runtime:
        raise ValueError('Please provide start time and end time within the simulation time window!')
        
    SynchFreq_Pyr = {'PING': np.zeros((len(IterArray1),len(IterArray2))),
                     'ING': np.zeros((len(IterArray1),len(IterArray2)))}
    SynchFreqPow_Pyr = copy.deepcopy(SynchFreq_Pyr)
    PhaseShift_Pyr = copy.deepcopy(SynchFreq_Pyr)
    AvgCellRate_Pyr = np.zeros((len(IterArray1),len(IterArray2)))
    SynchMeasure_Pyr = np.zeros_like(AvgCellRate_Pyr)
    PhaseShiftCurr_Pyr = np.zeros_like(AvgCellRate_Pyr)
    
    SynchFreq_Int = {'PING': np.zeros((len(IterArray1),len(IterArray2))),
                     'ING': np.zeros((len(IterArray1),len(IterArray2)))}
    SynchFreqPow_Int = copy.deepcopy(SynchFreq_Int)
    PhaseShift_Int = copy.deepcopy(SynchFreq_Int)
    AvgCellRate_Int = np.zeros((len(IterArray1),len(IterArray2)))
    SynchMeasure_Int = np.zeros_like(AvgCellRate_Int)
    PhaseShiftCurr_Int = np.zeros_like(AvgCellRate_Int)
    
    for pi,IterItem1 in enumerate(IterArray1):
        for ii,IterItem2 in enumerate(IterArray2):
            
            if verbose:
                if not printevery is None:
                    if (pi*len(IterArray2)+ii)%printevery==0:
                        print('Analyzing network %d/%d...' %(pi*len(IterArray2)+ii+1, len(IterArray1)*len(IterArray2)))
            
            idx = pi*len(IterArray2)+ii
            
            if int((start_time)/sim_dt) > len(PopRateSig_Pyr_list[idx]):
                raise ValueError('Please provide start time and end time within the simulation time window!')
            
            RateSigRaw_Pyr = PopRateSig_Pyr_list[idx][int(start_time/sim_dt):int(end_time/sim_dt)]
            RateSigRaw_Int = PopRateSig_Int_list[idx][int(start_time/sim_dt):int(end_time/sim_dt)]

            Spike_t_Pyr = Spike_t_Pyr_list[idx]
            Spike_i_Pyr = Spike_i_Pyr_list[idx]
            Spike_t_Int = Spike_t_Int_list[idx]
            Spike_i_Int = Spike_i_Int_list[idx]

            rates_Pyr = np.zeros(N_p)
            spikes_Pyr = np.asarray([n for j,n in enumerate(Spike_i_Pyr) if Spike_t_Pyr[j]/(0.001) >= start_time and Spike_t_Pyr[j]/(0.001) <= end_time])
            for j in range(N_p):
                rates_Pyr[j] = sum(spikes_Pyr==j)/((end_time-start_time)*1e-3)

            rates_Int = np.zeros(N_i)
            spikes_Int = np.asarray([n for j,n in enumerate(Spike_i_Int) if Spike_t_Int[j]/(0.001) >= start_time and Spike_t_Int[j]/(0.001) <= end_time])
            for j in range(N_i):
                rates_Int[j] = sum(spikes_Int==j)/((end_time-start_time)*1e-3)
        
            AvgCellRate_Pyr[pi,ii] = np.mean(rates_Pyr)
            AvgCellRate_Int[pi,ii] = np.mean(rates_Int)

            fs = np.round(1/(sim_dt*1e-3))
            fmax = np.round(fs/2)
            
            N_whole = len(PopRateSig_Pyr_list[idx])
            T_whole = (N_whole/fs)*1000.
            time_whole = np.arange(0,T_whole,sim_dt)
            time_v = time_whole[int(start_time/sim_dt):]
            N = len(time_v)
            N_segs = int((N-W)/ws+2)
            MTS_time = time_v[np.arange(W/2, (N_segs)*ws+(W/2), ws).astype(int)]

            RateSig_Pyr = RateSigRaw_Pyr-np.mean(RateSigRaw_Pyr)
            RateSig_Int = RateSigRaw_Int-np.mean(RateSigRaw_Int)
    
            MTS_Pyr, freq_vect = comp_mtspectrogram(RateSig_Pyr, fs=fs, freq_limit=300, W=W, PlotFlag=False)
            MTSavg_Pyr = np.mean(MTS_Pyr, axis=1)
            MTS_Int, freq_vect = comp_mtspectrogram(RateSig_Int, fs=fs, freq_limit=300, W=W, PlotFlag=False)
            MTSavg_Int = np.mean(MTS_Int, axis=1)
            
            pyrthresh = np.mean(MTS_Pyr)+2*np.std(MTS_Pyr)
            MaxInds_Pyr = peak_local_max(MTS_Pyr, threshold_abs=pyrthresh, min_distance=2)
            intthresh = np.mean(MTS_Int)+2*np.std(MTS_Int)
            MaxInds_Int = peak_local_max(MTS_Int, threshold_abs=intthresh, min_distance=2)

            nW,binW = histogram(freq_vect[MaxInds_Pyr[:,0]], bins=20)
            binW += 0.5*np.diff(binW)[0]
            binW = binW[:-1]
            winfreq = binW[np.argmax(nW)]
            mindif = 20/np.diff(binW)[0]
            histmxind_W = findpeaks(nW, x_mindist=mindif, maxima_minval=np.mean(nW[nW>0]))
            if len(histmxind_W)>1:
                Midfreq_Pyr = np.mean(binW[histmxind_W])
            else:
                Midfreq_Pyr = 105.
                
            nW,binW = histogram(freq_vect[MaxInds_Int[:,0]], bins=20)
            binW += 0.5*np.diff(binW)[0]
            binW = binW[:-1]
            winfreq = binW[np.argmax(nW)]
            mindif = 20/np.diff(binW)[0]
            histmxind_W = findpeaks(nW, x_mindist=mindif, maxima_minval=np.mean(nW[nW>0]))
            if len(histmxind_W)>1:
                Midfreq_Int = np.mean(binW[histmxind_W])
            else:
                Midfreq_Int = 105.

            LowModeInds_Pyr = np.array([(f,t) for f,t in MaxInds_Pyr if freq_vect[f]<Midfreq_Pyr])
            LowModeInds_Int = np.array([(f,t) for f,t in MaxInds_Int if freq_vect[f]<Midfreq_Int])
            
            HighModeInds_Pyr = np.array([])
            HighModeInds_Int = np.array([])
            if IterItem1-IterItem2 < 2*kHz:
                HighModeInds_Pyr = np.array([(f,t) for f,t in MaxInds_Pyr if freq_vect[f]>=Midfreq_Pyr])
                HighModeInds_Int = np.array([(f,t) for f,t in MaxInds_Int if freq_vect[f]>=Midfreq_Int])
            
            MTSavg_PyrPING = np.zeros_like(freq_vect)
            MTSavg_IntPING = np.zeros_like(freq_vect)
            MTSavg_PyrING = np.zeros_like(freq_vect)
            MTSavg_IntING = np.zeros_like(freq_vect)
            if len(LowModeInds_Pyr)>0:
                MTS_PyrPING = MTS_Pyr[:, LowModeInds_Pyr[:,1]]
                MTSavg_PyrPING = np.mean(MTS_PyrPING, axis=1)
            if len(LowModeInds_Int)>0:
                MTS_IntPING = MTS_Int[:, LowModeInds_Int[:,1]]
                MTSavg_IntPING = np.mean(MTS_IntPING, axis=1)
            if len(HighModeInds_Pyr)>0:
                MTS_PyrING = MTS_Pyr[:, HighModeInds_Pyr[:,1]]
                MTSavg_PyrING = np.mean(MTS_PyrING, axis=1)
            if len(HighModeInds_Int)>0:
                MTS_IntING = MTS_Int[:, HighModeInds_Int[:,1]]
                MTSavg_IntING = np.mean(MTS_IntING, axis=1)
                
            ##### Pyr.:

            if np.max(MTSavg_Pyr)==0:
                SynchMeasure_Pyr[pi,ii] = float('nan')
            else:
                corr_sig = np.correlate(RateSigRaw_Pyr, RateSigRaw_Pyr, mode='full')
                corr_sig = corr_sig[int(len(corr_sig)/2):]/(N*np.mean(RateSigRaw_Pyr)**2)
                maxtab, mintab = peakdet(corr_sig, 0.1)
                if not mintab.any():
                    SynchMeasure_Pyr[pi,ii] = float('nan')
                else:
                    SynchMeasure_Pyr[pi,ii] = (corr_sig[0] - mintab[0,1])
            
            if np.max(MTSavg_PyrPING)==0:
                SynchFreq_Pyr['PING'][pi,ii] = float('nan')
                SynchFreqPow_Pyr['PING'][pi,ii] = float('nan')
            else:
                SynchFreq_Pyr['PING'][pi,ii] = freq_vect[np.argmax(MTSavg_PyrPING)]
                SynchFreqPow_Pyr['PING'][pi,ii] = np.max(MTSavg_PyrPING)
                
            if np.max(MTSavg_PyrING)==0:
                SynchFreq_Pyr['ING'][pi,ii] = float('nan')
                SynchFreqPow_Pyr['ING'][pi,ii] = float('nan')
            else:
                SynchFreq_Pyr['ING'][pi,ii] = freq_vect[np.argmax(MTSavg_PyrING)]
                SynchFreqPow_Pyr['ING'][pi,ii] = np.max(MTSavg_PyrING)
                
            fpeak = freq_vect[np.argmax(MTSavg_PyrPING)]
            Cxy_ph = np.zeros(LowModeInds_Pyr.shape[0])
            sigs_pyr = []
            sigs_int = []
            for bursti, burst in enumerate(LowModeInds_Pyr):
                raw_ind = np.where(time_v==MTS_time[burst[1]])[0][0]
                segment_pyr = np.copy(RateSigRaw_Pyr[int(raw_ind-W/2):int(raw_ind+W/2)])
                segment_pyr -= np.mean(segment_pyr)
                sigs_pyr.append(segment_pyr)
                segment_int = np.copy(RateSigRaw_Int[int(raw_ind-W/2):int(raw_ind+W/2)])
                segment_int -= np.mean(segment_int)
                sigs_int.append(segment_int)
            if len(sigs_pyr)>0 and len(sigs_int)>0:
                sig_pyr = np.concatenate(sigs_pyr)
                sig_int = np.concatenate(sigs_int)
                f, Pxx = signal.csd(sig_pyr, sig_pyr, fs=fs, nperseg=W)
                f, Pyy = signal.csd(sig_int, sig_int, fs=fs, nperseg=W)
                f, Pxy = signal.csd(sig_pyr, sig_int, fs=fs, nperseg=W)
                Cxy_phs = np.angle(Pxy/(Pxx*Pyy))
                PhaseShift_Pyr['PING'][pi,ii] = Cxy_phs[np.argmin(abs(f-fpeak))]*(180./np.pi)
            else:
                PhaseShift_Pyr['PING'][pi,ii] = float('nan')
            
            fpeak = freq_vect[np.argmax(MTSavg_PyrING)]
            Cxy_ph = np.zeros(HighModeInds_Pyr.shape[0])
            sigs_pyr = []
            sigs_int = []
            for bursti, burst in enumerate(HighModeInds_Pyr):
                raw_ind = np.where(time_v==MTS_time[burst[1]])[0][0]
                segment_pyr = np.copy(RateSigRaw_Pyr[int(raw_ind-W/2):int(raw_ind+W/2)])
                segment_pyr -= np.mean(segment_pyr)
                sigs_pyr.append(segment_pyr)
                segment_int = np.copy(RateSigRaw_Int[int(raw_ind-W/2):int(raw_ind+W/2)])
                segment_int -= np.mean(segment_int)
                sigs_int.append(segment_int)
            if len(sigs_pyr)>0 and len(sigs_int)>0:
                sig_pyr = np.concatenate(sigs_pyr)
                sig_int = np.concatenate(sigs_int)
                f, Pxx = signal.csd(sig_pyr, sig_pyr, fs=fs, nperseg=W)
                f, Pyy = signal.csd(sig_int, sig_int, fs=fs, nperseg=W)
                f, Pxy = signal.csd(sig_pyr, sig_int, fs=fs, nperseg=W)
                Cxy_phs = np.angle(Pxy/(Pxx*Pyy))
                PhaseShift_Pyr['ING'][pi,ii] = Cxy_phs[np.argmin(abs(f-fpeak))]*(180./np.pi)
            else:
                PhaseShift_Pyr['ING'][pi,ii] = float('nan')
            
            ##### Int.:
            
            if np.max(MTSavg_Int)==0:
                SynchMeasure_Int[pi,ii] = float('nan')
            else:
                corr_sig = np.correlate(RateSigRaw_Int, RateSigRaw_Int, mode='full')
                corr_sig = corr_sig[int(len(corr_sig)/2):]/(N*np.mean(RateSigRaw_Int)**2)
                maxtab, mintab = peakdet(corr_sig, 0.1)
                if not mintab.any():
                    SynchMeasure_Int[pi,ii] = float('nan')
                else:
                    SynchMeasure_Int[pi,ii] = (corr_sig[0] - mintab[0,1])
                
            if np.max(MTSavg_IntPING)==0:
                SynchFreq_Int['PING'][pi,ii] = float('nan')
                SynchFreqPow_Int['PING'][pi,ii] = float('nan')
            else:
                SynchFreq_Int['PING'][pi,ii] = freq_vect[np.argmax(MTSavg_IntPING)]
                SynchFreqPow_Int['PING'][pi,ii] = np.max(MTSavg_IntPING)
                
            if np.max(MTSavg_IntING)==0:
                SynchFreq_Int['ING'][pi,ii] = float('nan')
                SynchFreqPow_Int['ING'][pi,ii] = float('nan')
            else:
                SynchFreq_Int['ING'][pi,ii] = freq_vect[np.argmax(MTSavg_IntING)]
                SynchFreqPow_Int['ING'][pi,ii] = np.max(MTSavg_IntING)
                
            fpeak = freq_vect[np.argmax(MTSavg_IntPING)]
            Cxy_ph = np.zeros(LowModeInds_Int.shape[0])
            sigs_pyr = []
            sigs_int = []
            for bursti, burst in enumerate(LowModeInds_Int):
                raw_ind = np.where(time_v==MTS_time[burst[1]])[0][0]
                segment_pyr = np.copy(RateSigRaw_Pyr[int(raw_ind-W/2):int(raw_ind+W/2)])
                segment_pyr -= np.mean(segment_pyr)
                sigs_pyr.append(segment_pyr)
                segment_int = np.copy(RateSigRaw_Int[int(raw_ind-W/2):int(raw_ind+W/2)])
                segment_int -= np.mean(segment_int)
                sigs_int.append(segment_int)
            if len(sigs_pyr)>0 and len(sigs_int)>0:
                sig_pyr = np.concatenate(sigs_pyr)
                sig_int = np.concatenate(sigs_int)
                f, Pxx = signal.csd(sig_pyr, sig_pyr, fs=fs, nperseg=W)
                f, Pyy = signal.csd(sig_int, sig_int, fs=fs, nperseg=W)
                f, Pxy = signal.csd(sig_pyr, sig_int, fs=fs, nperseg=W)
                Cxy_phs = np.angle(Pxy/(Pxx*Pyy))
                PhaseShift_Int['PING'][pi,ii] = Cxy_phs[np.argmin(abs(f-fpeak))]*(180./np.pi)
            else:
                PhaseShift_Int['PING'][pi,ii] = float('nan')
            
            fpeak = freq_vect[np.argmax(MTSavg_IntING)]
            Cxy_ph = np.zeros(HighModeInds_Int.shape[0])
            sigs_pyr = []
            sigs_int = []
            for bursti, burst in enumerate(HighModeInds_Int):
                raw_ind = np.where(time_v==MTS_time[burst[1]])[0][0]
                segment_pyr = np.copy(RateSigRaw_Pyr[int(raw_ind-W/2):int(raw_ind+W/2)])
                segment_pyr -= np.mean(segment_pyr)
                sigs_pyr.append(segment_pyr)
                segment_int = np.copy(RateSigRaw_Int[int(raw_ind-W/2):int(raw_ind+W/2)])
                segment_int -= np.mean(segment_int)
                sigs_int.append(segment_int)
            if len(sigs_pyr)>0 and len(sigs_int)>0:
                sig_pyr = np.concatenate(sigs_pyr)
                sig_int = np.concatenate(sigs_int)
                f, Pxx = signal.csd(sig_pyr, sig_pyr, fs=fs, nperseg=W)
                f, Pyy = signal.csd(sig_int, sig_int, fs=fs, nperseg=W)
                f, Pxy = signal.csd(sig_pyr, sig_int, fs=fs, nperseg=W)
                Cxy_phs = np.angle(Pxy/(Pxx*Pyy))
                PhaseShift_Int['ING'][pi,ii] = Cxy_phs[np.argmin(abs(f-fpeak))]*(180./np.pi)
            else:
                PhaseShift_Int['ING'][pi,ii] = float('nan')
            
            if comp_phase_curr:
                # Pyr.:
                I_AMPA = I_AMPA_Pyr_list[idx]/(1e-9)
                I_AMPA = I_AMPA[int(start_time/sim_dt):int(end_time/sim_dt)]
                I_AMPA -= np.mean(I_AMPA)
                I_GABA = I_GABA_Pyr_list[idx]/(1e-9)
                I_GABA = I_GABA[int(start_time/sim_dt):int(end_time/sim_dt)]
                I_GABA -= np.mean(I_GABA)

                N = I_AMPA.shape[0]
                NFFT = 2**(N-1).bit_length()
                freq_vect = np.linspace(0, fmax, NFFT/2)*Hz
                freq_vect = freq_vect[np.where(freq_vect<=300)]
                a = pmtm(I_GABA, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                I_MTS = I_MTS[np.where(freq_vect<=300)]
                fpeak = freq_vect[np.argmax(I_MTS)]

                corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
                phases = np.arange(1-N, N)

                PhaseShift = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
                PhaseShiftCurr_Pyr[pi,ii] = np.sign(PhaseShift)*(abs(PhaseShift)%360)

                # Int.:
                I_AMPA = I_AMPA_Int_list[idx]/(1e-9)
                I_AMPA = I_AMPA[int(start_time/sim_dt):int(end_time/sim_dt)]
                I_AMPA -= np.mean(I_AMPA)
                I_GABA = I_GABA_Int_list[idx]/(1e-9)
                I_GABA = I_GABA[int(start_time/sim_dt):int(end_time/sim_dt)]
                I_GABA -= np.mean(I_GABA)

                N = I_AMPA.shape[0]
                NFFT = 2**(N-1).bit_length()
                freq_vect = np.linspace(0, fmax, NFFT/2)*Hz
                freq_vect = freq_vect[np.where(freq_vect/Hz<=300)]
                a = pmtm(I_GABA, NFFT=NFFT, NW=2.5, method='eigen', show=False)
                I_MTS = np.mean(abs(a[0])**2 * a[1], axis=0)[:int(NFFT/2)]/N
                I_MTS = I_MTS[np.where(freq_vect<=300)]
                fpeak = freq_vect[np.argmax(I_MTS)]

                corr_sig = np.correlate(I_AMPA, I_GABA, mode='full')
                phases = np.arange(1-N, N)

                PhaseShift = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
                PhaseShiftCurr_Int[pi,ii] = np.sign(PhaseShift)*(abs(PhaseShift)%360)

    if not (out_file is None):
        h5file = tables.open_file(out_file+'.hf5', mode='w', title='Analysis')
        root = h5file.root
        h5file.create_carray(root, "AvgCellRate_Pyr", obj=AvgCellRate_Pyr)
        h5file.create_carray(root, "SynchMeasure_Pyr", obj=SynchMeasure_Pyr)
        h5file.create_carray(root, "SynchFreqPING_Pyr", obj=SynchFreq_Pyr['PING'])
        h5file.create_carray(root, "SynchFreqING_Pyr", obj=SynchFreq_Pyr['ING'])
        h5file.create_carray(root, "SynchFreqPowPING_Pyr", obj=SynchFreqPow_Pyr['PING'])
        h5file.create_carray(root, "SynchFreqPowING_Pyr", obj=SynchFreqPow_Pyr['ING'])
        h5file.create_carray(root, "PhaseShiftPING_Pyr", obj=PhaseShift_Pyr['PING'])
        h5file.create_carray(root, "PhaseShiftING_Pyr", obj=PhaseShift_Pyr['ING'])
        h5file.create_carray(root, "AvgCellRate_Int", obj=AvgCellRate_Int)
        h5file.create_carray(root, "SynchMeasure_Int", obj=SynchMeasure_Int)
        h5file.create_carray(root, "SynchFreqPING_Int", obj=SynchFreq_Int['PING'])
        h5file.create_carray(root, "SynchFreqING_Int", obj=SynchFreq_Int['ING'])
        h5file.create_carray(root, "SynchFreqPowPING_Int", obj=SynchFreqPow_Int['PING'])
        h5file.create_carray(root, "SynchFreqPowING_Int", obj=SynchFreqPow_Int['ING'])
        h5file.create_carray(root, "PhaseShiftPING_Int", obj=PhaseShift_Int['PING'])
        h5file.create_carray(root, "PhaseShiftING_Int", obj=PhaseShift_Int['ING'])
        h5file.create_carray(root, "IterArray1", obj=IterArray1)
        h5file.create_carray(root, "IterArray2", obj=IterArray2)
        if comp_phase_curr:
            h5file.create_carray(root, "PhaseShiftCurr_Pyr", obj=PhaseShiftCurr_Pyr)
            h5file.create_carray(root, "PhaseShiftCurr_Int", obj=PhaseShiftCurr_Int)
        h5file.close()
        if verbose:
            print('Saved analysis results successfully!')
    
    Sims_feats = {'SynchFreq_Pyr':SynchFreq_Pyr,
                  'SynchFreqPow_Pyr':SynchFreqPow_Pyr,
                  'PhaseShift_Pyr':PhaseShift_Pyr,
                  'AvgCellRate_Pyr':AvgCellRate_Pyr,
                  'SynchMeasure_Pyr':SynchMeasure_Pyr,
                  'SynchFreq_Int':SynchFreq_Int,
                  'SynchFreqPow_Int':SynchFreqPow_Int,
                  'PhaseShift_Int':PhaseShift_Int,
                  'AvgCellRate_Int':AvgCellRate_Int,
                  'SynchMeasure_Int':SynchMeasure_Int}
    if comp_phase_curr:
        Sims_feats['PhaseShiftCurr_Pyr'] = PhaseShiftCurr_Pyr
        Sims_feats['PhaseShiftCurr_Int'] = PhaseShiftCurr_Int
            
    if PlotFlag:
        plot_results(IterArray1, IterArray2, mode, Sims_feats, plot_file)
   
    return Sims_feats
