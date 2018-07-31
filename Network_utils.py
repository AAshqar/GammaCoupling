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
from NeuronsSpecs.NeuronEqs import *

#####################################################################################


def run_network(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInp=1, IntInp=1, PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, sim_dt=0.02, monitored=[], monitor_pois=False, record_vm=False, save_raw=False, filename=None, verbose=True, PlotFlag=False):
    '''
    Simulates a network consisting of the desired number of pyramidal neurons and interneurons with dynamics described by predefined set of equations.
    
    * Parameters:
    - N_p: Number of excitatory pyramidal neurons in the network
    - N_i: Number of inhibitory interneurons in the network
    - PyrEqs: Equations to use for the pyramidal population
    - IntEqs: Equations to use for the interneurons population
    - PreEqAMPA: Equations to use for AMPA (excitatory) synapses
    - PreEqGABA: Equations to use for GABA (inhibitory) synapses
    - PyrInp: Poisson input rate to the pyramidal neurons [kHz]
    - IntInp: Poisson input rate to the interneurons [kHz]
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
    
    monitors = [SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int]
    if not monitored_Pyr==[]:
        StateM_Pyr = StateMonitor(Pyr_pop, monitored_Pyr, record=True)
        StateM_Int = StateMonitor(Int_pop, monitored_Int, record=True)
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
                  SynIP, SynPI, SynII, SynPoiss_AMPA_Pyr,
                  SynPoiss_AMPA_Int, monitors)
    if verbose:
        print('Running the network...')

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
        rawfile.create_array(root, 'PyrInp', obj=PyrInp)
        rawfile.create_array(root, 'IntInp', obj=IntInp)
        rawfile.create_carray(root, 'SpikeM_t_Pyr_raw', obj=np.array(SpikeM_Pyr.t/ms)*ms)
        rawfile.create_carray(root, 'SpikeM_i_Pyr_raw', obj=np.array(SpikeM_Pyr.i))
        rawfile.create_carray(root, 'SpikeM_t_Int_raw', obj=np.array(SpikeM_Int.t/ms)*ms)
        rawfile.create_carray(root, 'SpikeM_i_Int_raw', obj=np.array(SpikeM_Int.i))
        rawfile.create_carray(root, 'PopRateSig_Pyr_raw', obj=PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
        rawfile.create_carray(root, 'PopRateSig_Int_raw', obj=PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
        if monitor_pois:
            rawfile.create_carray(root, 'SpikeM_t_PoisPyr_raw', obj=np.array(SpikeM_PoisPyr.t/ms)*ms)
            rawfile.create_carray(root, 'SpikeM_i_PoisPyr_raw', obj=np.array(SpikeM_PoisPyr.i))
            rawfile.create_carray(root, 'SpikeM_t_PoisInt_raw', obj=np.array(SpikeM_PoisInt.t/ms)*ms)
            rawfile.create_carray(root, 'SpikeM_i_PoisInt_raw', obj=np.array(SpikeM_PoisInt.i))
            rawfile.create_carray(root, 'PopRateSig_PoisPyr_raw', obj=PopRateM_PoisPyr.smooth_rate(window='gaussian', width=1*ms))
            rawfile.create_carray(root, 'PopRateSig_PoisInt_raw', obj=PopRateM_PoisInt.smooth_rate(window='gaussian', width=1*ms))

        if not monitored==[]:
            for i,var in enumerate(monitored):
                rawfile.create_carray(root, var+'_Pyr', obj=np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                rawfile.create_carray(root, var+'_Int', obj=np.array(StateM_Int.get_states()[var]).mean(axis=1))
    
        if record_vm:
            rawfile.create_vlarray(root, 'Vm_Pyr', obj=StateM_Pyr.get_states()['v_s'][:,[0,-1]])
            rawfile.create_vlarray(root, 'Vm_Int', obj=StateM_Pyr.get_states()['v'][:,[0,-1]])
        
        rawfile.close()
        
        if verbose:
            print('Saved raw data successfullty!')
          
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
    
    Monitors = {'SpikeM_Pyr':SpikeM_Pyr,
                'PopRateM_Pyr':PopRateM_Pyr,
                'SpikeM_Int':SpikeM_Int,
                'PopRateM_Int':PopRateM_Int}
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


def run_network_IP(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInp=1, IntInp=1, IPois_A=1., IPois_Atype='ramp', IPois_f=70, PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, sim_dt=0.02, monitored=[], monitor_pois=False, record_vm=False, save_raw=False, filename=None, verbose=True, PlotFlag=False):
    '''
    Simulates a network consisting of the desired number of pyramidal neurons and interneurons with dynamics described by predefined set of equations. Inputs are inhomogenous poisson processes (AC rates)
    
    * Parameters:
    - N_p: Number of excitatory pyramidal neurons in the network
    - N_i: Number of inhibitory interneurons in the network
    - PyrEqs: Equations to use for the pyramidal population
    - IntEqs: Equations to use for the interneurons population
    - PreEqAMPA: Equations to use for AMPA (excitatory) synapses
    - PreEqGABA: Equations to use for GABA (inhibitory) synapses
    - PyrInp: Poisson input rate to the pyramidal neurons [kHz]
    - IntInp: Poisson input rate to the interneurons [kHz]
    - IPois_A: Inhomogenous poisson's amplitude of variation [kHz]
    - IPois_Atype: Type of the inhomogenous poisson's amplitude of variation ('ramp' or 'const')
    - IPois_f: Inhomogenous poisson's frequency [Hz]
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
    
    monitors = [SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int]
    if not monitored_Pyr==[]:
        StateM_Pyr = StateMonitor(Pyr_pop, monitored_Pyr, record=True)
        StateM_Int = StateMonitor(Int_pop, monitored_Int, record=True)
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
                  SynIP, SynPI, SynII, SynPoiss_AMPA_Pyr,
                  SynPoiss_AMPA_Int, monitors)
    
    if verbose:
        print('Running the network...')

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
        rawfile.create_array(root, 'PyrInp', obj=PyrInp)
        rawfile.create_array(root, 'IntInp', obj=IntInp)
        rawfile.create_carray(root, 'SpikeM_t_Pyr_raw', obj=np.array(SpikeM_Pyr.t/ms)*ms)
        rawfile.create_carray(root, 'SpikeM_i_Pyr_raw', obj=np.array(SpikeM_Pyr.i))
        rawfile.create_carray(root, 'SpikeM_t_Int_raw', obj=np.array(SpikeM_Int.t/ms)*ms)
        rawfile.create_carray(root, 'SpikeM_i_Int_raw', obj=np.array(SpikeM_Int.i))
        rawfile.create_carray(root, 'PopRateSig_Pyr_raw', obj=PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
        rawfile.create_carray(root, 'PopRateSig_Int_raw', obj=PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
        if monitor_pois:
            rawfile.create_carray(root, 'SpikeM_t_PoisPyr_raw', obj=np.array(SpikeM_PoisPyr.t/ms)*ms)
            rawfile.create_carray(root, 'SpikeM_i_PoisPyr_raw', obj=np.array(SpikeM_PoisPyr.i))
            rawfile.create_carray(root, 'SpikeM_t_PoisInt_raw', obj=np.array(SpikeM_PoisInt.t/ms)*ms)
            rawfile.create_carray(root, 'SpikeM_i_PoisInt_raw', obj=np.array(SpikeM_PoisInt.i))
            rawfile.create_carray(root, 'PopRateSig_PoisPyr_raw', obj=PopRateM_PoisPyr.smooth_rate(window='gaussian', width=1*ms))
            rawfile.create_carray(root, 'PopRateSig_PoisInt_raw', obj=PopRateM_PoisInt.smooth_rate(window='gaussian', width=1*ms))

        if not monitored==[]:
            for i,var in enumerate(monitored):
                rawfile.create_carray(root, var+'_Pyr', obj=np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                rawfile.create_carray(root, var+'_Int', obj=np.array(StateM_Int.get_states()[var]).mean(axis=1))
    
        if record_vm:
            rawfile.create_vlarray(root, 'Vm_Pyr', obj=StateM_Pyr.get_states()['v_s'][:,[0,-1]])
            rawfile.create_vlarray(root, 'Vm_Int', obj=StateM_Pyr.get_states()['v'][:,[0,-1]])
        
        rawfile.close()
        
        if verbose:
            print('Saved raw data successfullty!')
        
    if PlotFlag:
        figure()
        subplot(2,1,1)
        plot(SpikeM_Pyr.t/ms, SpikeM_Pyr.i, '.', SpikeM_Int.t/ms, SpikeM_Int.i+4000, '.')
        xlim(PopRateM_Pyr.t[0]/ms, PopRateM_Pyr.t[-1]/ms)
        subplot(2,1,2)
        plot(PopRateM_Pyr.t/ms, PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms), PopRateM_Int.t/ms, PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
        xlabel('Time (ms)')
        xlim(PopRateM_Pyr.t[0]/ms, PopRateM_Pyr.t[-1]/ms)
        ylabel('Neuron Index')
        show()
    
    Monitors = {'SpikeM_Pyr':SpikeM_Pyr,
                'PopRateM_Pyr':PopRateM_Pyr,
                'SpikeM_Int':SpikeM_Int,
                'PopRateM_Int':PopRateM_Int}
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


def analyze_network(Monitors, comp_phase=False, N_p=4000, N_i=1000, start_time=None, end_time=None, sim_dt=0.02, mts_win='whole', W=2**12, ws=None, PlotFlag=False):
    '''
    Analyzes a pre-simulated network and extracts various features using the provided monitors
    
    * Parameters:
    - Monitors: A dictionary of the monitors used to record raw data during the simulation
    - comp_phase: Whether to include phase shift calculations between of AMPA and GABA currents
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

        PhaseShift_Pyr = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
        PhaseShift_Pyr = np.sign(PhaseShift_Pyr)*(PhaseShift_Pyr%360)
        
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

        PhaseShift_Int = (phases[np.argmax(corr_sig)]*(sim_dt)*fpeak*360)
        PhaseShift_Int = np.sign(PhaseShift_Int)*(PhaseShift_Int%360)
        
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
    if comp_phase:
        Network_feats['PhaseShift_Pyr'] = PhaseShift_Pyr
        Network_feats['PhaseShift_Int'] = PhaseShift_Int
            
    return Network_feats
    
#####################################################################################


def PopRateM_mtspectrogram(Monitors, W=2**12, ws=None, start_time=None, end_time=None, sim_dt=0.02, PlotFlag=True):
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
    
    N_segs = int((len(RateSig_Pyr) / ws)-(W-ws)/ws +1)
    result = np.zeros((NFFT/2, N_segs))
    for i in range(N_segs):
        data = RateSig_Pyr[i*ws:i*ws+W]
        a = pmtm(data, NFFT=NFFT, NW=2.5, method='eigen', show=False)
        Sks = np.mean(abs(a[0])**2 * a[1], axis=0)
        result[:,i] = Sks[:NFFT/2]/W
    RateMTS_Pyr = np.squeeze(result[np.where(freq_vect/Hz<=300),:])
        
    N_segs = int((len(RateSig_Int) / ws)-(W-ws)/ws +1)
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
    
    Rate_MTS = {'RateMTS_Pyr':RateMTS_Pyr, 'RateMTS_Int':RateMTS_Int}
    
    return Rate_MTS
    
#####################################################################################


def run_multsim(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInps=[0.5,1], IntInps=[0.5,1], PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, start_time=None, end_time=None, sim_dt=0.02, monitored=[], monitor_pois=False, mon_avg=True, comp_phase=False, record_vm=True, mts_win='whole', W=2**12, ws=None, verbose=True, analyze=True, save_analyzed=False, save_raw=False, filename=None):
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
        if monitor_pois:
            SpikeM_t_PoisPyr_raw = rawfile.create_vlarray(root, 'SpikeM_t_PoisPyr_raw', tables.Float64Atom(shape=()))
            SpikeM_i_PoisPyr_raw = rawfile.create_vlarray(root, 'SpikeM_i_PoisPyr_raw', tables.Float64Atom(shape=()))
            SpikeM_t_PoisInt_raw = rawfile.create_vlarray(root, 'SpikeM_t_PoisInt_raw', tables.Float64Atom(shape=()))
            SpikeM_i_PoisInt_raw = rawfile.create_vlarray(root, 'SpikeM_i_PoisInt_raw', tables.Float64Atom(shape=()))
            PopRateSig_PoisPyr_raw = rawfile.create_vlarray(root, 'PopRateSig_PoisPyr_raw', tables.Float64Atom(shape=()))
            PopRateSig_PoisInt_raw = rawfile.create_vlarray(root, 'PopRateSig_PoisInt_raw', tables.Float64Atom(shape=()))
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
            
            Monitors = run_network(N_p=N_p, N_i=N_i, PyrEqs=PyrEqs, IntEqs=IntEqs, PreEqAMPA=PreEqAMPA, PreEqGABA=PreEqGABA, PyrInp=PyrInp, IntInp=IntInp, PP_C=PP_C, IP_C=IP_C, II_C=II_C, PI_C=PI_C, runtime=runtime, sim_dt=sim_dt, monitored=monitored, monitor_pois=monitor_pois, record_vm=record_vm, verbose=verbose, PlotFlag=False)
            
            if analyze:
                Network_feats = analyze_network(Monitors, comp_phase=comp_phase, N_p=N_p, N_i=N_i, start_time=start_time, end_time=end_time, sim_dt=sim_dt, mts_win=mts_win, W=W, ws=ws)
                
                AvgCellRate_Pyr[pi,ii], SynchFreq_Pyr[pi,ii], SynchFreqPow_Pyr[pi,ii], PkWidth_Pyr[pi,ii], Harmonics_Pyr[pi,ii], SynchMeasure_Pyr[pi,ii], AvgCellRate_Int[pi,ii], SynchFreq_Int[pi,ii], SynchFreqPow_Int[pi,ii], PkWidth_Int[pi,ii], Harmonics_Int[pi,ii], SynchMeasure_Int[pi,ii] = Network_feats['AvgCellRate_Pyr'], Network_feats['SynchFreq_Pyr'], Network_feats['SynchFreqPow_Pyr'], Network_feats['PkWidth_Pyr'], Network_feats['Harmonics_Pyr'], Network_feats['SynchMeasure_Pyr'], Network_feats['AvgCellRate_Int'], Network_feats['SynchFreq_Int'], Network_feats['SynchFreqPow_Int'], Network_feats['PkWidth_Int'], Network_feats['Harmonics_Int'], Network_feats['SynchMeasure_Int']
            
                if comp_phase:
                    PhaseShift_Pyr[pi,ii], PhaseShift_Int[pi,ii] = Network_feats['PhaseShift_Pyr'], Network_feats['PhaseShift_Int']
                
            if save_raw:
                SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int = Monitors['SpikeM_Pyr'], Monitors['PopRateM_Pyr'], Monitors['SpikeM_Int'], Monitors['PopRateM_Int']

                rawfile = tables.open_file(filename+'_raw.h5', mode='a')
                Params = rawfile.root.Params
                Params.append(str((PyrInp, IntInp)))
                SpikeM_t_Pyr_raw = rawfile.root.SpikeM_t_Pyr_raw
                SpikeM_t_Pyr_raw.append(np.array(SpikeM_Pyr.t/ms)*ms)
                SpikeM_i_Pyr_raw = rawfile.root.SpikeM_i_Pyr_raw
                SpikeM_i_Pyr_raw.append(np.array(SpikeM_Pyr.i))
                SpikeM_t_Int_raw = rawfile.rawfile.root.SpikeM_t_Int_raw
                SpikeM_t_Int_raw.append(np.array(SpikeM_Int.t/ms)*ms)
                SpikeM_i_Int_raw = rawfile.rawfile.root.SpikeM_i_Int_raw
                SpikeM_i_Int_raw.append(np.array(SpikeM_Int.i))
                
                PopRateSig_Pyr_raw = rawfile.rawfile.root.PopRateSig_Pyr_raw
                PopRateSig_Pyr_raw.append(PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
                PopRateSig_Int_raw = rawfile.rawfile.root.PopRateSig_Int_raw
                PopRateSig_Int_raw.append(PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
                if not monitored==[]:
                    StateM_Pyr, StateM_Int = Monitors['StateM_Pyr'], Monitors['StateM_Int']
                    for i,var in enumerate(monitored): 
                        if mon_avg:
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                            rawfile.get_node('/', name=var+'_Int').append(np.array(StateM_Int.get_states()[var]).mean(axis=1))
                        else:
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var]))
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Int.get_states()[var]))
                if monitor_pois:
                    SpikeM_PoisPyr, PopRateM_PoisPyr, SpikeM_PoisInt, PopRateM_PoisInt = Monitors['SpikeM_PoisPyr'], Monitors['PopRateM_PoisPyr'], Monitors['SpikeM_PoisInt'], Monitors['PopRateM_PoisInt']
                    SpikeM_t_PoisPyr_raw = rawfile.root.SpikeM_t_PoisPyr_raw
                    SpikeM_t_PoisPyr_raw.append(np.array(SpikeM_PoisPyr.t/ms)*ms)
                    SpikeM_i_PoisPyr_raw = rawfile.root.SpikeM_i_PoisPyr_raw
                    SpikeM_i_PoisPyr_raw.append(np.array(SpikeM_PoisPyr.i))
                    SpikeM_t_PoisInt_raw = rawfile.root.SpikeM_t_PoisInt_raw
                    SpikeM_t_PoisInt_raw.append(np.array(SpikeM_PoisInt.t/ms)*ms)
                    SpikeM_i_PoisInt_raw = rawfile.root.SpikeM_i_PoisInt_raw
                    SpikeM_i_PoisInt_raw.append(np.array(SpikeM_PoisInt.i))
                    PopRateSig_PoisPyr_raw = rawfile.root.PopRateSig_PoisPyr_raw
                    PRPois_Pyr = PopRateM_PoisPyr.smooth_rate(window='gaussian', width=1*ms)
                    PopRateSig_PoisPyr_raw.append(PRPois_Pyr)
                    PopRateSig_PoisInt_raw = rawfile.root.PopRateSig_PoisInt_raw
                    PRPois_Int = PopRateM_PoisInt.smooth_rate(window='gaussian', width=1*ms)
                    PopRateSig_PoisInt_raw.append(PRPois_Int)
                if record_vm:
                    Vm_Pyr = rawfile.root.Vm_Pyr
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,0])
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,-1])
                    Vm_Int = rawfile.root.Vm_Int
                    Vm_Int.append(StateM_Int.get_states()['v'][:,0])
                    Vm_Int.append(StateM_Int.get_states()['v'][:,-1])
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
        if comp_phase:
            Sims_feats['PhaseShift_Pyr'] = PhaseShift_Pyr
            Sims_feats['PhaseShift_Int'] = PhaseShift_Int
            
        return Sims_feats
    
#####################################################################################


def run_multsim_IP(N_p=4000, N_i=1000, PyrEqs=eqs_P, IntEqs=eqs_I, PreEqAMPA=PreEq_AMPA, PreEqGABA=PreEq_GABA, PyrInp=1, IntInp=1, IPois_As=[1.], IPois_Atype='ramp', IPois_fs=[70], PP_C=0.01, IP_C=0.1, II_C=0.1, PI_C=0.1, runtime=1000, start_time=None, end_time=None, sim_dt=0.02, monitored=[], monitor_pois=False, mon_avg=True, record_vm=True, mts_win='whole', W=2**12, ws=None, verbose=True, analyze=True, save_analyzed=False, save_raw=False, filename=None):
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
    
    if ws is None:
        ws = W/10
    
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
        if monitor_pois:
            SpikeM_t_PoisPyr_raw = rawfile.create_vlarray(root, 'SpikeM_t_PoisPyr_raw', tables.Float64Atom(shape=()))
            SpikeM_i_PoisPyr_raw = rawfile.create_vlarray(root, 'SpikeM_i_PoisPyr_raw', tables.Float64Atom(shape=()))
            SpikeM_t_PoisInt_raw = rawfile.create_vlarray(root, 'SpikeM_t_PoisInt_raw', tables.Float64Atom(shape=()))
            SpikeM_i_PoisInt_raw = rawfile.create_vlarray(root, 'SpikeM_i_PoisInt_raw', tables.Float64Atom(shape=()))
            PopRateSig_PoisPyr_raw = rawfile.create_vlarray(root, 'PopRateSig_PoisPyr_raw', tables.Float64Atom(shape=()))
            PopRateSig_PoisInt_raw = rawfile.create_vlarray(root, 'PopRateSig_PoisInt_raw', tables.Float64Atom(shape=()))
    
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
            
            Monitors = run_network_IP(N_p=N_p, N_i=N_i, PyrEqs=PyrEqs, IntEqs=IntEqs, PreEqAMPA=PreEqAMPA, PreEqGABA=PreEqGABA, PyrInp=PyrInp, IntInp=IntInp, IPois_A=IP_A, IPois_Atype=IPois_Atype, IPois_f=IP_f, PP_C=PP_C, IP_C=IP_C, II_C=II_C, PI_C=PI_C, runtime=runtime, sim_dt=sim_dt, monitored=monitored, monitor_pois=monitor_pois, record_vm=record_vm, verbose=verbose, PlotFlag=False)
            
            if analyze:
                Network_feats = analyze_network(Monitors, comp_phase=comp_phase, N_p=N_p, N_i=N_i, start_time=start_time, end_time=end_time, sim_dt=sim_dt, mts_win=mts_win, W=W, ws=ws)
                
                AvgCellRate_Pyr[pi,ii], SynchFreq_Pyr[pi,ii], SynchFreqPow_Pyr[pi,ii], PkWidth_Pyr[pi,ii], Harmonics_Pyr[pi,ii], SynchMeasure_Pyr[pi,ii], AvgCellRate_Int[pi,ii], SynchFreq_Int[pi,ii], SynchFreqPow_Int[pi,ii], PkWidth_Int[pi,ii], Harmonics_Int[pi,ii], SynchMeasure_Int[pi,ii] = Network_feats['AvgCellRate_Pyr'], Network_feats['SynchFreq_Pyr'], Network_feats['SynchFreqPow_Pyr'], Network_feats['PkWidth_Pyr'], Network_feats['Harmonics_Pyr'], Network_feats['SynchMeasure_Pyr'], Network_feats['AvgCellRate_Int'], Network_feats['SynchFreq_Int'], Network_feats['SynchFreqPow_Int'], Network_feats['PkWidth_Int'], Network_feats['Harmonics_Int'], Network_feats['SynchMeasure_Int']
            
                if comp_phase:
                    PhaseShift_Pyr[pi,ii], PhaseShift_Int[pi,ii] = Network_feats['PhaseShift_Pyr'], Network_feats['PhaseShift_Int']
                
            if save_raw:
                SpikeM_Pyr, PopRateM_Pyr, SpikeM_Int, PopRateM_Int = Monitors['SpikeM_Pyr'], Monitors['PopRateM_Pyr'], Monitors['SpikeM_Int'], Monitors['PopRateM_Int']

                rawfile = tables.open_file(filename+'_raw.h5', mode='a')
                Params = rawfile.root.Params
                Params.append(str((PyrInp, IntInp)))
                SpikeM_t_Pyr_raw = rawfile.root.SpikeM_t_Pyr_raw
                SpikeM_t_Pyr_raw.append(np.array(SpikeM_Pyr.t/ms)*ms)
                SpikeM_i_Pyr_raw = rawfile.root.SpikeM_i_Pyr_raw
                SpikeM_i_Pyr_raw.append(np.array(SpikeM_Pyr.i))
                SpikeM_t_Int_raw = rawfile.rawfile.root.SpikeM_t_Int_raw
                SpikeM_t_Int_raw.append(np.array(SpikeM_Int.t/ms)*ms)
                SpikeM_i_Int_raw = rawfile.rawfile.root.SpikeM_i_Int_raw
                SpikeM_i_Int_raw.append(np.array(SpikeM_Int.i))
                
                PopRateSig_Pyr_raw = rawfile.rawfile.root.PopRateSig_Pyr_raw
                PopRateSig_Pyr_raw.append(PopRateM_Pyr.smooth_rate(window='gaussian', width=1*ms))
                PopRateSig_Int_raw = rawfile.rawfile.root.PopRateSig_Int_raw
                PopRateSig_Int_raw.append(PopRateM_Int.smooth_rate(window='gaussian', width=1*ms))
                if not monitored==[]:
                    StateM_Pyr, StateM_Int = Monitors['StateM_Pyr'], Monitors['StateM_Int']
                    for i,var in enumerate(monitored): 
                        if mon_avg:
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var]).mean(axis=1))
                            rawfile.get_node('/', name=var+'_Int').append(np.array(StateM_Int.get_states()[var]).mean(axis=1))
                        else:
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Pyr.get_states()[var]))
                            rawfile.get_node('/', name=var+'_Pyr').append(np.array(StateM_Int.get_states()[var]))
                if monitor_pois:
                    SpikeM_PoisPyr, PopRateM_PoisPyr, SpikeM_PoisInt, PopRateM_PoisInt = Monitors['SpikeM_PoisPyr'], Monitors['PopRateM_PoisPyr'], Monitors['SpikeM_PoisInt'], Monitors['PopRateM_PoisInt']
                    SpikeM_t_PoisPyr_raw = rawfile.root.SpikeM_t_PoisPyr_raw
                    SpikeM_t_PoisPyr_raw.append(np.array(SpikeM_PoisPyr.t/ms)*ms)
                    SpikeM_i_PoisPyr_raw = rawfile.root.SpikeM_i_PoisPyr_raw
                    SpikeM_i_PoisPyr_raw.append(np.array(SpikeM_PoisPyr.i))
                    SpikeM_t_PoisInt_raw = rawfile.root.SpikeM_t_PoisInt_raw
                    SpikeM_t_PoisInt_raw.append(np.array(SpikeM_PoisInt.t/ms)*ms)
                    SpikeM_i_PoisInt_raw = rawfile.root.SpikeM_i_PoisInt_raw
                    SpikeM_i_PoisInt_raw.append(np.array(SpikeM_PoisInt.i))
                    PopRateSig_PoisPyr_raw = rawfile.root.PopRateSig_PoisPyr_raw
                    PRPois_Pyr = PopRateM_PoisPyr.smooth_rate(window='gaussian', width=1*ms)
                    PopRateSig_PoisPyr_raw.append(PRPois_Pyr)
                    PopRateSig_PoisInt_raw = rawfile.root.PopRateSig_PoisInt_raw
                    PRPois_Int = PopRateM_PoisInt.smooth_rate(window='gaussian', width=1*ms)
                    PopRateSig_PoisInt_raw.append(PRPois_Int)
                if record_vm:
                    Vm_Pyr = rawfile.root.Vm_Pyr
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,0])
                    Vm_Pyr.append(StateM_Pyr.get_states()['v_s'][:,-1])
                    Vm_Int = rawfile.root.Vm_Int
                    Vm_Int.append(StateM_Int.get_states()['v'][:,0])
                    Vm_Int.append(StateM_Int.get_states()['v'][:,-1])
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
        if comp_phase:
            Sims_feats['PhaseShift_Pyr'] = PhaseShift_Pyr
            Sims_feats['PhaseShift_Int'] = PhaseShift_Int
            
        return Sims_feats
    
#####################################################################################


def analyze_raw(filename, mode, N_p=4000, N_i=1000, start_time=None, end_time=None, sim_dt=0.02, comp_phase=False, mts_win='whole', W=2**12, ws=None, verbose=False, PlotFlag=False, plot_file=None, out_file=None):
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
    - comp_phase: Whether to include phase shift calculations between of AMPA and GABA currents
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
    
    PopRateSig_Pyr_list = rawfile.root.PopRateSig_Pyr_raw.read()
    PopRateSig_Int_list = rawfile.root.PopRateSig_Int_raw.read()
    
    Spike_t_Pyr_list = rawfile.root.SpikeM_t_Pyr_raw.read()
    Spike_i_Pyr_list = rawfile.root.SpikeM_i_Pyr_raw.read()
    Spike_t_Int_list = rawfile.root.SpikeM_t_Int_raw.read()
    Spike_i_Int_list = rawfile.root.SpikeM_i_Int_raw.read()
    
    if comp_phase:
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
            
            if verbose:
                print('Analyzing network %d/%d...' %(pi*len(IterArray2)+ii+1, len(IterArray1)*len(IterArray2)))
            
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
                PhaseShift_Pyr[pi,ii] = np.sign(PhaseShift)*(PhaseShift%360)

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
                PhaseShift_Int[pi,ii] = np.sign(PhaseShift)*(PhaseShift%360)

    if not (out_file is None):
        with tables.open_file(out_file+'.hf5', mode='w', title='Analysis') as h5file:
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
            h5file.create_carray(root, "IterArray1", obj=IterArray1)
            h5file.create_carray(root, "IterArray2", obj=IterArray2)
            if comp_phase:
                h5file.create_carray(root, "PhaseShift_Pyr", obj=PhaseShift_Pyr)
                h5file.create_carray(root, "PhaseShift_Int", obj=PhaseShift_Int)

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
                  'SynchMeasure_Int':SynchMeasure_Int}
    if comp_phase:
        Sims_feats['PhaseShift_Pyr'] = PhaseShift_Pyr
        Sims_feats['PhaseShift_Int'] = PhaseShift_Int
            
    if PlotFlag:
        plot_results(IterArray1, IterArray2, mode, Sims_feats, plot_file)
   
    return Sims_feats
