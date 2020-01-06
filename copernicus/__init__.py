"""The copernicus python package"""

import math
import pathlib


from elephant.spike_train_generation import spike_extraction
import matplotlib.pyplot as plt
from neo.core import AnalogSignal
from neuronunit.models import StaticModel
from neuronunit import bbp  # Contains functions for dealing with .ibw files
import numpy as np
import pandas as pd
import quantities as pq
from scipy.io import matlab
import sciunit

HOME = pathlib.Path(__file__).parent.parent
DATA_ROOT = pathlib.Path(HOME, 'data')
DATA = DATA_ROOT  # To overwrite from notebook


def matfile_to_sweeps(name, layer=0):
    """Layer 0 is spontaneous activity"""
    path = DATA / name / ('%s.mat' % name)
    if not path.exists():
        path = DATA / ('%s.mat' % name)
    data = matlab.loadmat(path)
    sweeps_raw = data[name][0][0][layer]
    n_samples, n_sweeps = sweeps_raw.shape
    times = np.linspace(0, 2, n_samples, endpoint=False)
    sweeps = pd.DataFrame(sweeps_raw, index=times, columns=['Sweep %d'%i for i in range(1, n_sweeps+1)])
    sweeps.index.name = 'Time (s)'
    sweeps.title = name
    return sweeps


def protocol_to_sweeps(name, protocol, suffix='_0'):
    path = DATA / name / ('%s%s.txt' % (protocol, suffix))
    df = pd.read_csv(path, sep='\t', header=None)
    dt = 0.0001
    df.index = np.arange(0, df.shape[0]*dt, dt)
    df.index.name = 'Time (s)'
    n_sweeps = df.shape[1]
    df.columns = ['Sweep %d' % i for i in range(1, n_sweeps+1)]
    df.title = '%s: %s' % (name, protocol)
    return df


def sweeps_to_image(sweeps, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()
    else:
        fig = plt.gcf()
    image = ax.imshow(sweeps.T, aspect='auto', vmin=-75, vmax=-20)
    ax.grid(False)
    from matplotlib import cm
    fig.colorbar(image, ax=ax, label='Vm (mV)')
    ax.set_title(sweeps.title)
    ax.set_xticks(np.arange(0, 20000, 5000))
    ax.set_xticklabels(np.arange(0, 2, 0.5))
    ax.set_xlabel('Time (s)')
    n_sweeps = sweeps.shape[1]
    y_skip = 1 if n_sweeps<20 else int(n_sweeps/10)
    ax.set_yticks(range(0, n_sweeps, y_skip))
    ax.set_yticklabels(range(1, n_sweeps+1, y_skip))
    ax.set_ylabel('Sweep Number')
    ax.set_ylim(n_sweeps-0.5, -0.5)


def get_recordings():
    recordings = []
    for path in DATA.iterdir():
        if path.is_dir():
            recordings.append(path.name)
    return recordings


def make_summary(recordings, protocol=None, suffix='_0', max_sweeps=None):
    fig, axes = plt.subplots(math.ceil(len(recordings)/4), 4, sharex=True, figsize=(30,60))
    for i, recording in enumerate(recordings):
        ax = axes.flat[i]
        if protocol is None or protocol.lower()=='spontaneous':
            sweeps = matfile_to_sweeps(recording)
        else:
            sweeps = protocol_to_sweeps(recording, protocol.title(), suffix=suffix)
        sweeps_to_image(sweeps, ax=ax)
        if max_sweeps:
            ax.set_ylim(max_sweeps-0.5, -0.5)
    # Hide axes with no images
    while True:
        i += 1
        try:
            axes.flat[i].axis('off')
        except:
            break
    plt.tight_layout()
    protocol = protocol if protocol else 'spontaneous'
    plt.savefig('%s.pdf' % protocol.lower(), format='pdf')
    

def make_plot_summary(recordings, protocol=None, suffix='_0'):
    cols = 4
    fig, axes = plt.subplots(math.ceil(len(recordings)/cols), cols, sharex=True, sharey=True, figsize=(30,60))
    for i, recording in enumerate(recordings):
        ax = axes.flat[i]
        if protocol is None or protocol.lower()=='spontaneous':
            sweeps = matfile_to_sweeps(recording)
        else:
            sweeps = protocol_to_sweeps(recording, protocol.title(), suffix=suffix)
        sweeps.plot(legend=False, ax=ax, title=sweeps.title, ylim=(-80, 40)).set(ylabel='Vm (mV)');
    # Hide axes with no images
    while True:
        i += 1
        try:
            axes.flat[i].axis('off')
        except:
            break
    plt.tight_layout()
    protocol = protocol if protocol else 'spontaneous'
    plt.savefig('%s_plots.jpg' % protocol.lower(), format='jpg', dpi=150)
    

def sweep_to_neo(sweeps, n):
    sweep_name = 'Sweep %d' % n
    t = sweeps[sweep_name].index
    vm = AnalogSignal(sweeps[sweep_name], times=t.values, units=pq.mV, sampling_period=(t[1]-t[0])*pq.s)
    return vm


def count_stray_spikes(vm, stim_start, stim_end):
    spike_train = spike_extraction(vm, threshold=-20*pq.mV)
    before_stim_train = spike_train.time_slice(0*pq.s, stim_start*pq.s-1*pq.ms)
    after_stim_train = spike_train.time_slice(stim_end*pq.s+5*pq.ms, None)
    n_before = len(before_stim_train)
    n_after = len(after_stim_train)
    return n_before + n_after


def get_holding_potential(vm, stim_start):
    return vm.time_slice(0*pq.s, stim_start*pq.s).mean()


def get_spike_count_by_threshold(vm, thresholds=np.arange(-30, 15, 10)*pq.mV):
    counts = {float(thresh): len(spike_extraction(vm, threshold=thresh)) for thresh in thresholds}
    return counts


def spike_height_health(vm):
    counts = np.array(list(get_spike_count_by_threshold(vm).values()))
    return np.exp(-counts.std() / counts.mean())


def plot_spike_height_health(sweeps):
    fig, axes = plt.subplots(20, 2, figsize=(7, 45))
    for i in range(20):
        vm = sweep_to_neo(sweeps, i+1)
        axes[i, 0].plot(vm.times, vm)
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Vm (mV)')
        axes[i, 0].set_title("Sweep %d" % (i+1))
        axes[i, 1].scatter(*zip(*get_spike_count_by_threshold(vm).items()))
        axes[i, 1].set_xlabel('Threshold (mV)')
        axes[i, 1].set_ylabel('Spike Count')
        axes[i, 1].set_title('Spike Height Health = %.2g' % spike_height_health(vm))
    plt.tight_layout()
    
    
class IBWModel(StaticModel):
    pass

    
#### Plotly support (experimental) ####
# Requires:
# `import plotly.graph_objects as go`

def go_scatter(sweeps, i):
    return go.Scatter(y=sweeps['Sweep %d' % i],
                      x=sweeps.index,
                      mode='lines',
                      name='Sweep %d' % i)


def plot_sweeps(sweeps, sweep_list=None):
    if sweep_list is None:
        n_sweeps = sweeps.shape[1]
        sweep_list = range(1, n_sweeps+1)
    data = [go_scatter(sweeps, i) for i in sweep_list]
    fig = go.Figure(data=data, layout={'xaxis': {'title': 'Time (s)'},
                                       'yaxis': {'title': 'Vm (mV)'},
                                       'title': sweeps.title})
    fig.show()

####

