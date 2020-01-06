"""The copernicus python package"""

import pathlib

import pandas as pd
from scipy.io import matlab

HOME = pathlib.Path(__file__).parent.parent
DATA = pathlib.Path(HOME, 'ExportedDorsoventralData')

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