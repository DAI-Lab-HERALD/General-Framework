import numpy as np
import os

split_file = ('.' + os.sep + 'Framework' + os.sep + 'Results' + os.sep + 
              'Forking Paths (interactive pedestrians - augmented)' + os.sep + 'Splitting' + os.sep + 
              'Fork_P_Aug--t0=start_ss--dt=0.40_nI=08m08_nO=12m12_EC--max_000_agents_0--crossv_split_0.npy')

split = np.load(split_file, allow_pickle=True)[1][25]

data_file = ('.' + os.sep + 'Framework' + os.sep + 'Results' + os.sep +
             'Forking Paths (interactive pedestrians - augmented)' + os.sep + 'Data' + os.sep +
             'Fork_P_Aug--t0=start_ss--dt=0.40_nI=08m08_nO=12m12_EC--max_000_agents_0.npy')

data = np.load(data_file, allow_pickle=True)

Domain = data[-3]

domain = Domain.iloc[split]

print(domain)