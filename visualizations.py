# %% Imports
import numpy as np
import networkx as nx
import matplotlib as mpl
from matplotlib import pyplot as plt

# %% Load data and visualize as graph

results = np.load('results/sbm_test.npz') 

adj_mat = results['arr_0']
belief_hist = results['arr_1']

believers = np.where(belief_hist[-1,1,:] > 0.5)[0]
nonbelievers = np.where(belief_hist[-1,1,:] < 0.5)[0]

G = nx.from_numpy_array(adj_mat)

color_lookup = {node_id:1 if node_id in believers else 0 for node_id in range(belief_hist.shape[2])}

low, *_, high = sorted(color_lookup.values())
norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

fig, ax = plt.subplots(figsize=(16,12))
nx.draw(G, node_size = 8000, linewidths=5, node_color=[mapper.to_rgba(i) for i in color_lookup.values()], ax = ax)

# %%
fig, ax = plt.subplots(figsize=(16,12))
for non_believe_idx in nonbelievers:
    if non_believe_idx == nonbelievers[-1]:
       plt.plot(belief_hist[:,0,non_believe_idx],c='b',lw=2.5, label='Believe in Idea 1')
    else:
        plt.plot(belief_hist[:,0,non_believe_idx],c='b',lw=2.5)

for believe_idx in believers:
    if believe_idx == believers[-1]:
       plt.plot(belief_hist[:,0,believe_idx],c='r',lw=2.5, label='Believe in Idea 2')
    else:
        plt.plot(belief_hist[:,0,believe_idx],c='r',lw=2.5)
plt.legend(fontsize=29)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Time',fontsize=26)
plt.ylabel('Strength of belief',fontsize=26)
# %%
