# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 18:55:57 2025

@author: balazs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ret = np.load('simulation_1d_acoustic_resonator_04_06_2025_14_59_45.npy', allow_pickle=True).item()
fs, f_SS, xs_FS, ys_FS, x_SS, y_SS, xxs, xys, yxs, yys = ret['resonators']

SS_filt = (f_SS > 4000)*(f_SS < 4400)
f_SS = f_SS[SS_filt]

xxs, xys, yxs, yys = map(np.transpose, map(lambda x: x[:, SS_filt],map(np.array, [xxs, xys, yxs, yys])))

x_SS, x_FS, y_SS, y_FS = map(np.array, [x_SS, xs_FS, y_SS, ys_FS])
x_SS = x_SS[SS_filt].reshape(len(x_SS[SS_filt]), 1) 
y_SS = y_SS[SS_filt].reshape(len(y_SS[SS_filt]), 1)
x_SS, x_FS, y_SS, y_FS = map(lambda x: np.broadcast_to(x, xxs.shape), [x_SS, x_FS, y_SS, y_FS])


# Set figure size (A4 with margins)
fig_width, fig_height = 5.71, 9.72
fig = plt.figure(figsize=(fig_width, fig_height), dpi=100)

# Create GridSpec with space for colorbars
gs = gridspec.GridSpec(4, 4, width_ratios=[1, 1, 1, 0.1], wspace=0.5, hspace=0.2)

# Row and column annotations
row_labels = ["oscillation of x", "oscillation of y", "FS spectrum", "SS spectrum"]
col_labels = ["x", "y", "r"]

for i, (x, y) in enumerate(zip([xxs, yxs, x_FS, x_SS], [xys, yys, y_FS, y_SS])):
    # x, y = x.T, y.T
    r = np.sqrt(x**2 + y**2)

    vmin, vmax = min(d.min() for d in [x, y, r]), max(d.max() for d in [x, y, r])

    for j, v in enumerate([x, y, r]):
        ax = fig.add_subplot(gs[i, j])
        im = ax.imshow(v, 
                       origin = 'lower', 
                       aspect = 'auto', 
                       extent = [fs[0]/1000, fs[-1]/1000, f_SS[0], f_SS[-1]],
                       vmin=vmin, vmax=vmax)
        if j:
            ax.set_yticks([])
    cax = fig.add_subplot(gs[i, 3])
    fig.colorbar(im, cax=cax, orientation='vertical')

    fig.text(0.04, 0.8 - i * 0.2, row_labels[i], fontsize=10, va='center', ha='right', rotation=90)

# Add column labels at the top
for j in range(3):
    fig.text(0.225 + j * 0.25, 0.9, col_labels[j], fontsize=10, ha='center', va='bottom')

fig.suptitle("Phase oscillation simulation", fontsize=14, fontweight='bold', y=0.95)
fig.text(0.5, 0.05, "FS frequency (kHz)", ha='center', fontsize=12)
fig.text(-0.02, 0.5, "SS frequency (HZ)", va='center', rotation='vertical', fontsize=12)
plt.savefig("phase_oscillation_simulation.pdf", bbox_inches='tight')

