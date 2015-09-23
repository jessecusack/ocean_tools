# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 12:17:53 2014

@author: jc3e13
"""

import matplotlib.pyplot as plt
import os


def savefig(fig, fid, fname, sdir, fsize=None, lock_aspect=True, ftype='png',
            font_size=None):
    """Savefig function that rescales plots for publication in AMS journals."""

    scol = 3.125  # inches
    dcol = 6.5  # inches

    if fsize is None:
        pass
    elif isinstance(fsize, tuple):
        fig.set_size_inches(fsize)
    elif fsize == 'single_col':
        cfsize = fig.get_size_inches()
        if lock_aspect:
            r = scol/cfsize[0]
            fig.set_size_inches(cfsize*r)
        else:
            fig.set_size_inches(scol, cfsize[1])
    elif fsize == 'double_col':
        cfsize = fig.get_size_inches()
        if lock_aspect:
            r = dcol/cfsize[0]
            fig.set_size_inches(cfsize*r)
        else:
            fig.set_size_inches(dcol, cfsize[1])

    if font_size is None:
        pass
    else:
        axs = fig.get_axes()
        for ax in axs:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                         ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(font_size)
        fig.canvas.draw()

    fname = str(fid) + '_' + fname
    fname = os.path.join(sdir, fname) + '.' + ftype
    plt.savefig(fname, dpi=300., bbox_inches='tight')
