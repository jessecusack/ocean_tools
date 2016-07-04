# -*- coding: utf-8 -*-
"""
Created on Tue Apr 08 12:17:53 2014

@author: jc3e13
"""

import matplotlib.pyplot as plt
import os


def my_savefig(fig, name, sdir, fsize=None, lock_aspect=True,
               ftype='png', font_size=None, **kwargs):
    """Savefig function that rescales plots for publication in AMS journals.

    Parameters
    ----------
    fig: figure object
        Figure to save.
    name: string
        Filename not including extension.
    sdir: string
        Save directory.
    fsize: tuple
        Figure size in inches, e.g. (4, 2). Alternatively, pass 'single_col' or
        'double_col' to use AMS double and single column widths which are 3.125
        and 6 inches.
    lock_aspect: boolean, optional
        Fix aspect ratio on resize, true by default.
    ftype: string, optional
        Save file type, 'png' by default.
    font_size: int
        Universal font size for all figure elements. Currently not working.
    kwargs: optional
        Additional arguments to the savefig function.

    """

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

    if type(ftype) is tuple:
        for ft in ftype:
            fn = os.path.join(sdir, name) + '.' + ft
            plt.savefig(fn, dpi=300., bbox_inches='tight', pad_inches=0.,
                        **kwargs)
    else:
        fname = os.path.join(sdir, name) + '.' + ftype
        plt.savefig(fname, dpi=300., bbox_inches='tight', pad_inches=0.,
                    **kwargs)
