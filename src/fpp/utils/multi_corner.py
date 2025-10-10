import os
import numpy as np
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt

current_file_path = os.path.abspath(os.path.dirname(__file__))
mpl.rc_file(os.path.join(current_file_path, '../../../analysis/matplotlibrc'))


def multi_corner(
    samples_dict, plot_var_names, point_est=None,
    colors_dict=None, labels_dict=None,
    n_bins_1d=30, save_fn=None, **kwargs
):

    bins_1d_arr = []
    range_arr = []
    for vn in plot_var_names:
        vmin = np.min([np.min(s[vn]) for _, s in samples_dict.items()])
        vmax = np.max([np.max(s[vn]) for _, s in samples_dict.items()])
        bins_1d_arr.append(np.linspace(vmin, vmax, n_bins_1d+1))
        range_arr.append([vmin, vmax])
    
    fig = None
    for ie, (samples_name, samples) in enumerate(samples_dict.items()):
        color = mpl.colors.to_hex(colors_dict[samples_name])
        default_kwargs = dict(
            show_titles=True,
            title_fmt=None,
            title_kwargs={"fontsize": 12},
            levels=[0.68, 0.95],
            color=color,
            plot_contours=True,
            fill_contours=False,
            plot_density=False,
            plot_datapoints=False,
            hist_kwargs={'density': True},
        )
        default_kwargs.update(kwargs)
        fig = corner.corner(
            samples,
            bins_1d_arr=bins_1d_arr,
            range=range_arr,
            var_names=plot_var_names,
            fig=fig,
            **default_kwargs
        )
    if labels_dict is not None:
        fig.legend(
            [mpl.lines.Line2D([0], [0], color=c) for k, c in colors_dict.items()],
            [labels_dict[k] for k, c in colors_dict.items()],
            loc='upper right'
        )

    if point_est is not None:
        ndim = len(plot_var_names)
        axs = np.array(fig.axes).reshape((ndim, ndim))
        point_est_color = 'k'

        for i, vn in enumerate(plot_var_names):
            axs[i, i].axvline(point_est[vn], color=point_est_color)

        for ri in range(ndim):
            for ci in range(ri):
                axs[ri, ci].plot(point_est[plot_var_names[ci]],
                                 point_est[plot_var_names[ri]],
                                 '*', color=point_est_color, ms=10)
                
    if save_fn is not None:
        if save_fn.endswith('.pdf'):
            plt.savefig(save_fn)
        else:
            plt.savefig(save_fn, dpi=300)