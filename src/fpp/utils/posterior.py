import os
import numpy as np
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from fpp.models.scd import dnds, dnds_1b
from fpp.utils.utils import jnp_trapezoid

wdir = os.path.abspath(os.path.dirname(__file__))
mpl.rc_file(os.path.join(wdir, 'matplotlibrc'))


def dnds_posterior(
    samples, theta_keys, plot=True, ax=None, **kwargs
):
    """Posterior plot for dNdS.

    Args:
        samples (dict):       Dict of parameter name to array of samples.
        theta_keys (list):    Keys in samples to plot, e.g. ['Sps', 'n1', 'n2', 'n3', 'sb1', 'lambdas'] or ['Sps', 'n1', 'n2', 'sb'].
        ax (matplotlib.axis): Matplotlib axis to plot on. If None, a new figure and axis will be created.
        **kwargs: additional keyword arguments to pass to the plotting function.
    """

    Sps_arr = samples[theta_keys[0]]
    theta_arr = [np.ones_like(samples[theta_keys[0]])] # temp A = 1
    for k in theta_keys[1:]:
        theta_arr.append(samples[k])
    theta_arr = np.stack(theta_arr, axis=-1) # (n_samples, n_theta)

    s = jnp.logspace(-1, 2, 100)
    
    if len(theta_keys) == 6:
        theta_arr[:, -1] = theta_arr[:, -2] * theta_arr[:, -1] # sb2 = sb1 * lambdas
        dnds_vmap = jax.vmap(dnds, in_axes=(None, 0)) # vectorize over theta
    elif len(theta_keys) == 4:
        dnds_vmap = jax.vmap(dnds_1b, in_axes=(None, 0))
    else:
        raise ValueError("theta_keys should have length 4 or 6.")

    dnds_arr = dnds_vmap(s, theta_arr) # (n_samples, n_s)
    Stot_arr = jnp_trapezoid(s[None, :] * dnds_arr, s, axis=1) # (n_samples,)
    dnds_arr = dnds_arr / Stot_arr[:, None] * Sps_arr[:, None] # normalize to Sps

    dnds_med = np.median(dnds_arr, axis=0)
    dnds_68 = np.percentile(dnds_arr, [16, 84], axis=0)
    dnds_95 = np.percentile(dnds_arr, [2.5, 97.5], axis=0)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(s, dnds_med, color='k')
        ax.fill_between(s, dnds_68[0], dnds_68[1], alpha=0.5, fc='C0', ec='none')
        ax.fill_between(s, dnds_95[0], dnds_95[1], alpha=0.3, fc='C0', ec='none')
        ax.set(xscale='log', yscale='log')

    return s, dnds_med, dnds_68, dnds_95


def multi_corner(
    samples_dict, plot_var_names, point_est=None,
    colors_dict=None, legend_dict=None,
    n_bins_1d=30, save_fn=None,
    legend_loc=None, **kwargs
):

    bins_1d_arr = []
    range_arr = []
    for vn in plot_var_names:
        vmin = np.min([np.min(s[vn]) for _, s in samples_dict.items()])
        vmax = np.max([np.max(s[vn]) for _, s in samples_dict.items()])
        if point_est:
            vmin = min(vmin, point_est[vn])
            vmax = max(vmax, point_est[vn])
        bins_1d_arr.append(np.linspace(vmin, vmax, n_bins_1d+1))
        range_arr.append([vmin, vmax])
    
    fig = None
    for ie, (samples_name, samples) in enumerate(samples_dict.items()):
        color = mpl.colors.to_hex(colors_dict[samples_name])
        default_kwargs = dict(
            show_titles=False,
            title_fmt=None,
            title_kwargs={"fontsize": 14},
            levels=[0.68, 0.95],
            color=color,
            plot_contours=True,
            fill_contours=False,
            plot_density=False,
            plot_datapoints=False,
            hist_kwargs={'density': True},
            contour_kwargs={'linewidths': [1.5, 2.5]}
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
    if legend_dict is not None:
        fig.legend(
            [mpl.lines.Line2D([0], [0], color=colors_dict[k], lw=3) for k in samples_dict if legend_dict[k] is not None],
            [legend_dict[k] for k in samples_dict if legend_dict[k] is not None],
            loc=legend_loc if legend_loc is not None else 'upper right',
            frameon=False, fontsize=30
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
            plt.savefig(save_fn, dpi=200)