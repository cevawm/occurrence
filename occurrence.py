from collections import defaultdict
from dataclasses import dataclass
import os
import numpy as np
from scipy.integrate import quad
from tqdm import tqdm

from kima import KimaResults
from box import Box

@dataclass
class occurrence:
    """ 
    Calculate planet occurrence rates from radial velocities

    Args:
        RES (list):
            List of results objects from kima
        boxes (list):
            List of Box objects representing the regions in orbital period and
            planetary mass
    """
    RES: list[KimaResults | str]
    boxes: list[Box]
    star_masses: list[float] | None = None
    _parquet = False

    @classmethod
    def from_parquet(cls, RES: list[str], boxes: list[Box], star_masses: list[float] | None = None):
        """ Alternative constructor to load results from parquet files

        Args:
            RES (list):
                List of paths to parquet files containing the kima results
        """
        if star_masses is None:
            star_masses = [1.0] * len(RES)
        else:
            assert isinstance(star_masses, list), "`star_masses` must be a list"
            assert len(star_masses) == len(RES), "length of `star_masses` should match length of `RES`"

        instance = cls(RES, boxes, star_masses)
        instance._parquet = True
        instance.compute_fractions()
        return instance

    def __post_init__(self):
        if isinstance(self.boxes, Box):
            self.boxes = [self.boxes]

        self.f0 = {}
        self.fractions = {}

        # try getting stellar mass from results, or use 1 solar mass
        if self.star_masses is None:
            self.star_masses = []
            for res in self.RES:
                try:
                    self.star_masses.append(res.star_mass)
                except AttributeError:
                    self.star_masses.append(1.0)
        else:
            assert isinstance(self.star_masses, list), "`star_masses` must be a list"
            assert len(self.star_masses) == len(self.RES), "length of `star_masses` should match length of `RES`"

    def compute_f0(self, assume_same_star_mass=True, N=1_000_000, cache=False):
        RES = self.RES
        if self._parquet:
            RES = [os.path.join(R, 'prior.parquet') for R in RES]

        kw = dict(N=N, cache=cache)

        for box in self.boxes:
            print(f'computing f0 for {box}')
            if assume_same_star_mass:
                f0 = box.calculate_f0(RES[0], **kw)
                self.f0[box] = f0
            else:
                pbar = tqdm(zip(RES, self.star_masses), total=len(RES))
                f0 = []
                for obj in pbar:
                    res, star_mass = obj
                    f0.append(box.calculate_f0(res, star_mass=star_mass, **kw))
                self.f0[box] = np.array(f0)


    def compute_fractions(self):
        RES = self.RES
        if self._parquet:
            RES = [os.path.join(R, 'posterior_sample.parquet') for R in RES]

        # calculate fraction of posterior samples inside the box(es)
        print('computing posterior fractions')
        fractions = defaultdict(list)
        for i, res in tqdm(enumerate(RES), total=len(RES)):
            for box in self.boxes:
                fraction, cond = box.fraction_inside(res, star_mass=self.star_masses[i])
                fractions[box].append(fraction)
        # convert to numpy arrays
        for box in self.boxes:
            self.fractions[box] = np.array(fractions[box])

    def estimate_posterior(self, assume_same_star_mass=True, N=1_000_000,
                           cache=False, prior=None, beta=1.0, k=1):
        """
        Calculate the posterior distribution for planet occurrence in each box.
        This assumes a uniform prior and calculates the posterior on a grid.

        Args:
            assume_same_star_mass (bool, optional):
                If True, assume all stars have the same mass and don't redo the
                calculation of f0. Defaults to True.
            N (int, optional):
                Number of prior samples to compute f0. Defaults to 10^6.
            cache (bool, optional):
                If True, cache the prior samples. Only valid if all the priors
                are the same. Defaults to False.
        """
        if len(self.f0) == 0:
            self.compute_f0(assume_same_star_mass=assume_same_star_mass, N=N, cache=cache)
        if len(self.fractions) == 0:
            self.compute_fractions()

        self.f = np.linspace(0, 1, 100)

        if prior is None:
            self.logprior = np.zeros_like(self.f)
        else:
            self.logprior = prior(self.f)

        def gain(x, k):
            return x**k / (x**k + (1 - x) ** k)

        self.loglike = []
        for box in self.boxes:
            f0 = self.f0[box]
            fration = gain(self.fractions[box], k)
            # calculate log likelihood, p( {Dj} | f )
            self.loglike.append([
                np.nansum(
                    np.log(
                        (ff / f0) * fration + (1 - ff) / (1 - f0) * (1 - fration)
                    )
                )
                for ff in self.f
            ])
        self.loglike = np.array(self.loglike)
        # posterior distribution
        self.posterior = np.exp(beta * self.loglike + self.logprior)
        # normalize by integral
        self.normalized_posterior = self.posterior.copy()
        for i in range(len(self.boxes)):
            integral = quad(lambda x: np.interp(x, self.f, self.posterior[i]), 
                            0, 1, limit=300)[0]
            self.normalized_posterior[i] /= integral
        # returns unnormalized posterior
        return self.posterior

    def posterior_mean_std(self, **kwargs):
        """
        Calculate the mean and standard deviation of the posterior for planet
        occurrence in each box. Keyword arguments are passed directly to
        `estimate_posterior`
        """
        self.estimate_posterior(**kwargs)
        M, S = [], []
        for i, box in enumerate(self.boxes):
            f, p = self.f, self.normalized_posterior[i]
            m = np.sum(f * p) / np.sum(p)
            M.append(m)
            std = np.sqrt(np.sum(f**2 * p)/np.sum(p)- (np.sum(f * p)/np.sum(p))**2)
            S.append(std)
        return M, S

    def plot(self, ax=None, sharey=False, colors=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            nrows, ncols = 1, len(self.boxes)
            figsize = len(self.boxes) * 4, 4
            if len(self.boxes) > 6:
                nrows, ncols = 2, len(self.boxes) // 2
                figsize = len(self.boxes) * 2, 6
            if len(self.boxes) > 12:
                nrows, ncols = 3, len(self.boxes) // 3
                figsize = len(self.boxes) * 2, 8

            fig, axs = plt.subplots(nrows, ncols, constrained_layout=True,
                                    sharey=sharey, figsize=figsize)
            axs = np.atleast_1d(axs).ravel()
        else:
            axs = np.atleast_1d(ax)
            assert len(axs) == len(self.boxes), f"number of axes should match number of boxes ({len(self.boxes)})"
            fig = axs[0].figure

        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            # colors = [f'C{i}' for i in range(len(self.boxes))]

        lw = kwargs.pop('lw', 1)
        framealpha = kwargs.pop('framealpha', 0.5)
        legend_fontsize = kwargs.pop('legend_fontsize', 12)

        means, std = self.posterior_mean_std(**kwargs)

        for i, (box, ax) in enumerate(zip(self.boxes, axs)):
            # color = f'C{i}'
            color = colors[i]
            label = 'posterior\n' + fr'{means[i]:.2f} $\pm$ {std[i]:.2f}'
            ax.plot(self.f, self.normalized_posterior[i], color=color, lw=3*lw)
            ax.plot(self.f, np.exp(self.logprior), ls=':', color=color, lw=lw)
            if box.true_fraction is not None:
                ax.axvline(x=box.true_fraction, color=color, ls='--')
            ax.legend([label, 'prior', 'truth'], 
                      framealpha=framealpha, fontsize=legend_fontsize)
            units = r'$M_\oplus$' if box.units == 'earth' else r'M$_{\rm Jup}$'
            ax.set_title(
                r'$P \in$' + f'{box.period_limits} d' + '\n' + 
                r'$m \in$' + f'{box.mass_limits}' + f' {units}',
                loc='left'
            )
            ax.minorticks_on()
            ax.set(xlim=(0, 1), ylim=(0, None), xlabel='$f_R$')

        axs[0].set(ylabel='pdf')

        if sharey:
            axs[0].set_ylim(0, np.ceil(self.normalized_posterior.max()))

        fig.suptitle(f'{len(self.RES)} stars')

        return fig
    
    def plot_in_boxes(self, ax=None, just_boxes=False):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        means, std = self.posterior_mean_std()
        for i, (box, μ) in enumerate(zip(self.boxes, means)):
            box.show(ax, facecolor="none", edgecolor="k", alpha=0.1)
            if just_boxes:
                continue
            box.show(ax=ax, color=plt.cm.viridis(μ))
            _x, _y = box.center(log=True)
            ax.text(_x, _y, f'{μ:.2f}', ha='center', va='center',
                    color='w' if μ < 0.5 else 'k')





@dataclass
class occurrence_with_limit:
    """ 
    Calculate planet occurrence rates from radial velocities

    Args:
        RES (list):
            List of results objects from kima
        boxes (list):
            List of Box objects representing the regions in orbital period and
            planetary mass
    """
    RES: list[KimaResults | str]
    boxes: list[Box]
    star_masses: list[float] | None = None
    sensitivity_limits: list[float] | None = None
    _parquet = False

    @classmethod
    def from_parquet(cls, RES: list[str], boxes: list[Box], star_masses: list[float] | None = None):
        """ Alternative constructor to load results from parquet files

        Args:
            RES (list):
                List of paths to parquet files containing the kima results
        """
        if star_masses is None:
            star_masses = [1.0] * len(RES)
        else:
            assert isinstance(star_masses, list), "`star_masses` must be a list"
            assert len(star_masses) == len(RES), "length of `star_masses` should match length of `RES`"

        instance = cls(RES, boxes, star_masses)
        instance._parquet = True
        instance.compute_fractions()
        return instance

    def __post_init__(self):
        if isinstance(self.boxes, Box):
            self.boxes = [self.boxes]

        self.NR = len(self.RES)
        self.NB = len(self.boxes)

        self.f0 = {}
        self.f0_sens = {}
        self.fractions = {}

        # try getting stellar mass from results, or use 1 solar mass
        if self.star_masses is None:
            self.star_masses = []
            for res in self.RES:
                try:
                    self.star_masses.append(res.star_mass)
                except AttributeError:
                    self.star_masses.append(1.0)
        else:
            assert isinstance(self.star_masses, list), "`star_masses` must be a list"
            assert len(self.star_masses) == self.NR, "length of `star_masses` should match length of `RES`"

        if self.sensitivity_limits is None:
            self.sensitivity_limits = [np.inf] * self.NR
        else:
            assert isinstance(self.sensitivity_limits, list), "`sensitivity_limits` must be a list"
            assert len(self.sensitivity_limits) == self.NR, "length of `sensitivity_limits` should match length of `RES`"

    def compute_f0(self, assume_same_star_mass=True, N=1_000_000, cache=False):
        RES = self.RES
        if self._parquet:
            RES = [os.path.join(R, 'prior.parquet') for R in RES]

        kw = dict(N=N, cache=cache)

        for box in self.boxes:
            print(f'computing f0 for {box}')
            if assume_same_star_mass:
                f0, f0_sens = box.calculate_f0(RES[0], sensitivity_limit=self.sensitivity_limits[0], **kw)
                self.f0_sens[box] = np.full(self.NR, f0_sens)
                self.f0[box] = np.full(self.NR, f0)
            else:
                pbar = tqdm(zip(RES, self.star_masses, self.sensitivity_limits), total=self.NR)
                f0, f0_sens = [], []
                for obj in pbar:
                    res, star_mass, sensitivity_limit = obj
                    _f0, _f0_sens = box.calculate_f0(res, star_mass=star_mass, sensitivity_limit=sensitivity_limit, **kw)
                    f0.append(_f0)
                    f0_sens.append(_f0_sens)
                self.f0_sens[box] = np.array(f0_sens)
                self.f0[box] = np.array(f0)


    def compute_fractions(self):
        RES = self.RES
        if self._parquet:
            RES = [os.path.join(R, 'posterior_sample.parquet') for R in RES]

        # calculate fraction of posterior samples inside the box(es)
        print('computing posterior fractions')
        fractions = defaultdict(list)
        for i, res in tqdm(enumerate(RES), total=len(RES)):
            for box in self.boxes:
                fraction, cond = box.fraction_inside(
                    res, 
                    star_mass=self.star_masses[i],
                    sensitivity_limit=self.sensitivity_limits[i]
                )
                cond, cond_sens = cond
                ratio = self.f0_sens[box][i] / self.f0[box][i]
                # print(f'ratio={ratio}')
                contrib = cond.any(axis=1).astype(float)
                contrib[cond_sens.any(axis=1)] = ratio
                fraction = contrib.sum() / res.ESS

                fractions[box].append(fraction)

        # convert to numpy arrays
        for box in self.boxes:
            self.fractions[box] = np.array(fractions[box])

    def estimate_posterior(self, assume_same_star_mass=True, N=1_000_000,
                           cache=False, prior=None, beta=1.0):
        """
        Calculate the posterior distribution for planet occurrence in each box.
        This assumes a uniform prior and calculates the posterior on a grid.

        Args:
            assume_same_star_mass (bool, optional):
                If True, assume all stars have the same mass and don't redo the
                calculation of f0. Defaults to True.
            N (int, optional):
                Number of prior samples to compute f0. Defaults to 10^6.
            cache (bool, optional):
                If True, cache the prior samples. Only valid if all the priors
                are the same. Defaults to False.
        """
        if len(self.f0) == 0:
            self.compute_f0(assume_same_star_mass=assume_same_star_mass, N=N, cache=cache)
        if len(self.fractions) == 0:
            self.compute_fractions()

        self.f = np.linspace(0, 1, 100)

        if prior is None:
            self.logprior = np.zeros_like(self.f)
        else:
            self.logprior = prior(self.f)

        self.loglike = []
        for box in self.boxes:
            f0 = self.f0[box]
            fration = self.fractions[box]
            # calculate log likelihood, p( {Dj} | f )
            self.loglike.append([
                np.nansum(
                    np.log(
                        (ff / f0) * fration + (1 - ff) / (1 - f0) * (1 - fration)
                    )
                )
                for ff in self.f
            ])
        self.loglike = np.array(self.loglike)
        # posterior distribution
        self.posterior = np.exp(beta * self.loglike + self.logprior)
        # normalize by integral
        self.normalized_posterior = self.posterior.copy()
        for i in range(len(self.boxes)):
            integral = quad(lambda x: np.interp(x, self.f, self.posterior[i]), 
                            0, 1, limit=300)[0]
            self.normalized_posterior[i] /= integral
        # returns unnormalized posterior
        return self.posterior

    def posterior_mean_std(self, **kwargs):
        """
        Calculate the mean and standard deviation of the posterior for planet
        occurrence in each box. Keyword arguments are passed directly to
        `estimate_posterior`
        """
        self.estimate_posterior(**kwargs)
        M, S = [], []
        for i, box in enumerate(self.boxes):
            f, p = self.f, self.normalized_posterior[i]
            m = np.sum(f * p) / np.sum(p)
            M.append(m)
            std = np.sqrt(np.sum(f**2 * p)/np.sum(p)- (np.sum(f * p)/np.sum(p))**2)
            S.append(std)
        return M, S

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            nrows, ncols = 1, len(self.boxes)
            figsize = len(self.boxes) * 4, 4
            if len(self.boxes) > 6:
                nrows, ncols = 2, len(self.boxes) // 2
                figsize = len(self.boxes) * 2, 6
            if len(self.boxes) > 12:
                nrows, ncols = 3, len(self.boxes) // 3
                figsize = len(self.boxes) * 2, 8

            fig, axs = plt.subplots(nrows, ncols, constrained_layout=True,
                                    sharey=False, figsize=figsize)
            axs = np.atleast_1d(axs).ravel()
        else:
            axs = np.atleast_1d(ax)
            assert len(axs) == len(self.boxes), f"number of axes should match number of boxes ({len(self.boxes)})"
            fig = axs[0].figure

        color = kwargs.pop('color', 'C0')
        lw = kwargs.pop('lw', 1)

        means, std = self.posterior_mean_std(**kwargs)

        for i, (box, ax) in enumerate(zip(self.boxes, axs)):
            # color = f'C{i}'
            label = 'posterior\n' + fr'{means[i]:.2f} $\pm$ {std[i]:.2f}'
            ax.plot(self.f, self.normalized_posterior[i], color=color, lw=lw)
            ax.plot(self.f, np.exp(self.logprior), ls=':', color=color, lw=lw)
            if box.true_fraction is not None:
                ax.axvline(x=box.true_fraction, color=color, ls='--')
            ax.legend([label, 'prior', 'truth'])
            units = r'$M_\oplus$' if box.units == 'earth' else r'M$_{\rm Jup}$'
            ax.set_title(
                r'$P \in$' + f'{box.period_limits} d' + '\n' + 
                r'$m \in$' + f'{box.mass_limits}' + f' {units}',
                loc='left'
            )
            ax.minorticks_on()
            ax.set(xlim=(0, 1), ylim=(0, None), xlabel='$f_R$')

        axs[0].set(ylabel='pdf')
        fig.suptitle(f'{len(self.RES)} stars')

        return fig
    
    def plot_in_boxes(self, ax=None, just_boxes=False):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()

        means, std = self.posterior_mean_std()
        for i, (box, μ) in enumerate(zip(self.boxes, means)):
            box.show(ax, facecolor="none", edgecolor="k", alpha=0.1)
            if just_boxes:
                continue
            box.show(ax=ax, color=plt.cm.viridis(μ))
            _x, _y = box.center(log=True)
            ax.text(_x, _y, f'{μ:.2f}', ha='center', va='center',
                    color='w' if μ < 0.5 else 'k')

