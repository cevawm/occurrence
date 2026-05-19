from dataclasses import dataclass
import gc

from kima import KimaResults
import kima
from kima.pykima.analysis import get_planet_mass
from kima.pykima.utils import mjup2mearth
import numpy as np


@dataclass
class Box:
    """ A box in orbital period and planetary mass """
    period_limits: tuple
    mass_limits: tuple
    true_fraction: float | None = None
    units: str = 'earth'
    _cache = {}

    def __hash__(self) -> int:
        return hash((self.period_limits, self.mass_limits))

    def _prior_samples(self, res: KimaResults, N: int = 1_000_000,
                       cache: bool=False):
        """ Generate samples from the priors (for f0 calculation)

        Args:
            res (KimaResults): _kima_ results object containing the priors
            N (int, optional): Number of prior samples. Defaults to 10^6.
            cache (bool, optional): If True, cache the prior samples. Defaults to False.

        Returns:
            tuple: (P, K, e) prior samples
        """
        if cache and (N, hash(self)) in self._cache:
            return self._cache[(N, hash(self))]
        # sample from the priors
        P = np.vectorize(res.priors['Pprior'].ppf)(np.random.rand(N))
        K = np.vectorize(res.priors['Kprior'].ppf)(np.random.rand(N))
        e = np.vectorize(res.priors['eprior'].ppf)(np.random.rand(N))
        if cache:
            self._cache[(N, hash(self))] = P, K, e
        return P, K, e

    def calculate_f0(self, res: KimaResults, star_mass: float = 1.0,
                     N: int = 1_000_000, cache: bool=False,
                     full_output: bool = False, lazy: bool = False):
        """
        Calculate the prior probability for "at least one planet in the box"

        Args:
            res (KimaResults):
                _kima_ results object containing the priors
            N (int, optional):
                Number of prior samples. Defaults to 10^6.
            cache (bool, optional):
                If True, cache the prior samples. Defaults to False.
            full_output (bool, optional):
                If True, return the prior samples for period and mass. Defaults to False.
        """
        if isinstance(res, str) and lazy:
            res = kima.load_results(res)
            P, K, e = self._prior_samples(res, N, cache=cache)
            npmax = res.npmax
            del res
            gc.collect()
        else:
            P, K, e = self._prior_samples(res, N, cache=cache)
            npmax = res.npmax
        # P, K, e = self._prior_samples(res, N, cache=cache)
        # calculate the planet mass
        m = get_planet_mass(P, K, e, star_mass=star_mass, full_output=True)[-1]
        if self.units == 'earth':
            m *= mjup2mearth
        # check if inside the box
        cond = (m > self.mass_limits[0]) & (m < self.mass_limits[1])
        cond &= (P > self.period_limits[0]) & (P < self.period_limits[1])
        F = cond.sum() / N
        prior_Np = 1 / (npmax + 1)
        f0 = 1 - sum([prior_Np * (1 - F) ** Np for Np in range(npmax + 1)])
        if full_output:
            return (f0, F), (P, m, cond)
        return f0

    def fraction_inside(self, res: KimaResults, star_mass: float=1.0, lazy: bool = False):
        """ Calculate the fraction of posterior samples inside the box """
        if isinstance(res, str) and lazy:
            res = kima.load_results(res)
            P = res.posteriors.P
            m = res.posteriors.msini(star_mass=star_mass, units=self.units)
            ESS = res.ESS
            del res
            gc.collect()
        else:
            P = res.posteriors.P
            m = res.posteriors.msini(star_mass=star_mass, units=self.units)
            ESS = res.ESS
        # P = res.posteriors.P
        # m = res.posteriors.msini(star_mass=star_mass, units=self.units)
        # check if inside the box
        cond = (m > self.mass_limits[0]) & (m < self.mass_limits[1])
        cond &= (P > self.period_limits[0]) & (P < self.period_limits[1])
        # "at least one planet"
        nsamples = cond.any(axis=1).sum()
        # fraction of all posterior samples
        fraction = nsamples / ESS
        return fraction, cond

    def show(self, ax, **kwargs):
        """ Show the box as a rectangle """
        from matplotlib.patches import Rectangle
        xy = (self.period_limits[0], self.mass_limits[0])
        width = self.period_limits[1] - self.period_limits[0]
        height = self.mass_limits[1] - self.mass_limits[0]
        rect = Rectangle(xy, width, height, **kwargs)
        ax.add_artist(rect)
        return rect

    def show_prior_samples(self, ax, res: KimaResults, all_samples=False, **kwargs):
        """ Show the prior samples on a matplotlib axis """
        _, (P, m, cond) = self.calculate_f0(res, full_output=True)
        kwargs.setdefault('ms', 1)
        kwargs.setdefault('color', 'k')
        kwargs.setdefault('alpha', 0.1)
        if all_samples:
            ax.plot(P, m, '.', **kwargs)
        else:
            ax.plot(P[cond], m[cond], '.', **kwargs)
        return P, m, cond