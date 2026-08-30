from dataclasses import dataclass
import gc
from linecache import cache

from kima import KimaResults
import kima
from kima.pykima.analysis import get_planet_mass, get_planet_semimajor_axis
from kima.pykima.utils import mjup2mearth, get_semi_amplitude
import numpy as np


@dataclass
class Box:
    """ A box in orbital period and planetary mass """
    period_limits: tuple
    mass_limits: tuple
    true_fraction: float | None = None
    units: str = 'earth'
    _cache = {}

    def __repr__(self) -> str:
        with np.printoptions(precision=3):
            p = f'period_limits={np.array(self.period_limits)}'
            m = f'mass_limits={np.array(self.mass_limits)}'
        return f"Box({p}, {m})"

    def __hash__(self) -> int:
        return hash((self.period_limits, self.mass_limits))

    def center(self, log=False, sma=False, star_mass=1.0):
        Pcenter_log = np.geomspace(*self.period_limits, 3)[1]
        Mcenter_log = np.geomspace(*self.mass_limits, 3)[1]
        Pcenter = np.linspace(*self.period_limits, 3)[1],
        Mcenter = np.linspace(*self.mass_limits, 3)[1],
        if sma:
            if log:
                a = float(get_planet_semimajor_axis(Pcenter_log, None, star_mass=star_mass))
                print(a)
                return a, Mcenter_log
            else:
                a = float(get_planet_semimajor_axis(Pcenter[0], None, star_mass=star_mass))
                print(a)
                return a, Mcenter
        if log:
            return Pcenter_log, Mcenter_log
        return Pcenter, Mcenter

    def get_a_lims(self, star_mass=1.0):
        a0 = float(get_planet_semimajor_axis(self.period_limits[0], None, star_mass=star_mass))
        a1 = float(get_planet_semimajor_axis(self.period_limits[1], None, star_mass=star_mass))
        return a0, a1

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

    def _clear_cache(self):
        self._cache = {}

    def calculate_f0(self, res: KimaResults | str, star_mass: float = 1.0,
                     N: int = 1_000_000, cache: bool=False,
                     sensitivity_limit: float | None = None,
                     full_output: bool = False, lazy: bool = False):
        """
        Calculate the prior probability for "at least one planet in the box"

        Args:
            res (KimaResults or str):
                _kima_ results object containing the priors, or path to a prior
                parquet file.
            N (int, optional):
                Number of prior samples. Defaults to 10^6.
            cache (bool, optional):
                If True, cache the prior samples. Defaults to False.
            full_output (bool, optional):
                If True, return the prior samples for period and mass. Defaults to False.
        """
        if isinstance(res, str):
            if lazy:
                res = kima.load_results(res)
                P, K, e = self._prior_samples(res, N, cache=cache)
                npmax = res.npmax
                del res
                gc.collect()
            else:
                import pandas as pd
                prior_samples = pd.read_parquet(
                    res,
                    filters=[
                        ("P", ">", self.period_limits[0]),
                        ("P", "<", self.period_limits[1]),
                    ],
                )
                npmax = prior_samples.attrs['npmax']
                N = prior_samples.attrs['N']
                P, K, e, star_mass = prior_samples.to_numpy().T
        else:
            P, K, e = self._prior_samples(res, N, cache=cache)
            npmax = res.npmax

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
    
        if sensitivity_limit is not None:
            cond &= (K < sensitivity_limit)
            F_sens = cond.sum() / N
            f0_sens = 1 - sum([prior_Np * (1 - F_sens) ** Np for Np in range(npmax + 1)])
            f0 = (f0, f0_sens)

        if full_output:
            return (f0, F), (P, m, cond)
        return f0

    def fraction_inside(self, res: KimaResults | str, star_mass: float=1.0,
                        sensitivity_limit: float | None = None, lazy: bool = False):
        """ Calculate the fraction of posterior samples inside the box """
        if isinstance(res, str):
            if lazy:
                res = kima.load_results(res)
                P = res.posteriors.P
                K = res.posteriors.K
                m = res.posteriors.msini(star_mass=star_mass, units=self.units)
                ESS = res.ESS
                del res
                gc.collect()
            else:
                import pandas as pd
                df = pd.read_parquet(res)
                P = df[df.columns[df.attrs['indices']['planets.P']]].to_numpy()
                K = df[df.columns[df.attrs['indices']['planets.K']]].to_numpy()
                e = df[df.columns[df.attrs['indices']['planets.e']]].to_numpy()
                m = get_planet_mass(P, K, e, star_mass=star_mass, full_output=True)[-1]
                if self.units == 'earth':
                    m *= mjup2mearth
                ESS = df.shape[0]
        else:
            P = res.posteriors.P
            K = res.posteriors.K
            m = res.posteriors.msini(star_mass=star_mass, units=self.units)
            ESS = res.ESS
        # check if inside the box
        cond = (m > self.mass_limits[0]) & (m < self.mass_limits[1])
        cond &= (P > self.period_limits[0]) & (P < self.period_limits[1])
        # "at least one planet"
        nsamples = cond.any(axis=1).sum()
        # fraction of all posterior samples
        fraction = nsamples / ESS

        if sensitivity_limit is not None:
            cond_sens = cond & (K < sensitivity_limit)
            cond = (cond, cond_sens)

        return fraction, cond, ESS

    def show(self, ax, sma=False, semi_amplitude=False, star_mass=1.0,
             **kwargs):
        """ Show the box as a rectangle on a matplotlib axis """
        from matplotlib.patches import Rectangle, Polygon
        pl = self.period_limits
        ml = self.mass_limits
        if semi_amplitude:
            kl = []
            factor = 1 if self.units == 'earth' else mjup2mearth
            kl.append(get_semi_amplitude(pl[0], ml[0] * factor))
            kl.append(get_semi_amplitude(pl[0], ml[1] * factor))
            kl.append(get_semi_amplitude(pl[1], ml[0] * factor))
            kl.append(get_semi_amplitude(pl[1], ml[1] * factor))
            print(kl)
            x = [pl[0], pl[1], pl[1], pl[0]]
            y = [kl[0], kl[3], kl[2], kl[1]]
            ax.add_patch(Polygon(xy=list(zip(x, y)), fill=False))
        if sma:
            a0 = float(get_planet_semimajor_axis(pl[0], K=None, star_mass=star_mass))
            a1 = float(get_planet_semimajor_axis(pl[1], K=None, star_mass=star_mass))
            xy = a0, ml[0]
            width = a1 - a0
            height = ml[1] - ml[0]
            print(xy, width, height)
            rect = Rectangle(xy, width, height, **kwargs)
            ax.add_artist(rect)
            return rect
        else:
            xy = pl[0], ml[0]
            width = pl[1] - pl[0]
            height = ml[1] - ml[0]
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

@dataclass
class Triangle:
    """ A triangle in parameter space """
    period_limits: tuple
    mass_limits: tuple
    units: str = 'earth'
    # m | .__.
    #   | ./
    #   ______ P

    def show(self, ax, semi_amplitude=False, **kwargs):
        """ Show the box as a rectangle on a matplotlib axis """
        from matplotlib.patches import Rectangle, Polygon
        pl = self.period_limits
        ml = self.mass_limits
        if semi_amplitude:
            raise NotImplementedError
        else:
            x = [pl[0], pl[0], pl[1], pl[0]]
            y = [ml[0], ml[1], ml[1], ml[0]]
            ax.plot(x, y, **kwargs)
            ax.fill(x, y)


def boxes_in_range(period_range, mass_range, n_period, n_mass, units='earth'):
    plims = np.geomspace(*period_range, n_period + 1).tolist()
    mlims = np.geomspace(*mass_range, n_mass + 1).tolist()
    print(plims, mlims)
    return [
        Box((plims[i], plims[i + 1]), (mlims[j], mlims[j + 1]), units=units)
        for i in range(n_period)
        for j in range(n_mass)
    ]