from dataclasses import dataclass

from kima import KimaResults
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

    def calculate_f0(self, res: KimaResults, star_mass: float = 1.0,
                     N: int = 1_000_000):
        """
        Calculate the prior probability for "at least one planet in the box"
        """
        # sample from the priors
        P = np.vectorize(res.priors['Pprior'].ppf)(np.random.rand(N))
        K = np.vectorize(res.priors['Kprior'].ppf)(np.random.rand(N))
        e = np.vectorize(res.priors['eprior'].ppf)(np.random.rand(N))
        # calculate the planet mass
        m = get_planet_mass(P, K, e, star_mass=star_mass, full_output=True)[-1]
        if self.units == 'earth':
            m *= mjup2mearth
        # check if inside the box
        cond = (m > self.mass_limits[0]) & (m < self.mass_limits[1])
        cond &= (P > self.period_limits[0]) & (P < self.period_limits[1])
        F = cond.sum() / N
        prior_Np = 1 / (res.npmax + 1)
        f0 = 1 - sum([prior_Np * (1 - F) ** Np for Np in range(res.npmax + 1)])
        return f0

    def fraction_inside(self, res: KimaResults, star_mass: float=1.0):
        """ Calculate the fraction of posterior samples inside the box """
        P = res.posteriors.P
        m = res.posteriors.msini(star_mass=star_mass, units=self.units)
        # check if inside the box
        cond = (m > self.mass_limits[0]) & (m < self.mass_limits[1])
        cond &= (P > self.period_limits[0]) & (P < self.period_limits[1])
        # "at least one planet"
        nsamples = cond.any(axis=1).sum()
        # fraction of all posterior samples
        fraction = nsamples / res.ESS
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
