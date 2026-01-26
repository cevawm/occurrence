from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from scipy.integrate import quad

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
    RES: list[KimaResults]
    boxes: list[Box]
    star_masses: list[float] | None = None

    def __post_init__(self):
        if isinstance(self.boxes, Box):
            self.boxes = [self.boxes]

        fractions = defaultdict(list)
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

        # calculate fraction of posterior samples inside the box(es)
        for i, res in enumerate(self.RES):
            for box in self.boxes:
                fraction, _ = box.fraction_inside(res, star_mass=self.star_masses[i])
                fractions[box].append(fraction)
        # convert to numpy arrays
        for box in self.boxes:
            self.fractions[box] = np.array(fractions[box])

    def estimate_posterior(self, assume_same_star_mass=True):
        """
        Calculate the posterior distribution for planet occurrence in each box.
        This assumes a uniform prior and calculates the posterior on a grid.
        """
        self.f = np.linspace(0, 1, 100)
        self.logprior = np.zeros_like(self.f)
        self.loglike = []
        for box in self.boxes:
            if assume_same_star_mass:
                f0 = box.calculate_f0(self.RES[0])
            else:
                f0 = np.array([
                    box.calculate_f0(res, star_mass=star_mass)
                    for res, star_mass in zip(self.RES, self.star_masses)
                ])
            # calculate log likelihood, p( {Dj} | f )
            self.loglike.append([
                np.nansum(
                    np.log(
                        (ff / f0) * self.fractions[box] + (1 - ff) / (1 - f0) * (1 - self.fractions[box])
                    )
                )
                for ff in self.f
            ])
        # posterior distribution
        self.posterior = np.exp(self.loglike + self.logprior)
        # normalize by integral
        self.normalized_posterior = self.posterior.copy()
        for i in range(len(self.boxes)):
            integral = quad(lambda x: np.interp(x, self.f, self.posterior[i]), 
                            0, 1, limit=300)[0]
            self.normalized_posterior[i] /= integral
        # returns unnormalized posterior
        return self.posterior

    def posterior_mean_std(self):
        if not hasattr(self, 'normalized_posterior'):
            self.estimate_posterior()
        M, S = [], []
        for i, box in enumerate(self.boxes):
            f, p = self.f, self.normalized_posterior[i]
            m = np.sum(f * p) / np.sum(p)
            M.append(m)
            std = np.sqrt(np.sum(f**2 * p)/np.sum(p)- (np.sum(f * p)/np.sum(p))**2)
            S.append(std)
        return M, S

    def plot(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(self.boxes), constrained_layout=True,
                                sharey=True, figsize=(len(self.boxes) * 4, 4))

        if not hasattr(self, 'normalized_posterior'):
            self.estimate_posterior()

        means, std = self.posterior_mean_std()

        for i, (box, ax) in enumerate(zip(self.boxes, axs)):
            color = f'C{i}'
            label = 'posterior\n' + fr'{means[i]:.2f} $\pm$ {std[i]:.2f}'
            ax.plot(self.f, self.normalized_posterior[i], color=color)
            ax.plot(self.f, np.exp(self.logprior), ls=':', color=color)
            if box.true_fraction is not None:
                ax.axvline(x=box.true_fraction, color=color, ls='--')
            ax.legend([label, 'prior', 'truth'])
            ax.set_title(
                r'$P \in$' + f'{box.period_limits} d' + '\n' + 
                r'$m \in$' + f'{box.mass_limits}' + f' {box.units} mass',
                loc='left'
            )
            ax.minorticks_on()
            ax.set(xlim=(0, 1), ylim=(0, None), xlabel='$f_R$')

        axs[0].set(ylabel='pdf')
        fig.suptitle(f'{len(self.RES)} stars')

        return fig