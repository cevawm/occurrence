<p align="center">
  <img width = "300" src="https://github.com/kima-org/occurrence/blob/main/logo.png?raw=true"/>
</p>
<h2 align="center">
planet occurrence rates from radial velocities
</h2>

This repository contains an implementation of the method described in [Faria,
Delisle, and Ségransan
(2025)](https://ui.adsabs.harvard.edu/abs/bibcode/abstract) for
the calculation of planet occurrence rates from radial-velocity observations.

The method is simple, and so is the code.  
The file `box.py` defines a class `Box` that represents the region in orbital
period and planetary minimum mass for which the planet occurrence rate is
calculated.

```py
>>> Box((1, 25), (10, 30))
Box(period_limits=(1, 25), mass_limits=(10, 30), true_fraction=None, units='earth')
```

This class contains only two methods:

- `calculate_f0`: calculates the prior probability for "at least one planet in
  the box", denoted by $f_0$ in the paper, using Monte Carlo. It takes as
  arguments a `KimaResults` object that provides the priors used in the RV
  analysis and, optionally, the stellar mass in solar masses.

- `fraction_inside`: calculates the fraction of posterior samples inside the
  box, taking the same arguments. 


The file `occurrence.py` defines a class `occurrence` which calculates the
posterior distribution for the planet occurrence rate in a given box.