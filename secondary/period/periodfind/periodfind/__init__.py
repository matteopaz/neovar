import warnings
import numpy as np

# Copyright 2020 California Institute of Technology. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
# Author: Ethan Jaszewski

"""
Provides an API for analyzing light curves using periodograms.
"""

class Statistics:
    """Stores statistics about a single set of parameters.

    Stores various periodogram statistics, as well as the parameters and
    value for a single test statistic of interest.

    Parameters
    ----------
    params : list of float
        List of paramters that produce this object's value

    value : float
        Value of the test statistic with the given params

    mean : float
        Periodogram test statistic mean

    std : float
        Periodogram test statistic std

    median : float
        Periodogram test statistic median

    mad : float
        Periodogram test statistic median absolute deviation

    significance_type : {'stdmean', 'madmedian'}, default='stdmean'
        Specifies the significance statistic that should be used. The `stdmean`
        statistic gives a rough estimate of how likely the value is. The
        `madmedian` gives a roughly analagous statistic, but is more robust.

    Attributes
    ----------
    params : list of float
        List of paramters that produce this object's value

    value : float
        Value of the test statistic with the given params

    mean : float
        Periodogram test statistic mean

    std : float
        Periodogram test statistic std

    median : float
        Periodogram test statistic median

    mad : float
        Periodogram test statistic median absolute deviation

    significance_type : {'stdmean', 'madmedian'}
        Specifies the significance statistic that should be used. The `stdmean`
        statistic gives a rough estimate of how likely the value is. The
        `madmedian` gives a roughly analagous statistic, but is more robust.

    significance : float
        Significance statistic, computed according to the significance_type
    """
    def __init__(self,
                 params,
                 value,
                 mean,
                 std,
                 median,
                 mad,
                 significance_type='stdmean'):
        self.params = params
        self.value = value
        self.mean = mean
        self.std = std
        self.median = median
        self.mad = mad
        self.significance_type = significance_type

    @property
    def significance(self):
        if self.significance_type == 'stdmean':
            return abs(self.value - self.mean) / self.std
        elif self.significance_type == 'madmedian':
            return abs(self.value - self.median) / self.mad
        else:
            raise NotImplementedError('Statistic ' + self.significance_type +
                                      ' not implemented')

    @staticmethod
    def statistics_from_data(data,
                             params,
                             use_max,
                             mean=None,
                             std=None,
                             median=None,
                             mad=None,
                             n=1,
                             significance_type='stdmean'):
        """Constructs statistics objects from a periodogram.

        Parameters
        ----------
        data : ndarray
            Periodogram data to find statistics for

        mean : float, default=None
        Periodogram test statistic mean. Calculated if not provided.

        std : float, default=None
            Periodogram test statistic std. Calculated if not provided.

        median : float, default=None
            Periodogram test statistic median. Calculated if not provided.

        mad : float, default=None
            Periodogram test statistic median absolute deviation. Calculated
            if not provided.

        n : int, default=1
            Number of `Statistics` to generate

        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Specifies the significance statistic that should be used. See class
            documentation for more information.

        Returns
        -------
        stats : `Statistics` or list of `Statistics`
            Statistics for the top `n` parameters
        """

        # Find best parameters
        if not use_max:
            partition = np.argpartition(data, n, axis=None)[:n]
        else:
            partition = np.argpartition(data, len(data) - n, axis=None)[-n:]

        idxs = np.unravel_index(partition, data.shape)
        idxs_t = []
        for i in range(n):
            idx = tuple(dim[i] for dim in idxs)
            idxs_t.append(idx)
        
        values = data[idxs]

        # Calculate the data-wide statistics
        if mean == None:
            mean = np.mean(data)
        if std == None:
            std = np.std(data)
        if median == None:
            median = np.median(data)
        if mad == None:
            mad = np.median(np.abs(data - median))

        best = []
        for (idx, val) in zip(idxs_t, values):
            param = [params[i][idx[i]] for i in range(len(params))]
            best.append(
                Statistics(
                    param,
                    val,
                    mean,
                    std,
                    median,
                    mad,
                    significance_type,
                ))

        # Sort by value so most significant is first
        best.sort(key=lambda s: s.value, reverse=use_max)

        if n == 1:
            return best[0]
        else:
            return best


class Periodogram:
    """Stores a full periodogram.

    Stores a full periodogram, including both the test statistic values and
    all trial parameters. Allows for plotting of the periodogram, as well as
    analysis beyond the statistics results returned by period finding
    algorithms.

    Parameters
    ----------
    periodogram : ndarray
        Periodogram test statistic values

    params : list of ndarray
        List of periodogram test parameters

    use_max : bool
        Whether likely periods are periodogram maxima

    Attributes
    ----------
    use_max : bool
        Whether likely periods are periodogram maxima

    data : ndarray
        Periodogram test statistic values

    params : list of ndarray
        List of periodogram test parameters

    mean : float
        Periodogram test statistic mean

    std : float
        Periodogram test statistic std

    median : float
        Periodogram test statistic median

    mad : float
        Periodogram test statistic median absolute deviation

    Notes
    -----
    For large periodograms, the memory usage of storing full periodograms
    can be prohibitively high, necessitating the use of the `Statistics`
    class instead.

    Computes parameters on demand, and does not cache them, so statistics
    should be stored if necessary.
    """
    def __init__(self, periodogram, params, use_max):
        self.use_max = use_max
        self.data = periodogram
        self.params = params

        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.median = np.median(self.data)
        self.mad = np.median(np.abs(self.data - self.median))

    def best_params(self, n=1, significance_type='stdmean'):
        """Returns the best parameters of the periodogram.

        Computes the best parameters of the periodogram, returning them as
        statistics objects.

        Parameters
        ----------
        n : int, default=1
            The number of top parameters to return

        significance_type : {'stdmean', 'madmedian'}, default='stdmean'
            Specifies the significance statistic that should be used. See the
            documentation for the `Statistics` class for more information.

        Returns
        -------
        stats : `Statistics` or list of `Statistics`
            Statistics for the top `n` parameters
        """
        return Statistics.statistics_from_data(
            self.data,
            self.params,
            self.use_max,
            mean=self.mean,
            std=self.std,
            median=self.median,
            mad=self.mad,
            n=n,
            significance_type=significance_type)

def _py_warn_periodfind(message, type):
    """Wrapper around warnings.warn for Cython code."""
    warnings.warn(message, type, stacklevel=2)
