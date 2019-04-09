import numpy as np
from scipy import stats

class NumericStatistics:
    def __init__(self):
        pass

    @staticmethod
    def minimum(self, array):
        return np.amin(array)

    @staticmethod
    def maximum(self, array):
        return np.amax(array)

    @staticmethod
    def minimum_with_nan(self, array):
        return np.nanmin(array)

    @staticmethod
    def maximum_with_nan(self, array):
        return np.nanmax(array)

    @staticmethod
    def percentile(self, array):
        return np.percentile(array)

    @staticmethod
    def percentile_with_nan(self, array):
        return np.nanpercentile(array)

    @staticmethod
    def quantile(self, array):
        return np.quantile(array)

    @staticmethod
    def quantile_with_nan(self, array):
        return np.nanquantile(array)

    @staticmethod
    def median(self, array):
        return np.median(array)

    @staticmethod
    def mean(self, array):
        return np.mean(array)

    @staticmethod
    def standard_deviation(self, array):
        return np.std(array)

    @staticmethod
    def variance(self, array):
        return np.variance(array)

    @staticmethod
    def median_with_nan(self, array):
        return np.nanmedian(array)

    @staticmethod
    def mean_with_nan(self, array):
        return np.nanmean(array)

    @staticmethod
    def standard_deviation_with_nan(self, array):
        return np.nanstd(array)

    @staticmethod
    def variance_with_nan(self, array):
        return np.nanvar(array)

    @staticmethod
    def geometric_mean(self, array):
        return stats.gmean(array)

    @staticmethod
    def harmonic_mean(self, array):
        return stats.hmean(array)

    @staticmethod
    def kurtosis(self, array):
        return stats.kurtosis(array)

    @staticmethod
    def mode(self, array):
        return stats.mode(array)

    @staticmethod
    def skew(self, array):
        return stats.skew(array)

    @staticmethod
    def variation(self, array):
        return stats.variation(array)

    @staticmethod
    def find_repeats(self, array):
        return stats.find_repeats(array)

    @staticmethod
    def interquartile_range(self, array):
        return stats.iqr(array)

    @staticmethod
    def entropy(self, probabilities, alternative_probabilities=None):
        return stats.entropy(probabilities, alternative_probabilities)

    # trimean
    # things like this
    # trimmed statistics
    # investigate here: https://en.wikipedia.org/wiki/Descriptive_statistics
    
    # hypothesis testing class
    
    
