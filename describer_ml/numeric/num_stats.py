import pandas as pd
import numpy as np
from thinkbayes2 import Cdf as CDF
from scipy import stats
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing 

def minimum(array):
    return np.amin(array)

def maximum(array):
    return np.amax(array)

def minimum_with_nan(array):
    return np.nanmin(array)

def maximum_with_nan(array):
    return np.nanmax(array)

def percentile(array):
    return np.percentile(array)

def percentile_with_nan(array):
    return np.nanpercentile(array)

def quantile(array, q):
    return np.quantile(array, q)

def quantile_with_nan(array, q):
    return np.nanquantile(array, q)

def median(array):
    return np.median(array)

def mean(array):
    return np.mean(array)

def standard_deviation(array):
    return np.std(array)

def variance(array):
    return np.variance(array)

def median_with_nan(array):
    return np.nanmedian(array)

def mean_with_nan(array):
    return np.nanmean(array)

def standard_deviation_with_nan(array):
    return np.nanstd(array)

def variance_with_nan(array):
    return np.nanvar(array)

def geometric_mean(array):
    return stats.gmean(array)

def harmonic_mean(array):
    return stats.hmean(array)

def kurtosis(array):
    return stats.kurtosis(array)

def mode(array):
    return stats.mode(array)

def skew(array):
    return stats.skew(array)

def variation(array):
    return stats.variation(array)

def find_repeats(array):
    return stats.find_repeats(array)

def interquartile_range(array):
    return stats.iqr(array)

def entropy(probabilities, alternative_probabilities=None):
    return stats.entropy(probabilities, alternative_probabilities)

def trimean(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    median = np.median(data)
    return (q1 + 2*median + q3)/4

def interquartile_mean(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    sorted_data = np.sort(data)
    trimmed_data = sorted_data[(sorted_data >= q1) & (sorted_data <= q3)]
    return np.mean(trimmed_data)

def midhinge(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    return np.mean([q1, q3])

def value_range(data):
    max_val = np.max(data)
    min_val = np.min(data)
    return abs(max_val - min_val)

def trimean_absolute_deviation(data):
    trimean = trimean(data)
    numerator = [abs(elem - trimean) for elem in data]
    return sum(numerator)/len(data)

def mean_absolute_deviation(data):
    mean = mean(data)
    numerator = [abs(elem - mean) for elem in data]
    return sum(numerator)/len(data)

def median_absolute_deviation(data):
    median = median(data)
    numerator = [abs(elem - median) for elem in data]
    return sum(numerator)/len(data)

def _get_cdf(dist):
    cdf = CDF(dist)
    return dict(
        zip(
            list(cdf.xs),
            list(cdf.ps)
        )
    )

def _get_prob_values(cdf):
    return list(cdf.values())

def get_inliers_outliers(dist):
    clf = IsolationForest(
        behaviour='new', contamination='auto'
    )
    result = clf.fit_predict(dist.reshape(-1, 1))
    result = pd.Series(result)
    outliers = dist[result[result == -1].index]
    inliers = dist[result[result != -1].index]
    return inliers, outliers
    
def isclose(value_one, value_two,
            max_deviance,
            distance_function=None):
    """
    Here we define the max_deviance parameter.
    This parameter assumes you have a sense of
    the appropriate deviance.
    I suggest looking at some plots of your data,
    when doing exploratory data analysis to figure this out.
    The appropriate deviance is a maximum threshold for what
    should be considered close in your dataset.
    Eventually, the goal is to figure out an automated method
    to figure this out.  However, for now, no such metric
    appears obvious, therefore I leave it to the user to figure out
    what 'closeness' means for your data.
    
    If this measure is hard to come by, or changes often, then this
    is probably not the right metric for you.
    
    Some sense of stability in the scale of your data is expected,
    to make use of this function.
    """
    if not distance_function:
        distance_function = distance.euclidean
    if distance_function(value_one, value_two) < max_deviance:
        return True
    else:
        return False

def get_compare_value(value, cdf,
                      max_deviance,
                      distance_function=None):
    """
    Please see doc string in isclose.
    """
    if value in cdf:
        return value
    for value_two in cdf:
        if np.isclose(value, value_two):
            return value_two
        close_enough = isclose(
            value, value_two, max_deviance,
            distance_function=distance_function
        )
        if close_enough:
            return value_two
    return None

def get_within_boundary(cdf_one, cdf_two, spread,
                        max_deviance,
                        distance_function=None):
    """
    Please see doc string in isclose. 
    """
    within_upper_bound = []
    within_lower_bound = []
    for value in cdf_one:
        other_value = get_compare_value(
            value, cdf_two, max_deviance,
            distance_function=distance_function
        )
        if not other_value:
            within_upper_bound.append(False)
            within_lower_bound.append(False)
        else:
            within_upper_bound.append(
                cdf_two[other_value] < cdf_one[value] + spread
            )
            within_lower_bound.append(
                cdf_two[other_value] > cdf_one[value] - spread
            )
    within_upper_bound = np.array(within_upper_bound)
    within_lower_bound = np.array(within_lower_bound)
    return within_upper_bound & within_lower_bound

def get_inliers(dist):
    dist = np.array(dist)
    inliers, outliers = get_inliers_outliers_if(dist)
    return inliers

def compare_cdf_mean_absolute_deviation(dist_one, dist_two,
                                        max_deviance,
                                        distance_function=None,
                                        remove_outliers=True):
    """
    We assume dist_one and dist_two are of the same size.
    I.E. len(dist_one) == len(dist_two)
    
    Please see doc string for isclose.
    """
    if remove_outliers:
        dist_one = get_inliers(dist_one)
        dist_two = get_inliers(dist_two)
    cdf_one = _get_cdf(dist_one)
    cdf_two = _get_cdf(dist_two)
    mad = mean_absolute_deviation(_get_prob_values(cdf_one))
    within_boundary = get_within_boundary(
        cdf_one, cdf_two, mad,
        max_deviance,
        distance_function=distance_function
    )
    return (within_boundary).sum()/len(dist_one)

def compare_cdf_median_absolute_deviation(dist_one, dist_two,
                                          max_deviance,
                                          distance_function=None,
                                          remove_outliers=True):
    """
    We assume dist_one and dist_two are of the same size.
    I.E. len(dist_one) == len(dist_two)

    Please see doc string for isclose
    """
    if remove_outliers:
        dist_one = get_inliers(dist_one)
        dist_two = get_inliers(dist_two)
    cdf_one = _get_cdf(dist_one)
    cdf_two = _get_cdf(dist_two)
    mad = median_absolute_deviation(_get_prob_values(cdf_one))
    within_boundary = get_within_boundary(
        cdf_one, cdf_two, mad,
        max_deviance,
        distance_function=distance_function
    )
    return (within_boundary).sum()/len(dist_one)

def compare_cdf_trimean_absolute_deviation(dist_one, dist_two,
                                           max_deviance,
                                           distance_function=None,
                                           remove_outliers=True):
    """
    We assume dist_one and dist_two are of the same size.
    I.E. len(dist_one) == len(dist_two)

    Please see doc string for isclose
    """
    if remove_outliers:
        dist_one = get_inliers(dist_one)
        dist_two = get_inliers(dist_two)
    cdf_one = _get_cdf(dist_one)
    cdf_two = _get_cdf(dist_two)
    tad = trimean_absolute_deviation(_get_prob_values(cdf_one))
    within_boundary = get_within_boundary(
        cdf_one, cdf_two, tad,
        max_deviance,
        distance_function=distance_function
    )
    return (within_boundary).sum()/len(dist_one)

def compare_cdf_hard_coded_boundary(dist_one, dist_two,
                                    max_deviance,
                                    distance_function=None,
                                    boundary=0.01,
                                    remove_outliers=True):
    """
    We assume dist_one and dist_two are of the same size.
    I.E. len(dist_one) == len(dist_two)

    Please see doc string for isclose
    """
    if remove_outliers:
        dist_one = get_inliers(dist_one)
        dist_two = get_inliers(dist_two)
    cdf_one = _get_cdf(dist_one)
    cdf_two = _get_cdf(dist_two)
    within_boundary = get_within_boundary(
        cdf_one, cdf_two, boundary,
        max_deviance,
        distance_function=distance_function
    )
    return (within_boundary).sum()/len(dist_one)

# things like this
# trimmed statistics
# investigate here: https://en.wikipedia.org/wiki/Descriptive_statistics
