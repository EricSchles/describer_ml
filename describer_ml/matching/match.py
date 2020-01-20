from itertools import combinations
from describer_ml.numeric.num_stats import (
    trimean, mode, variance, standard_deviation,
    skew, kurtosis, variation, interquartile_range,
    midhinge, entropy, mean_absolute_deviation,
    median_absolute_deviation, trimean_absolute_deviation,
    compare_cdf_hard_coded_boundary,
    compare_cdf_trimean_absolute_deviation,
    compare_cdf_median_absolute_deviation,
    compare_cdf_mean_absolute_deviation
)

def get_diffs(listing):
    diffs = []
    for elem in listing:
        for other in listing:
            diffs.append(
                abs(elem - other)
            )
    return diffs

def within_tolerance(diffs, max_diff):
    for diff in diffs:
        if diff > max_diff:
            return False
    return True

def get_matches(grouped_df, max_diff):
    matching_columns = []
    for column in grouped_df.columns:
        diffs = get_diffs(grouped_df[column])
        if within_tolerance(diffs, max_diff):
            matching_columns.append(column)
    return matching_columns

def get_multi_matches(df, match_column, max_diffs):
    match_map = {
        "mean": mean_match,
        "median": median_match,
        "trimean": trimean_match,
        "mode": mode_match,
        "standard_deviation": standard_deviation_match,
        "variance": variance_match,
        "skew": skew_match,
        "kurtosis": kurtosis_match,
        "variation": variation_match,
        "interquartile_range": interquartile_range_match,
        "midhinge": midhinge_match,
        "entropy": entropy_match,
        "mean_absolute_deviation": mean_absolute_deviation_match,
        "median_absolute_deviation": median_absolute_deviation_match,
        "trimean_absolute_deviation": trimean_absolute_deviation_match
    }
    matches = {}
    for match_algo in max_diffs:
        condition = max_diffs[matcher]
        matcher = match_map(match_algo)
        matches[match_algo] = matcher(
            df, match_column, condition
        )
    return matches

def get_multi_matching_columns(df, match_column, max_diffs):
    matches = get_multi_matches(df, match_column, max_diffs)
    matches = list(matches.values())
    first_match = set(matches[0])
    matches = [set(elem) for elem in matches[1:]]
    return list(first_match.intersection(*matches))
        
def mean_match(df, match_column, max_diff):
    average_per_class = df.groupby(match_column).mean()
    return get_matches(mean_per_class, max_diff)

def median_match(df, match_column, max_diff):
    median_per_class = df.groupby(match_column).median()
    return get_matches(median_per_class, max_diff)

def trimean_match(df, match_column, max_diff):
    trimean_per_class = df.groupby(match_column).agg(trimean)
    return get_matches(trimean_per_class, max_diff)

def mode_match(df, match_column, max_diff):
    mode_per_class = df.groupby(match_column).agg(mode)
    return get_matches(mode_per_class, max_diff)

def variance_match(df, match_column, max_diff):
    variance_per_class = df.groupby(match_column).agg(variance)
    return get_matches(variance_per_class, max_diff)

def standard_deviation_match(df, match_column, max_diff):
    standard_deviation_per_class = df.groupby(match_column).agg(standard_deviation)
    return get_matches(standard_deviation_per_class, max_diff)

def skew_match(df, match_column, max_diff):
    skew_per_class = df.groupby(match_column).agg(skew)
    return get_matches(skew_per_class, max_diff)

def kurtosis_match(df, match_column, max_diff):
    kurtosis_per_class = df.groupby(match_column).agg(kurtosis)
    return get_matches(kurtosis_per_class, max_diff)

def variation_match(df, match_column, max_diff):
    variation_per_class = df.groupby(match_column).agg(variation)
    return get_matches(variation_per_class, max_diff)

def interquartile_range_match(df, match_column, max_diff):
    interquartile_range_per_class = df.groupby(match_column).agg(interquartile_range)
    return get_matches(interquartile_range_per_class, max_diff)

def midhinge_match(df, match_column, max_diff):
    midhinge_per_class = df.groupby(match_column).agg(midhinge)
    return get_matches(midhinge_per_class, max_diff)

def entropy_match(df, match_column, max_diff):
    entropy_per_class = df.groupby(match_column).agg(entropy)
    return get_matches(entropy_per_class, max_diff)

def mean_absolute_deviation_match(df, match_column, max_diff):
    mean_absolute_deviation_per_class = df.groupby(match_column).agg(mean_absolute_deviation)
    return get_matches(mean_absolute_deviation_per_class, max_diff)

def median_absolute_deviation_match(df, match_column, max_diff):
    median_absolute_deviation_per_class = df.groupby(match_column).agg(median_absolute_deviation)
    return get_matches(median_absolute_deviation_per_class, max_diff)

def trimean_absolute_deviation_match(df, match_column, max_diff):
    trimean_absolute_deviation_per_class = df.groupby(match_column).agg(trimean_absolute_deviation)
    return get_matches(trimean_absolute_deviation_per_class, max_diff)

def get_groups_classes(df, match_column):
    columns = df.columns.tolist()
    columns.remove(match_column)
    groups = [tmp_df for _, tmp_df in df.groupby(match_column)]
    classes = list(df[match_column].unique())
    return groups, classes

def get_percent_matches(percent_matches, min_percent_match):
    matches = []
    for percent_match in percent_matches:
        if percent_match[0] > min_percent_match:
            matches.append((
                percent_match[1],
                percent_match[2]
            ))
    return matches

def distribution_match_cdf_hard_coded(df, match_column, max_deviances,
                                      min_percent_match=0.9,
                                      distance_function=None,
                                      boundary=0.01, remove_outliers=False):
    groups, classes = get_groups_classes(
        df, match_column
    )
    class_combinations = list(combinations(classes, 2))
    percent_matches_per_class_per_column = []
    for index, combination in enumerate(combinations(groups, 2)):
        first = combination[0]
        second = combination[1]
        for column in columns:
            max_deviance = max_deviances[column]
            percent_match = compare_cdf_hard_coded_boundary(
                first[column],
                second[column],
                max_deviance,
                distance_function=distance_function,
                boundary=boundary,
                remove_outliers=remove_outliers
            )
            percent_matches_per_class_per_column.append(
                percent_match,
                class_combinations[index],
                column
            )    
    return get_percent_matches(
        percent_matches_per_class_per_column,
        min_percent_match
    )

def distribution_match_cdf_mean_absolute_deviation(df,
                                                   match_column,
                                                   max_deviances,
                                                   min_percent_match=0.9,
                                                   distance_function=None,
                                                   remove_outliers=False):
    groups, classes = get_groups_classes(
        df, match_column
    )
    class_combinations = list(combinations(classes, 2))
    percent_matches_per_class_per_column = []
    for index, combination in enumerate(combinations(groups, 2)):
        first = combination[0]
        second = combination[1]
        for column in columns:
            max_deviance = max_deviances[column]
            percent_match = compare_cdf_mean_absolute_deviation(
                first[column],
                second[column],
                max_deviance,
                distance_function=distance_function,
                remove_outliers=remove_outliers
            )
            percent_matches_per_class_per_column.append(
                percent_match,
                class_combinations[index],
                column
            )
    return get_percent_matches(
        percent_matches_per_class_per_column,
        min_percent_match
    )

def distribution_match_cdf_median_absolute_deviation(df,
                                                     match_column,
                                                     max_deviances,
                                                     min_percent_match=0.9,
                                                     distance_function=None,
                                                     remove_outliers=False):
    groups, classes = get_groups_classes(
        df, match_column
    )
    class_combinations = list(combinations(classes, 2))
    percent_matches_per_class_per_column = []
    for index, combination in enumerate(combinations(groups, 2)):
        first = combination[0]
        second = combination[1]
        for column in columns:
            max_deviance = max_deviances[column]
            percent_match = compare_cdf_median_absolute_deviation(
                first[column],
                second[column],
                max_deviance,
                distance_function=distance_function,
                remove_outliers=remove_outliers
            )
            percent_matches_per_class_per_column.append(
                percent_match,
                class_combinations[index],
                column
            )
    return get_percent_matches(
        percent_matches_per_class_per_column,
        min_percent_match
    )

def distribution_match_cdf_trimean_absolute_deviation(df,
                                                      match_column,
                                                      max_deviances,
                                                      min_percent_match=0.9,
                                                      distance_function=None,
                                                      remove_outliers=False):
    groups, classes = get_groups_classes(
        df, match_column
    )
    class_combinations = list(combinations(classes, 2))
    percent_matches_per_class_per_column = []
    for index, combination in enumerate(combinations(groups, 2)):
        first = combination[0]
        second = combination[1]
        for column in columns:
            max_deviance = max_deviances[column]
            percent_match = compare_cdf_trimean_absolute_deviation(
                first[column],
                second[column],
                max_deviance,
                distance_function=distance_function,
                remove_outliers=remove_outliers
            )
            percent_matches_per_class_per_column.append(
                percent_match,
                class_combinations[index],
                column
            )
    return get_percent_matches(
        percent_matches_per_class_per_column,
        min_percent_match
    )
