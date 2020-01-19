from describer_ml.numeric.num_stats import (
    trimean, mode, variance, standard_deviation,
    skew, kurtosis, variation, interquartile_range,
    midhinge, entropy, mean_absolute_deviation,
    median_absolute_deviation, trimean_absolute_deviation,
    get_inliers_outliers
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
