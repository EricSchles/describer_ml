from describer_ml.numeric.num_stats import trimean

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
