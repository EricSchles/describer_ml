from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy.signal import correlate
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def is_significant(cross_correlation):
    num_obs = len(cross_correlation)
    middle_index = len(cross_correlation)//2
    cross_correlation = pd.Series(cross_correlation)
    cross_correlation.index = range(len(cross_correlation))
    max_index = cross_correlation[
        cross_correlation == cross_correlation.max()
    ].index[0]
    lag = abs(middle_index - max_index)
    return cross_correlation.max() > (2/np.sqrt(num_obs - lag))

def cross_correlation_plot(feature_one, feature_two):
    feature_one = feature_one - feature_one.mean()
    feature_two = feature_two - feature_two.mean()
    cross_correlation = correlate(feature_one, feature_two)
    cross_correlation /= (len(feature_one) * feature_one.std() * feature_two.std())
    plt.xcorr(feature_one, feature_two, maxlags=5)
    absolute_cross_correlation = abs(cross_correlation)
    print("Max cross correlation", cross_correlation.max())
    print("Average cross correlation", cross_correlation[:20].mean())
    if is_significant(cross_correlation):
        statistically_significant = True
        print("and is statistically significant")
    else:
        statistically_significant = False
        print("and is not statistically significant")
    print()
    plt.show()
    cross_correlation = pd.Series(cross_correlation)
    cross_correlation.index = range(len(cross_correlation))
    return cross_correlation, statistically_significant

def compare_timeseries(feature_one, feature_two):
    cross_correlation, statistically_significant = cross_correlation_plot(
        feature_one, feature_two
    )

def smooth_feature(feature):
    feature_smoother = ExponentialSmoothing(
        feature,
        trend="add"
    ).fit(use_boxcox=True)
    smoothed_feature = feature_smoother.predict(start=0, end=len(feature)-1)
    smoothed_feature.fillna(0, inplace=True)
    return smoothed_feature

def check_smoothed_feature(smoothed_feature):
    zero_count = (smoothed_feature == 0).astype(int).sum(axis=0)
    return (zero_count == 0) and np.isfinite(smoothed_feature).all()

def analyze_cross_correlation_timeseries(df, col_one, col_two, time_column, significance_threshold=0.05, zero_percent_threshold=0.05):
    series_one = df[col_one].copy()
    series_two = df[col_two].copy()
    series_one.index = df[time_column].copy()
    series_two.index = df[time_column].copy()
    series_one = series_one.dropna()
    series_two = series_two.dropna()
    
    if breaks_cusumolsresid(series_one)[1] > significance_threshold:
        print("cumulative sum test failed for feature")
    if breaks_cusumolsresid(series_two)[1] > significance_threshold:
        print("cumulative sum test failed for display")

    # no serial correlation
    if adfuller(series_one)[1] < significance_threshold and adfuller(series_two)[1] < significance_threshold:
        compare_timeseries(series_one, series_two)
        cross_correlated += 1

    # serial correlation in series_one
    if adfuller(series_one)[1] > significance_threshold and adfuller(series_two)[1] < significance_threshold:
        try:
            smoothed_series_one = smooth_feature(series_one)
            if np.isfinite(smoothed_series_one).all() and (smoothed_series_one.iloc[0] != smoothed_series_one).all():
                compare_timeseries(smoothed_series_one, series_two)
        except ValueError:
            zero_percent = (series_one == 0).astype(int).sum(axis=0)/len(series_one)
            if zero_percent < zero_percent_threshold:
                series_one = series_one.replace(to_replace=0, method='ffill')
                smoothed_series_one = smooth_feature(series_one)
                if check_smoothed_feature(smoothed_series_one):
                    compare_timeseries(smoothed_series_one, series_two)
    # serial correlation in series_two
    if adfuller(series_one)[1] < significance_threshold and adfuller(series_two)[1] > significance_threshold:
        try:
            smoothed_series_two = smooth_feature(series_two)
            if np.isfinite(smoothed_feature).all() and (smoothed_feature.iloc[0] != smoothed_feature).all():
                compare_timeseries(series_one, smoothed_series_two)
        except ValueError:
            zero_percent = (series_two == 0).astype(int).sum(axis=0)/len(series_two)
            if zero_percent < zero_percent_threshold:
                series_two = series_two.replace(to_replace=0, method='ffill')
                smoothed_series_two = smooth_feature(series_two)
                if check_smoothed_feature(smoothed_series_two):
                    compare_timeseries(feature, smoothed_series_two)
    
    # serial correlation in both therefore use cointegration
    if adfuller(series_one)[1] > significance_threshold and adfuller(series_two)[1] > significance_threshold:
        cointegration_results = coint(series_one, series_two)[1]
        if cointegration_results < significance_threshold:
            print(f"""
            The t-statistic of the unit-root test {cointegration_results[0],
            The pvalue {cointegration_results[1]} is less than signifiance threshold of {significance_threshold},
            So we reject the null hypothesis.  And therefore, we believe there is cointegration (a relationship)
            between the two series.
            """)
        else:
            print(f"""
            The t-statistic of the unit-root test {cointegration_results[0],
            The pvalue {cointegration_results[1]} is greater than signifiance threshold of {significance_threshold},
            So we fail to reject the null hypothesis.  And therefore, we believe there is no relation between the series.
            """)

        
            
