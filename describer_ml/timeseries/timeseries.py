#from mlxtend.evaluate import permutation_test
from statsmodels.tsa import stattools
from statsmodels.stats import diagnostic
from statsmodels.tsa.arima.model import ARIMA
import warnings
from collections import namedtuple
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from scipy.stats import mstats
warnings.filterwarnings("ignore")

class TimeSeriesMetrics:
    def __init__(self):
        pass
    
    @staticmethod
    def unscaled_mean_bounded_relative_absolute_error(y_true, y_pred):
        """
        Unscaled Mean Bounded Relative Absolute Error
        Formula taken from:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
        @y_true - Y[i]
        @y_pred - F[i]
        """
        numerator = [abs(elem - y_pred[idx]) for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx]) for idx, elem in enumerate(series_one)]
        final_series = [numerator[idx]/(numerator[idx] + denominator[idx])
                        for idx in range(len(denominator))]
        mbrae = np.mean(final_series)
        return mbrae/(1-mbrae)

    @staticmethod
    def mean_bounded_relative_absolute_error(y_true, y_pred):
        """
        Mean Bounded Relative Absolute Error
        Formula taken from:
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5365136/
        @y_true - Y[i]
        @y_pred - F[i]
        """
        numerator = [abs(elem - y_pred[idx]) for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx]) for idx, elem in enumerate(series_one)]
        final_series = [numerator[idx]/(numerator[idx] + denominator[idx])
                        for idx in range(len(denominator))]
        return np.mean(final_series)

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

    @staticmethod
    def mean_relative_absolute_error(y_true, y_pred):
        """
        formula comes from: 
        http://www.spiderfinancial.com/support/documentation/numxl/reference-manual/forecasting-performance/mrae
        """
        numerator = [abs(elem - y_pred[idx])
                     for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx])
                       for idx, elem in enumerate(series_one)]    
        return np.mean([
            numerator[i]/denominator[i] for i in range(len(numerator))])

    @staticmethod
    def median_relative_absolute_error(y_true, y_pred):
        """
        formula comes from: 
        http://www.spiderfinancial.com/support/documentation/numxl/reference-manual/forecasting-performance/mrae
        """
        numerator = [abs(elem - y_pred[idx])
                     for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx])
                       for idx, elem in enumerate(series_one)]    
        return np.median([
            numerator[i]/denominator[i] for i in range(len(numerator))])

    @staticmethod
    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        """
        formula comes from:
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        numerator = [abs(y_pred[idx] - elem) for idx, elem in enumerate(y_true)]
        denominator = [abs(elem) + abs(y_pred[idx]) for idx, elem in enumerate(y_true)]
        denominator = [elem/2 for elem in denominator]
        result = np.mean([numerator[i]/denominator[i] for i in range(len(numerator))])
        return result * 100

    @staticmethod
    def symmetric_median_absolute_percentage_error(y_true, y_pred):
        """
        formula comes from:
        https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
        """
        numerator = [abs(y_pred[idx] - elem) for idx, elem in enumerate(y_true)]
        denominator = [abs(elem) + abs(y_pred[idx]) for idx, elem in enumerate(y_true)]
        denominator = [elem/2 for elem in denominator]
        result = np.median([numerator[i]/denominator[i] for i in range(len(numerator))])
        return result * 100

    @staticmethod
    def mean_absolute_scaled_error(y_true, y_pred):
        """
        formula comes from:
        https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
        """
        numerator = sum([abs(y_pred[idx] - elem)  for idx, elem in enumerate(y_true)])
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = sum([abs(elem - series_two[idx])
                       for idx, elem in enumerate(series_one)])
        coeficient = len(y_true)/(len(y_true)-1)
        return numerator/(coeficient * denominator)

    @staticmethod
    def geometric_mean_relative_absolute_error(y_true, y_pred):
        numerator = [abs(y_pred[idx] - elem)  for idx, elem in enumerate(y_true)]
        series_one = y_true[1:]
        series_two = y_true[:-1]
        denominator = [abs(elem - series_two[idx])
                       for idx, elem in enumerate(series_one)]
        return mstats.gmean([numerator[i]/denominator[i] for i in range(len(numerator))])


class TimeSeriesHypothesisTests:
    def __init__(self):
        pass

    @staticmethod
    def ad_fuller_test(timeseries):
        result = stattools.adfuller(timeseries)
        AdFullerResult = namedtuple('AdFullerResult', 'statistic pvalue')
        return AdFullerResult(result[0], result[1])

    @staticmethod
    def kpss(timeseries):
        result = stattools.kpss(timeseries)
        KPSSResult = namedtuple('KPSSResult', 'statistic pvalue')
        return KPSSResult(result[0], result[1])

    @staticmethod
    def cointegration(timeseries, alt_timeseries):
        result = stattools.coint(timeseries, alt_timeseries)
        CointegrationResult = namedtuple('CointegrationResult', 'statistic pvalue')
        return CointegrationResult(result[0], result[1])

    @staticmethod
    def bds(timeseries):
        result = stattools.bds(timeseries)
        BdsResult = namedtuple('BdsResult', 'statistic pvalue')
        return BdsResult(result[0], result[1])

    @staticmethod
    def q_stat(timeseries):
        autocorrelation_coefs = stattools.acf(timeseries)
        result = stattools.q_stat(autocorrelation_coefs)
        QstatResult = namedtuple('QstatResult', 'statistic pvalue')
        return QstatResult(result[0], result[1])

    def _evaluate_arima_model(X, arima_order):
        # prepare training dataset
        train, test, _, _ = train_test_split(X, np.zeros(X.shape[0]))
        history = list(train)
        # make predictions
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order=arima_order)
            model_fit = model.fit(disp=0)
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])
        # calculate out of sample error
        error = mean_squared_error(test, predictions)
        return error

    # evaluate combinations of p, d and q values for an ARIMA model
    @staticmethod
    def generate_model(timeseries):
        best_score, best_cfg = float("inf"), None
        p_values = [0, 1, 2, 4, 6, 8, 10]
        d_values = range(0, 3)
        q_values = range(0, 3)
        best_order = (1, 0, 0)
        # if no 'best order' then simply go with AR(1)
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p,d,q)
                    try:
                        mse = self._evaluate_arima_model(timeseries, order)
                        if mse < best_score:
                            best_score = mse
                            best_order = order
                    except:
                        continue
        model = ARIMA(timeseries, order=best_order)
        model_result = model.fit()
        return model, model_result

    @staticmethod
    def acorr_breusch_godfrey(timeseries):
        result = diagnostic.acorr_breusch_godfrey(timeseries)
        AcorrBreuschGodfreyResult = namedtuple('BreuschGodfreyResult', 'statistic pvalue')
        return AcorrBreuschGodfreyResult(result[0], result[1])

    @staticmethod
    def het_arch(timeseries):
        result = diagnostic.het_arch(timeseries)
        HetArchResult = namedtuple('HetArchResult', 'statistic pvalue')
        return HetArchResult(result[0], result[1])

    @staticmethod
    def breaks_cumsumolsresid(timeseries):
        result = diagnostic.breaks_cusumolsresid(timeseries)
        BreaksCumSumResult = namedtuple('BreaksCumSumResult', 'statistic pvalue')
        return BreaksCumSumResult(result[0], result[1])

