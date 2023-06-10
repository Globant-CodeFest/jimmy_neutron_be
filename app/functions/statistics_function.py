"""
given a dataset of candlesticks enlarge it by adding staticical features
"""
import pandas as pd
import numpy as np
import fastapi as fa
import yfinance as yf
import datetime as dt


def add_volatility_data(df):
    """
    add volatility data to the dataset
    """
    df['volatility'] = df['high'] - df['low']
    df['volatility_change'] = df['volatility'].pct_change()

    return df


def add_jensens_alpha(df):
    """
    add jensen's alpha to the dataset
    """
    df[str(windows) + "|" + 'jensens_alpha'] = df['close'].pct_change() - df['risk_free_rate']
    return df


def add_calculate_moving_averages(df, window=30):
    """
    add moving averages to the dataset
    """
    df[str(windows) + "|" + 'EMA'] = df['close'].rolling(window).mean()
    return df


def add_rolling_std(df, window=30):
    """
    add rolling standard deviation to the dataset
    """
    df[str(windows) + "|" + 'rolling_std'] = df['close'].rolling(window).std()
    return df


def add_rolling_skew(df, window=30):
    """
    add rolling skew to the dataset
    """
    df[str(windows) + "|" + 'rolling_skew'] = df['close'].rolling(window).skew()
    return df


def add_rolling_kurtosis(df, window=30):
    """
    add rolling kurtosis to the dataset
    """
    df[str(windows) + "|" + 'rolling_kurtosis'] = df['close'].rolling(window).kurtosis()
    return df


def add_rolling_sharpe(df, window=30):
    """
    add rolling sharpe to the dataset
    """
    df[str(windows) + "|" + 'rolling_sharpe'] = df['close'].rolling(window).sharpe()
    return df


def add_rolling_sortino(df, window=30):
    """
    add rolling sortino to the dataset
    """
    df[str(windows) + "|" + 'rolling_sortino'] = df['close'].rolling(window).sortino()
    return df


def add_rolling_jensens_alpha(df, window=30):
    """
    add rolling jensen's alpha to the dataset
    """
    df[str(windows) + "|" + 'rolling_jensens_alpha'] = df['close'].rolling(window).jensens_alpha()
    return df


def add_rolling_beta(df, window=30):
    """
    add rolling beta to the dataset
    """
    df[str(windows) + "|" + 'rolling_beta'] = df['close'].rolling(window).beta()
    return df


def add_rolling_r_squared(df, window=30):
    """
    add rolling r squared to the dataset
    """
    df[str(windows) + "|" + 'rolling_r_squared'] = df['close'].rolling(window).r_squared()
    return df


def add_rolling_alpha(df, window=30):
    """
    add rolling alpha to the dataset
    """
    df[str(windows) + "|" + 'rolling_alpha'] = df['close'].rolling(window).alpha()
    return df


def add_rolling_volatility(df, window=30):
    """
    add rolling volatility to the dataset
    """
    df[str(windows) + "|" + 'rolling_volatility'] = df['close'].rolling(window).volatility()
    return df


def add_rolling_volatility_change(df, window=30):
    """
    add rolling volatility change to the dataset
    """
    df[str(windows) + "|" + 'rolling_volatility_change'] = df['close'].rolling(window).volatility_change()
    return df


def add_rolling_volatility_skew(df, window=30):
    """
    add rolling volatility skew to the dataset
    """
    df[str(windows) + "|" + 'rolling_volatility_skew'] = df['close'].rolling(window).volatility_skew()
    return df


def add_rolling_volatility_kurtosis(df, window=30):
    """
    add rolling volatility kurtosis to the dataset
    """
    df[str(windows) + "|" + 'rolling_volatility_kurtosis'] = df['close'].rolling(window).volatility_kurtosis()
    return df


def add_rolling_volatility_sharpe(df, window=30):
    """
    add rolling volatility sharpe to the dataset
    """
    df[str(windows) + "|" + 'rolling_volatility_sharpe'] = df['close'].rolling(window).volatility_sharpe()
    return df


def add_rolling_volatility_sortino(df, window=30):
    """
    add rolling volatility sortino to the dataset
    """
    df[str(windows) + "|" + 'rolling_volatility_sortino'] = df['close'].rolling(window).volatility_sortino()
    return df


def add_traynor_ratio(df):
    """
    add traynor ratio to the dataset
    """

    df[str(windows) + "|" + 'traynor_ratio'] = df['close'].rolling(window).traynor_ratio()
    return df


def compute_parabolic_sar(df, window=30, acceleration=0.02, maximum=0.2):
    """
    create a python function to calculate Parabolic SAR  from a ticker symbol using pandas and numpy
    results will be added to the dataframe"""
    df[str(windows) + "|" + 'parabolic_sar'] = df['close'].rolling(window).parabolic_sar(acceleration=acceleration,
                                                                                         maximum=maximum)
    return df


def compute_bollinger_bands(df, window=30, window_dev=2):
    """
    create a python function to calculate Bollinger Bands from a ticker symbol using pandas and numpy
    results will be added to the dataframe"""
    df[str(windows) + "|" + 'bollinger_bands'] = df['close'].rolling(window).bollinger_bands(window_dev=window_dev)
    return df


def compute_keltner_channel(df, window=30, window_atr=10, window_dev=2):
    """
    create a python function to calculate Keltner Channel from a ticker symbol using pandas and numpy
    results will be added to the dataframe"""
    df[str(windows) + "|" + 'keltner_channel'] = df['close'].rolling(window).keltner_channel(window_atr=window_atr,
                                                                                             window_dev=window_dev)
    return df


def compute_donchian_channel(df, window=30):
    """
    create a python function to calculate Donchian Channel from a ticker symbol using pandas and numpy
    results will be added to the dataframe"""
    df[str(windows) + "|" + 'donchian_channel'] = df['close'].rolling(window).donchian_channel()
    return df


def begin_end_dates_from_a_window(window, current_date=datetime.now()):
    """
    create a python function to calculate begin and end dates from a window
    """
    begin_date = current_date - timedelta(days=window)
    end_date = current_date
    return begin_date, end_date


def add_bollinger_bands(df, window=30, num_std=2):
    rolling_mean = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    df['Upper Band'] = upper_band
    df['Lower Band'] = lower_band
    return df


def add_stochastic_oscillator(df, window=30):
    high_n = df['high'].rolling(window).max()
    low_n = df['low'].rolling(window).min()
    k_percent = 100 * (df['close'] - low_n) / (high_n - low_n)
    df['%K'] = k_percent
    df['%D'] = k_percent.rolling(window=3).mean()
    return df


def add_average_true_range(df, period=14):
    df['TR'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df


def add_relative_strength(df, benchmark):
    benchmark_pct_change = benchmark.pct_change()
    stock_pct_change = df['close'].pct_change()
    relative_strength = stock_pct_change - benchmark_pct_change
    return relative_strength


def add_on_balance_volume(df):
    df['OBV'] = np.where(df['close'].diff() > 0, df['volume'],
                         np.where(df['close'].diff() < 0, -df['volume'], 0)).cumsum()
    return df


def parabolic_sar(df, acceleration_factor=0.02, max_acceleration_factor=0.2):
    # initialize values
    af = acceleration_factor
    max_af = max_acceleration_factor
    ep = df['low'][0]
    sar = df['high'][0]
    long_position = True

    # loop through data
    for i in range(1, len(df)):
        # determine current SAR value
        if long_position:
            sar = sar + af * (ep - sar)
            sar = max(sar, df['low'][i - 1], df['low'][i])
        else:
            sar = sar - af * (sar - ep)
            sar = min(sar, df['high'][i - 1], df['high'][i])

        # determine new extreme price
        if long_position:
            ep = max(ep, df['high'][i])
        else:
            ep = min(ep, df['low'][i])

        # check for reversal
        if long_position:
            if df['low'][i] < sar:
                sar = ep
                long_position = False
                af = acceleration_factor
        else:
            if df['high'][i] > sar:
                sar = ep
                long_position = True
                af = acceleration_factor

        # check for acceleration factor adjustment
        if long_position:
            if df['high'][i] > ep:
                af = min(af + acceleration_factor, max_acceleration_factor)
        else:
            if df['low'][i] < ep:
                af = min(af + acceleration_factor, max_acceleration_factor)

        # store SAR value in DataFrame
        df.loc[i, 'sar'] = sar

    return df

