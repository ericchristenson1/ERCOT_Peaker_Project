"""Feature engineering utilities for ERCOT peaker project.

This module provides a function to combine hourly load actuals with hourly
weather and to compute commonly used features: hour, day of week, weekend,
season, US federal holiday flag, lagging loads, temperature interactions,
cooling/heating degree hours, and weather ramps.

Usage:
    from feature_engineering import merge_load_and_weather
    df = merge_load_and_weather(
        'clean_data/hourly_load_act.csv',
        'clean_data/weather_hourly.csv',
        save_path='clean_data/merged_features.csv'
    )
"""
from typing import List, Optional
import pandas as pd
import numpy as np

try:
    from pandas.tseries.holiday import USFederalHolidayCalendar
except Exception:
    USFederalHolidayCalendar = None


def _infer_datetime_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if not candidates:
        return None
    return candidates[0]


def _infer_temp_col(df: pd.DataFrame) -> Optional[str]:
    for key in ['temp', 'temperature', 'air_temp', 't2m']:
        for c in df.columns:
            if key in c.lower():
                return c
    # fallback: numeric columns with typical names
    return None


def _infer_load_col(df: pd.DataFrame) -> Optional[str]:
    for key in ['load', 'mw', 'demand']:
        for c in df.columns:
            if key in c.lower():
                return c
    # fallback: first numeric column
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric[0] if numeric else None


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return 'winter'
    if month in (3, 4, 5):
        return 'spring'
    if month in (6, 7, 8):
        return 'summer'
    return 'fall'


def merge_load_and_weather(
    load_csv: str,
    weather_csv: str,
    datetime_col: Optional[str] = None,
    tz: Optional[str] = None,
    lag_hours: List[int] = [1, 24, 168],
    base_temp_f: float = 65.0,
    solar_csv: Optional[str] = None,
    wind_csv: Optional[str] = None,
    lmp_csv: Optional[str] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """Read load and weather CSVs, merge on hourly datetime, and add features.

    Parameters
    - load_csv: path to hourly load actuals CSV
    - weather_csv: path to hourly weather CSV
    - datetime_col: optional datetime column name if not auto-detected
    - tz: timezone to localize the parsed datetimes (if provided)
    - lag_hours: list of integer hour lags to create for the load column
    - base_temp_f: base temperature (F) for degree hours calculations
    - save_path: if provided, save the resulting DataFrame to this CSV

    Returns
    - pandas DataFrame with merged data and added features
    Note: optional `solar_csv`, `wind_csv`, and `lmp_csv` are merged (as-of) on the
    same datetime column if provided. They are merged after weather so their numeric
    columns are available for ramp calculations.
    """
    # Read files
    load = pd.read_csv(load_csv)
    weather = pd.read_csv(weather_csv)
    solar = pd.read_csv(solar_csv) if solar_csv else None
    wind = pd.read_csv(wind_csv) if wind_csv else None
    lmp = pd.read_csv(lmp_csv) if lmp_csv else None

    # Infer datetime column if not provided
    dt_col = datetime_col or _infer_datetime_col(load) or _infer_datetime_col(weather)
    if dt_col is None:
        raise ValueError('Could not infer datetime column in inputs; please pass datetime_col')

    # Parse datetimes
    load[dt_col] = pd.to_datetime(load[dt_col])
    weather[dt_col] = pd.to_datetime(weather[dt_col])
    if tz:
        load[dt_col] = load[dt_col].dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
        weather[dt_col] = weather[dt_col].dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')

    # Standardize column name
    load = load.rename(columns={dt_col: 'datetime'})
    weather = weather.rename(columns={dt_col: 'datetime'})
    if solar is not None:
        if dt_col in solar.columns:
            solar = solar.rename(columns={dt_col: 'datetime'})
        else:
            # try infer
            solar = solar.rename(columns={_infer_datetime_col(solar): 'datetime'}) if _infer_datetime_col(solar) else solar
        solar['datetime'] = pd.to_datetime(solar['datetime'])
        if tz:
            solar['datetime'] = solar['datetime'].dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
    if wind is not None:
        if dt_col in wind.columns:
            wind = wind.rename(columns={dt_col: 'datetime'})
        else:
            wind = wind.rename(columns={_infer_datetime_col(wind): 'datetime'}) if _infer_datetime_col(wind) else wind
        wind['datetime'] = pd.to_datetime(wind['datetime'])
        if tz:
            wind['datetime'] = wind['datetime'].dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')
    if lmp is not None:
        if dt_col in lmp.columns:
            lmp = lmp.rename(columns={dt_col: 'datetime'})
        else:
            lmp = lmp.rename(columns={_infer_datetime_col(lmp): 'datetime'}) if _infer_datetime_col(lmp) else lmp
        lmp['datetime'] = pd.to_datetime(lmp['datetime'])
        if tz:
            lmp['datetime'] = lmp['datetime'].dt.tz_localize(tz, ambiguous='infer', nonexistent='shift_forward')

    # Merge on datetime (hourly) -- inner join to keep aligned hours
    df = pd.merge_asof(
        load.sort_values('datetime'),
        weather.sort_values('datetime'),
        on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta('30m'),
    )

    # Merge optional datasets (as-of) to align to the same hourly datetime
    if solar is not None:
        df = pd.merge_asof(
            df.sort_values('datetime'),
            solar.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta('30m'),
            suffixes=(None, '_solar'),
        )
    if wind is not None:
        df = pd.merge_asof(
            df.sort_values('datetime'),
            wind.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta('30m'),
            suffixes=(None, '_wind'),
        )
    if lmp is not None:
        df = pd.merge_asof(
            df.sort_values('datetime'),
            lmp.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta('30m'),
            suffixes=(None, '_lmp'),
        )

    # Ensure sorted by time
    df = df.sort_values('datetime').reset_index(drop=True)

    # Basic time features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'] >= 5
    df['season'] = df['datetime'].dt.month.apply(_season_from_month)

    # Holidays (US Federal) if available
    if USFederalHolidayCalendar is not None:
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df['datetime'].min().date(), end=df['datetime'].max().date())
        df['is_holiday'] = df['datetime'].dt.normalize().isin(pd.to_datetime(holidays))
    else:
        df['is_holiday'] = False

    # Identify load and temperature columns
    load_col = _infer_load_col(load)
    temp_col = _infer_temp_col(weather)

    if load_col is None:
        raise ValueError('Could not infer load column from load CSV')

    # Rename inferred columns to standard names for downstream features
    if load_col in df.columns:
        df = df.rename(columns={load_col: 'load'})
    else:
        # if not present after merge, try original load DataFrame
        if load_col in load.columns:
            df['load'] = load[load_col].values

    if temp_col and temp_col in df.columns:
        df = df.rename(columns={temp_col: 'temp'})
    else:
        # try to find any numeric column that looks like temperature
        if temp_col is None:
            for c in weather.columns:
                if weather[c].dtype.kind in 'fi' and 'wind' not in c.lower() and 'humidity' not in c.lower():
                    # assign first numeric as temp fallback
                    df = df.rename(columns={c: 'temp'})
                    temp_col = c
                    break

    # Create lagging load features
    for lag in lag_hours:
        df[f'load_lag_{lag}h'] = df['load'].shift(lag)

    # Temperature-based features
    if 'temp' in df.columns:
        # temperature interactions
        df['temp_sq'] = df['temp'] ** 2
        df['temp_x_hour'] = df['temp'] * df['hour']

        # cooling and heating degree hours (assuming Fahrenheit)
        df['cooling_degree_hours'] = (df['temp'] - base_temp_f).clip(lower=0)
        df['heating_degree_hours'] = (base_temp_f - df['temp']).clip(lower=0)

        # weather ramps: 1h and 3h absolute changes
        df['temp_ramp_1h'] = df['temp'].diff().abs()
        df['temp_ramp_3h'] = df['temp'].diff(3).abs()
        # rolling variability (std) as another ramp measure
        df['temp_rollstd_6h'] = df['temp'].rolling(window=6, min_periods=1).std()

    # humidity interactions if present
    humidity_col = None
    for c in df.columns:
        if 'humid' in c.lower():
            humidity_col = c
            break
    if humidity_col:
        df = df.rename(columns={humidity_col: 'humidity'})
        df['temp_x_humidity'] = df['temp'] * df['humidity'] if 'temp' in df.columns else None
        df['humidity_ramp_1h'] = df['humidity'].diff().abs()

    # Weather ramps for generic numeric weather vars
    weather_numeric_cols = [c for c in weather.columns if weather[c].dtype.kind in 'fi']
    for c in weather_numeric_cols:
        # skip temp and humidity since handled
        if c == temp_col or 'humid' in c.lower():
            continue
        colname = c
        if colname in df.columns:
            df[f'{colname}_ramp_1h'] = df[colname].diff().abs()
            df[f'{colname}_ramp_3h'] = df[colname].diff(3).abs()

    # Flag rows with missing lags (start of series)
    df['has_full_lags'] = df[[f'load_lag_{lag}h' for lag in lag_hours]].notnull().all(axis=1)

    # Optionally save
    if save_path:
        df.to_csv(save_path, index=False)

    return df


if __name__ == '__main__':
    # Simple CLI for ad-hoc runs
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('load_csv')
    p.add_argument('weather_csv')
    p.add_argument('--datetime-col', default=None, help='datetime column name if different between files')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    # Explicit CSVs to merge (hard-coded)
    solar_csv_path = 'clean_data/ercot_solar_actuals_allzones_2023_2024.csv'
    wind_csv_path = 'clean_data/ercot_wind_actuals_hourly_2023_2024.csv'
    lmp_csv_path = 'clean_data/LMP_2023_2024_Hubs.csv'

    df = merge_load_and_weather(
        args.load_csv,
        args.weather_csv,
        datetime_col=args.datetime_col,
        solar_csv=solar_csv_path,
        wind_csv=wind_csv_path,
        lmp_csv=lmp_csv_path,
        save_path=args.out,
    )
    print('Merged rows:', len(df))
