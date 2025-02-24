#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for downloading and processing electricity load forecast data from ENTSO-E.

Created on: Mar 12, 2022
Updated for GitHub release.
"""

import os
import time
import pandas as pd
from datetime import datetime
from entsoe import EntsoePandasClient

# Set API key from environment variable for security
token = os.getenv("ENTSOE_API_KEY")
if not token:
    raise ValueError("API key not found. Set 'ENTSOE_API_KEY' as an environment variable.")

# Initialize client
client = EntsoePandasClient(api_key=token)

# Define parameters
COUNTRY_CODE = 'UK'
START_DATE = '2016-06-01'
END_DATE = '2021-05-31'

dates = pd.date_range(START_DATE, END_DATE, tz='UTC')
uk_load_forecast = pd.DataFrame(columns=['Date', 'Forecasted Load', 'Actual Load'])

# Fetch data
for i in range(len(dates) - 1):
    start = dates[i]
    end = dates[i + 1]
    print(f"Fetching data for: {start} - {end}")
    
    try:
        load = client.query_load_and_forecast(COUNTRY_CODE, start=start, end=end)
        load['Date'] = load.index
        load = load[['Date', 'Forecasted Load', 'Actual Load']]
        uk_load_forecast = pd.concat([uk_load_forecast, load], ignore_index=True)
        time.sleep(3)  # Avoid API rate limits
    except Exception as e:
        print(f"Error fetching data for {start}: {e}")
        continue

# Data processing function
def change_date_format(df):
    """Reformat date column."""
    df.columns = ['Date', 'Forecasted Load', 'Actual Load']
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    return df.dropna()

# Process the entire dataset
uk_load_forecast = change_date_format(uk_load_forecast)

# Save results
output_file = "UKLoadForecast.csv"
uk_load_forecast.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
