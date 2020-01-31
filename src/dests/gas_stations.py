'''
Determine datetime of closing and reopening
'''

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk


state = 'nc'
# import data
fname = 'gas_station_{}'.format(state)
df = pd.read_csv('data/{}.csv'.format(fname))

# convert appropriate columns to time
df.time = pd.to_datetime(df.time, dayfirst = True)
df.loc[df.has_gas_updatedate == 'FALSE', 'has_gas_updatedate'] = None
df.loc[df.has_gas_updatedate == 'TRUE', 'has_gas_updatedate'] = None
df.loc[df.has_gas_updatedate == True, 'has_gas_updatedate'] = None
df.has_gas_updatedate = pd.to_datetime(df.has_gas_updatedate)
# which stations have false power update timestamps
df.loc[df.has_power_updatedate == 'FALSE', 'has_power_updatedate'] = None
df.loc[df.has_power_updatedate == 'TRUE', 'has_power_updatedate'] = None
df.loc[df.has_power_updatedate == True, 'has_power_updatedate'] = None
df.has_power_updatedate = pd.to_datetime(df.has_power_updatedate)

# update the times to account for UTC = EST + 4
df.has_gas_updatedate = df.has_gas_updatedate - timedelta(hours=4)
df.has_power_updatedate = df.has_power_updatedate - timedelta(hours=4)


if state == 'FL':
    time_record = datetime(2018,10,9,0,0)
    time_end = datetime(2019,1,1,0,0)
    time_scraping_started = datetime(2018,10,9,12,0)
    # consider the stations which I have data at the last timestamp for
    ids_used = np.unique(df.id)#[df.time==datetime(2019,1,1,0,0)])
else:
    time_record = datetime(2018,9,8,0,0)
    time_end = datetime(2018,10,9,13,0)
    time_scraping_started = datetime(2018,9,19,10,0)
    # consider the stations which I have data at the last timestamp for
    ids_used = np.unique(df.id[df.time==datetime(2018,10,9,10,0)])

time_difference = (time_end - time_record)
time_steps = (time_difference.days + 1) * 24

# init dictionary for station outages
stations_over_time = []

# what are the unique station ids
unique_ids = np.unique(df.id)

# loop through the times
for i in range(time_steps):
    # create a list of operational stations
    stations_operational = []
    # loop through the ids
    for station_id in ids_used:
        ######
        ### this is for time before I start scraping
        ######
        df_station = df.loc[df.id == station_id].copy()
        if time_record < time_scraping_started:
            # are any of the gas or power updates recorded before now?
            prior_gas = time_record > df_station['has_gas_updatedate']
            prior_power = time_record > df_station['has_power_updatedate']
            # check if the station has power
            if prior_power.any():
                has_power = df_station['has_power'].values[0]
            else:
                has_power = True
            # check if the station has gas
            if prior_gas.any():
                has_gas = df_station['has_gas'].values[0]
            else:
                has_gas = True
        else:
            ### Now I'm looking at times that I am recording within, so take the closest time.
            # what is the index of the timestamp that is nearest
            # print(str(i) + ': ' + str(station_id))
            closest_idx = ((df_station.time-time_record)/timedelta(minutes=1)).abs().values.argsort()[:1]
            # check if I have gas and power
            has_gas = df_station.has_gas.iloc[closest_idx].values[0]
            has_power = df_station.has_power.iloc[closest_idx].values[0]
        # now check both and if either are out, record the station_id
        if has_power and has_gas:
            stations_operational.append(station_id)
    # add the dictionary
    stations_dict = {'datetime':time_record, 'operational_ids':stations_operational}
    # append to the list
    stations_over_time.append(stations_dict)
    # iterate time
    time_record += timedelta(hours=1)


# save list of operational stations over time
with open('data/{}.pk'.format(fname),'wb') as fp:
    pk.dump(stations_over_time, fp)
