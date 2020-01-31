'''
Determine datetime of closing and reopening of super markets
'''

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk

state = 'nc'

# import data
fname = 'super_market_nc'
df = pd.read_csv('data/{}.csv'.format(fname), encoding = "ISO-8859-1")
df = df.dropna(how='all')

# convert appropriate columns to time
df.close = pd.to_datetime(df.close,dayfirst=True)
df.open = pd.to_datetime(df.open,dayfirst=True)

if state == 'FL':
    time_record = datetime(2018,10,9,0,0)
    time_end = datetime(2019,1,1,0,0)
else:
    time_record = datetime(2018,9,8,0,0)
    time_end = datetime(2018,10,9,13,0)

time_difference = (time_end - time_record)
time_steps = (time_difference.days + 1) * 24

# init dictionary for store outages
stores_over_time = []

# what are the unique store ids
unique_ids = np.unique(df.id)

# loop through the times
for i in range(time_steps):
    # create a list of operational stores
    stores_operational = []
    # loop through the ids
    for id in unique_ids:
        df_store = df.loc[df.id == id].copy()
        ######
        ### this is for time before I start scraping
        ######
        # is the time between the closed and open dates?
        closed = (time_record > df_store.close).values[0] and (time_record < df_store.open).values[0]
        if not closed:
            stores_operational.append(id)
        # if time_record > datetime(2018,9,14,7,0):
        #     import code
        #     code.interact(local=locals())
    # add the dictionary
    stores_dict = {'datetime':time_record, 'operational_ids':stores_operational}
    # append to the list
    stores_over_time.append(stores_dict)
    # iterate time
    time_record += timedelta(hours=1)

# save list of operational stations over time
with open('data/{}.pk'.format(fname),'wb') as fp:
    pk.dump(stores_over_time, fp)
