'''
Populate the database for the nearest proximity throughout time
'''

import pickle as pk
import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy.engine import create_engine
from datetime import datetime, timedelta
import itertools

# SQL connection
engine = create_engine('postgresql+psycopg2://postgres:@localhost/nc?port=5444')
con = psycopg2.connect("host='localhost' dbname='nc' user='postgres' password='' port='5444'")
cursor = con.cursor()

def populate_database():
    '''
    Loop through time and determine closest
    '''
    # import outage data
    outs = {}
    services = ['gas_station', 'super_market']
    for service in services:
        outs[service] = import_outages(service)

    # init the dataframe
    df = init_df(services, outs)

    # loop through the data frame and add to each row
    # -> this would be more efficient if I looped through time, service, then orig_id
    total_len = df.shape[0]
    p = np.linspace(0,1,21)
    idx = np.round(total_len * p)
    progress = {idx[i]:p[i] for i in range(len(p))}
    for index, row in df.iterrows():
        # progress report
        if index in progress.keys():
            print("{0:.2f} percent completed querying task".format(progress[index]))
        # what is the time and orig I'm considering
        time_now = row['time_stamp']
        # which destination points are operational/available
        ids_open = outs[row['service']][row['time_stamp']]
        # what is the distance to the closest destination
        if len(ids_open) == 0:
            df.loc[index,'distance'] = np.inf
        else:
            sql = 'SELECT MIN(distance) FROM distance_matrix WHERE id_orig = %s AND id_dest IN %s'
            cursor.execute(sql,(row['id_orig'],ids_open,))
            # save to df
            df.loc[index,'distance'] = np.float(cursor.fetchone()[0])

    # add df to sql
    df.to_sql('nearest_in_time', engine)
    # add index
    cursor.execute('CREATE INDEX on nearest_in_time (time_stamp);')
    # commit
    con.commit()


def import_outages(service_name):
    '''
    import the station and store outages and prepare the dict
    '''
    # import data
    with open('data/destinations/{}_outages.pk'.format(service_name), 'rb') as fp:
        outages = pk.load(fp)
    # convert to dict for faster querying
    if service_name == 'gas_station':
        key_name = 'stations_operational'
    else:
        key_name = 'stores_operational'
    dict = {d['datetime']:tuple([str(x) for x in d[key_name]]) for d in outages}
    return(dict)


def init_df(services, outs):
    '''
    initialize the dataframe for storing the temporal proximity data
    '''
    # get the times
    times = sorted(outs[services[0]].keys())
    # get the block ids
    sql = "SELECT block.geoid10 FROM block, city WHERE ST_Intersects(block.geom, ST_Transform(city.geom, 4269)) AND city.juris = 'WM'"
    cursor.execute(sql)
    orig_ids = cursor.fetchall()
    orig_ids = [x[0] for x in orig_ids]
    # init the df
    df = pd.DataFrame(list(itertools.product(orig_ids, times, services)), columns = ['id_orig','time_stamp','service'])
    # add additional columns
    df['distance'] = None
    df['id_dest'] = None
    return(df)

if __name__ == "__main__":
    populate_database()
