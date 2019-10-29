'''
Export data from SQL into pandas
1. orig x dest travel distance
2. list of orig and population
3. list of dest ids, type of dest, and whether closed during hazard event
'''

from sqlalchemy.engine import create_engine
import psycopg2
import pandas as pd
import geopandas as gpd
import os.path
import osgeo.ogr
# from DB_CONN import *
import shapely
from geoalchemy2 import Geometry, WKTElement
import itertools
import numpy as np
import requests
import pickle as pk

# user inputs
state = 'nc'
passw = 'resil.north-carolina'
port = '5451'

# connect to database
engine = create_engine('postgresql+psycopg2://postgres:' + passw + '@localhost/' + state + '?port=' + port)
connect_string = "host='localhost' dbname='" + state + "' user='postgres' password='" + passw + "' port='" + port + "'"
con = psycopg2.connect(connect_string)


# population and orig_id (block)
sql = 'SELECT "H7X001", geoid10 FROM demograph;'
pop = pd.read_sql(sql, con)
pop['pop'] = pop.H7X001
pop = pop.drop('H7X001', axis=1)

# travel times
sql = 'SELECT * FROM distance_matrix;'
dist = pd.read_sql(sql, con)


def import_outages(service_name):
    '''
    import the station and store outages and prepare the dict
    '''
    # import data
    with open('data/destinations/{}_operating.pk'.format(service_name), 'rb') as fp:
        outages = pk.load(fp)
    # convert to dict for faster querying
    dict = {d['datetime']:d['operational_ids'] for d in outages}
    return(dict)

# destination
sql = 'SELECT id, dest_type, lat, lon FROM destinations'
dest = pd.read_sql(sql, con)
dest.loc[dest.dest_type == 'super_market_operating', 'dest_type'] = 'super_market'
# did it close during the storm?
operational = {}
services = ['gas_station', 'super_market']
for service in services:
    operational[service] = import_outages(service)

# which stores are not open the entire time
dest['closed'] = 1
for service in services:
    lens = [len(x) for x in operational[service].values()]
    idx = lens.index(min(lens))
    open_ids = list(operational[service].values())[idx]
    for id in open_ids:
        dest.loc[dest.id == id, 'closed'] = 0


# save to csv
pop.to_csv('data/optimization/population.csv')
dest.to_csv('data/optimization/destinations.csv')
dist.to_csv('data/optimization/distances.csv')




# [45430, 46154, 46156, 46158, 108055, 111853, 116431, 134710, 141186, 159038, 192951, 197149]
#
#

# store_closed = {}
# for service in services:
#     store_closed[service] = []
#     dest_ids = dest[dest.dest_type == service].id
#     for id in dest_ids:
#         for open_ids in operational[service].values():
#             if id not in open_ids:
#                 store_closed[service] += [id]
#     store_closed[service] = np.unique(store_closed[service])
# # add to pandas
# dest['closed'] = 0
# closed_ids =list(store_closed[services[0]]) + list(store_closed[services[1]])
# # dest.set_index('id')
# for id in closed_ids:
#     dest.loc[dest.id == id, 'closed'] = 1
#
# # not closed?
# [45430, 46154, 46156, 46158, 108055, 111853, 116431, 134710, 141186, 159038, 192951, 197149]
# id = 46154
# i = 0
# for open_ids in operational[service].values():
#     i += 1
#     if id not in open_ids:
#         print(i-1)
