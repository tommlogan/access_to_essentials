'''
Imports functions and variables that are common throughout the project
'''

# functions - data analysis
import numpy as np
import pandas as pd
import itertools
# functions - geospatial
import geopandas as gpd
# functions - data management
import pickle as pk
import psycopg2
from sqlalchemy.engine import create_engine
# functions - coding
import code
import os
from datetime import datetime, timedelta
import time
from tqdm import tqdm
#plotting
from scipy.integrate import simps
import matplotlib.pyplot as plt
import random
import seaborn as sns
# logging
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

def cfg_init(state):
    # SQL connection
    db = dict()
    db['passw'] = open('pass.txt', 'r').read().strip('\n')
    db['host'] = '132.181.102.2'
    db['port'] = '5001'
    # city information
    context = dict()
    if state == 'nc':
        db['name'] = 'access_nc'
        context['city_code'] = 'wil'
        context['city'] = 'wilmington'
        # url to the osrm routing machine
        context['osrm_url'] = 'http://localhost:6002'
        context['services'] = ['supermarket', 'gas_station']
    elif state == 'fl':
        db['name'] = 'access_fl_pan'
        context['city_code'] = 'pan'
        context['city'] = 'Panama_City'
        context['osrm_url'] = 'http://localhost:6012'
        context['services'] = ['supermarket', 'gas_station']
    # connect to database
    db['engine'] = create_engine('postgresql+psycopg2://postgres:' + db['passw'] + '@' + db['host'] + '/' + db['name'] + '?port=' + db['port'])
    db['address'] = "host=" + db['host'] + " dbname=" + db['name'] + " user=postgres password='"+ db['passw'] + "' port=" + db['port']
    db['con'] = psycopg2.connect(db['address'])
    return(db, context)
