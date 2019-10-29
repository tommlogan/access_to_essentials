'''
Here the db tables are initialized for the analysis
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

state = 'nc'
passw = 'resil.north-carolina'
port = '5451'

engine = create_engine('postgresql+psycopg2://postgres:' + passw + '@localhost/' + state + '?port=' + port)
connect_string = "host='localhost' dbname='" + state + "' user='postgres' password='" + passw + "' port='" + port + "'"

osrm_url = 'http://localhost:5555'

def main():
    '''
    set up the db tables I need for the querying
    '''
    con = psycopg2.connect(connect_string)

    # init the destination tables
    # create_dest_table(con)

    # query the distances
    query_points(con)

    # temporal distance table (one for each destination type)
        # this will have columns: origin id,  time, and the distance to the nearest operation destination
    create_temporal_distance(con)


    con.close()

def create_dest_table(con):
    '''
    create a table with the supermarkets and groceries
    '''
    types = ['gas_station','super_market_operating']
    # import the csv's
    df = pd.DataFrame()
    for dest_type in types:
        df_type = pd.read_csv('data/destinations/' + dest_type + '_' + state + '.csv', encoding = "ISO-8859-1", usecols = ['id','name','lat','lon'])
        df_type['dest_type'] = dest_type
        df = df.append(df_type)

    # create geopandas df on lat lon
    geometry = [shapely.geometry.Point(xy) for xy in zip(df.lon, df.lat)]
    crs = {'init': 'epsg:4326'}
    gdf = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    # prepare for sql
    gdf['geom'] = gdf['geometry'].apply(lambda x: WKTElement(x.wkt, srid=4326))
        #drop the geometry column as it is now duplicative
    gdf.drop('geometry', 1, inplace=True)
    # set index
    gdf.set_index(['id','dest_type'], inplace=True)

    # export to sql
    gdf.to_sql('destinations', engine, dtype={'geom': Geometry('POINT', srid= 4326)})

    # update indices
    cursor = con.cursor()
    queries = ['CREATE INDEX "dest_id" ON destinations ("id");',
            'CREATE INDEX "dest_type" ON destinations ("dest_type");']
    for q in queries:
        cursor.execute(q)
    # commit to db
    con.commit()


def query_points(con):
    '''
    query OSRM for distances between origins and destinations
    '''
    # connect to db
    cursor = con.cursor()
    # get list of all origin ids
    # sql = "SELECT block.geoid10, block.geom FROM block, city WHERE ST_Intersects(block.geom, ST_Transform(city.geom, 4269)) AND city.name = 'Panama City'"
    sql = "SELECT block.geoid10, block.geom FROM block, city WHERE ST_Intersects(block.geom, ST_Transform(city.geom, 4269)) AND city.gid = 4"
    orig_df = gpd.GeoDataFrame.from_postgis(sql, con, geom_col='geom')
    orig_df['x'] = orig_df.geom.centroid.x
    orig_df['y'] = orig_df.geom.centroid.y
    # drop duplicates
    orig_df.drop('geom',axis=1,inplace=True)
    orig_df.drop_duplicates(inplace=True)
    # set index
    orig_df = orig_df.set_index('geoid10')
    # get list of destination ids
    sql = "SELECT id, lat, lon FROM destinations"
    dest_df = pd.read_sql(sql, con)
    dest_df = dest_df.set_index('id')
    # list of origxdest pairs
    origxdest = pd.DataFrame(list(itertools.product(orig_df.index, dest_df.index)), columns = ['id_orig','id_dest'])
    origxdest['distance'] = None
    ###
    # loop through the queries
    ###
    total_len = origxdest.shape[0]
    for index, row in origxdest.iterrows():
        # progress report
        if index/total_len in np.linspace(0,1,21):
            print("{} percent completed querying task".format(index/total_len*100))
        # prepare query
        lon_o,lat_o = orig_df.loc[row['id_orig']][['x','y']]
        lon_d,lat_d = dest_df.loc[row['id_dest']][['lon','lat']]
        # query
        query = '{}/route/v1/driving/{},{};{},{}?overview=false'.format(osrm_url, lon_o, lat_o, lon_d, lat_d)
        r = requests.get(query)
        origxdest.loc[index,'distance'] = r.json()['routes'][0]['legs'][0]['distance']
    # add df to sql
    origxdest.to_sql('distance_matrix', engine)
    # add index
    cursor.execute('CREATE INDEX on distance_matrix (id_orig);')
    # commit
    con.commit()


def create_temporal_distance(con):
    '''
    create the origin_nearest distance table for time
    '''
    # connect to db
    cursor = con.cursor()
    # sql query
    queries = ['''CREATE TABLE IF NOT EXISTS nearest_in_time (
                orig_id INT NOT NULL,
                dest_id INT NOT NULL,
                distance INT,
                date_time TIMESTAMP,
                dest_type VARCHAR);''',
            'CREATE INDEX on nearest_in_time (date_time);']
    # run the queries
    for q in queries:
        cursor.execute(q)
    # commit
    con.commit()


def add_column_demograph(con):
    '''
    Add a useful geoid10 column to join data with
    '''
    queries = ['ALTER TABLE demograph ADD COLUMN geoid10 CHAR(15)',
                '']

def import_csv(file_name, table_name,engine):
    '''
    import a csv into the postgres db
    '''
    if state=='FL':
        file_name = 'B:/research/resilience/data/nhgis/nhgis0032_ds172_2010_block.csv'
        # con = psycopg2.connect("host='localhost' dbname='fl' user='postgres' password='resil.florida' port='" + port + "'")
        county = '005'
    else:
        file_name = 'B:/research/resilience/data/nhgis/nhgis0030_ds172_2010_block.csv'
        # con = psycopg2.connect("host='localhost' dbname='nc' user='postgres' password='' port='" + port + "'")
        county = '129'
    #
    table_name = 'demograph'
    # add csv to nc.demograph
    df = pd.read_csv(file_name, dtype = {'STATEA':str, 'COUNTYA':str,'TRACTA':str,'BLOCKA':str, 'H7X001':int, 'H7X002':int, 'H7X003':int, 'H7X004':int})
    df = df[df.COUNTYA==county]
    df['geoid10'] = df['STATEA'] + df['COUNTYA'] + df['TRACTA'] + df['BLOCKA']
    df.to_sql(table_name, engine)
    # add the table indices

    cursor = con.cursor()
    queries = ['CREATE INDEX "geoid10" ON demograph ("geoid10");',
            'CREATE INDEX "id" ON demograph ("BLOCKA");']
    for q in queries:
        cursor.execute(q)
    # commit
    con.commit()



if __name__ == "__main__":
    main()
