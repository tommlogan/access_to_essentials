'''
Init the database
Query origins to dests in OSRM
'''
# user defined variables
state = input('State: ')
par = True
par_frac = 0.8

from config import *
db, context = cfg_init(state)
logger = logging.getLogger(__name__)
import os.path
import osgeo.ogr
import shapely
from geoalchemy2 import Geometry, WKTElement
import requests
from sqlalchemy.types import Float, Integer
import yagmail
if par == True:
    import multiprocessing as mp
    from joblib import Parallel, delayed
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry

def main(db, context):
    '''
    set up the db tables I need for the querying
    '''

    # init the destination tables
    create_dest_table(db)

    # query the distances
    query_points(db, context)

    # close the connection
    db['con'].close()
    logger.info('Database connection closed')

    # email completion notification
    utils.send_email(body='Querying {} complete'.format(context['city']))


def create_dest_table(con):
    '''
    create a table with the supermarkets and groceries
    '''
    # db connections
    con = db['con']
    engine = db['engine']
    # destinations and locations
    types = ['gas_station','supermarket']
    # import the csv's
    df = pd.DataFrame()
    for dest_type in types:
        df_type = pd.read_csv('data/destinations/' + dest_type + '.csv', encoding = "ISO-8859-1", usecols = ['id','name','lat','lon'])
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
    cursor = db['con'].cursor()

    # get list of all origin ids
    sql = "SELECT * FROM block"
    orig_df = gpd.GeoDataFrame.from_postgis(sql, db['con'], geom_col='geom')
    orig_df['x'] = orig_df.geom.centroid.x
    orig_df['y'] = orig_df.geom.centroid.y
    # drop duplicates
    orig_df.drop('geom',axis=1,inplace=True)
    orig_df.drop_duplicates(inplace=True)
    # set index
    orig_df = orig_df.set_index('geoid10')

    # get list of destination ids
    sql = "SELECT * FROM destinations"
    dest_df = gpd.GeoDataFrame.from_postgis(sql, db['con'], geom_col='geom')
    dest_df = dest_df.set_index('id')
    dest_df['lon'] = dest_df.geom.centroid.x
    dest_df['lat'] = dest_df.geom.centroid.y

    # list of origxdest pairs
    origxdest = pd.DataFrame(list(itertools.product(orig_df.index, dest_df.index)), columns = ['id_orig','id_dest'])
    origxdest['distance'] = None

    # build query list:
    query_0 = np.full(fill_value = context['osrm_url'] + '/route/v1/driving/', shape=origxdest.shape[0], dtype = object)
    # the query looks like this: '{}/route/v1/driving/{},{};{},{}?overview=false'.format(osrm_url, lon_o, lat_o, lon_d, lat_d)
    queries = query_0 + np.array(orig_df.loc[origxdest['id_orig'].values]['x'].values, dtype = str) + ',' + np.array(orig_df.loc[origxdest['id_orig'].values]['y'].values, dtype = str) + ';' + np.array(dest_df.loc[origxdest['id_dest'].values]['lon'].values, dtype = str) + ',' + np.array(dest_df.loc[origxdest['id_dest'].values]['lat'].values, dtype = str) + '?overview=false'
    ###
    # loop through the queries
    ###
    logger.info('Beginning to query {}'.format(context['city']))
    total_len = len(queries)
    if par == True:
        # Query OSRM in parallel
        num_workers = np.int(mp.cpu_count() * par_frac)
        distances = Parallel(n_jobs=num_workers)(delayed(single_query)(query) for query in tqdm(queries))
        # input distance into df
        origxdest['distance'] = distances
    else:
        for index, query in enumerate(tqdm(queries)):
            # single query
            r = requests.get(query)
            # input distance into df
            origxdest.loc[index,'distance'] = r.json()['routes'][0]['legs'][0]['distance']
    logger.info('Querying complete')

    # add df to sql
    logger.info('Writing data to SQL')
    origxdest.to_sql('distance', con=db['engine'], if_exists='replace', index=False, dtype={"distance":Float(), 'id_dest':Integer()})
    # update indices
    queries = ['CREATE INDEX "dest_idx" ON distance ("id_dest");',
            'CREATE INDEX "orig_idx" ON distance ("id_orig");']
    for q in queries:
        cursor.execute(q)

    # commit to db
    db['con'].commit()
    logger.info('Distances written successfully to SQL')


def single_query(query):
    '''
    this is for if we want it parallel
    query a value and add to the table
    '''
    # query
    # dist = requests.get(query).json()['routes'][0]['legs'][0]['distance']
    dist = requests_retry_session(retries=100, backoff_factor=0.01, status_forcelist=(500, 502, 504), session=None).get(query).json()['routes'][0]['legs'][0]['distance']
    # dist = r.json()['routes'][0]['legs'][0]['distance']
    return(dist)


def requests_retry_session(retries=10, backoff_factor=0.1, status_forcelist=(500, 502, 504), session=None):
    '''
    When par ==True, issues with connecting to the docker, can change the retries to keep trying to connect
    '''
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff_factor, status_forcelist=status_forcelist)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def import_csv(db):
    '''
    import a csv into the postgres db
    '''
    # db connections
    con = db['con']
    engine = db['engine']

    if state=='fl':
        file_name = 'data/pan/nhgis0038_ds172_2010_block.csv'
        # con = psycopg2.connect("host='localhost' dbname='fl' user='postgres' password='resil.florida' port='" + port + "'")
        county = '005'
    else:
        file_name = 'data/nc/nhgis0030_ds172_2010_block.csv'
        # con = psycopg2.connect("host='localhost' dbname='nc' user='postgres' password='' port='" + port + "'")
        county = '129'
    #
    table_name = 'demograph'
    # add csv to nc.demograph
    df = pd.read_csv(file_name, dtype = {'STATEA':str, 'COUNTYA':str,'TRACTA':str,'BLOCKA':str, 'H7X001':int, 'H7X002':int, 'H7X003':int, 'H7X004':int})
    df = df[df.COUNTYA==county]
    df['geoid10'] = df['STATEA'] + df['COUNTYA'] + df['TRACTA'] + df['BLOCKA']
    import code
    code.interact(local=locals())
    df.to_sql(table_name, engine)



    # add the table indices
    cursor = con.cursor()
    queries = ['CREATE INDEX "geoid10" ON demograph ("geoid10");',
            'CREATE INDEX "id" ON demograph ("BLOCKA");']
    for q in queries:
        cursor.execute(q)
    # commit
    con.commit()


def send_email(body):
    # send an email

    receiver = "tom.logan@canterbury.ac.nz"

    yag = yagmail.SMTP('toms.scrapers',open('pass_email.txt', 'r').read().strip('\n'))
    yag.send(
        to=receiver,
        subject="Query complete",
        contents=body,
        # attachments=filename,
    )


if __name__ == "__main__":
    main(db, context)
    # import_csv(db)
