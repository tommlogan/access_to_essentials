'''
Populate the database for the nearest proximity throughout time
'''
state = 'fl'
from config import *
db, context = cfg_init(state)
logger = logging.getLogger(__name__)



def populate_database(db):
    '''
    Loop through time and determine closest
    '''
    # db connections
    con = db['con']
    engine = db['engine']
    cursor = db['con'].cursor()
    # import outage data
    outs = {}
    services = ['gas_station','supermarket']
    for service in services:
        outs[service] = import_outages(service)

    # init the dataframe
    df = pd.DataFrame(columns = ['id_orig','distance','time_stamp','service'])

    # get the times
    times = sorted(outs[services[0]].keys())

    # get the distance matrix
    distances = pd.read_sql('SELECT * FROM distance', con)
    distances = distances.set_index('id_dest')
    distances.distance = pd.to_numeric(distances.distance)

    # block ids
    id_orig = np.unique(distances.id_orig)

    # loop through the times
    total_len = len(times)
    p = np.linspace(0,1,21)
    idx = np.round(total_len * p)
    progress = {idx[i]:p[i] for i in range(len(p))}
    for index in range(total_len):
        time_stamp = times[index]
        # progress report
        if index in progress.keys():
            print("{0:s} ----- {1:.0f}% completed querying task".format(time.ctime(), progress[index]*100))
        # loop services
        for service in services:
            # which stores are operating?
            ids_open = outs[service][time_stamp]
            if len(ids_open) == 0:
                df_min = pd.DataFrame({'id_orig' : id_orig})
                df_min['distance'] = np.inf
            else:
                # subset the distance matrix on dest_id
                dists_sub = distances.loc[ids_open]
                # get the minimum distance
                df_min = dists_sub.groupby('id_orig')['distance'].min()
                # prepare df to append
                df_min = df_min.to_frame('distance')
                df_min.reset_index(inplace=True)
            # prepare df to append
            df_min['time_stamp'] = time_stamp
            df_min['service'] = service
            # append
            df = df.append(df_min, ignore_index=True)
    print("{0:s} ----- {1:.0f}% completed querying task".format(time.ctime(), 100))
    # add df to sql
    df.to_sql(context['nearest_db_name'], engine)
    logger.info('Added to sql')
    # add index
    cursor.execute('CREATE INDEX on {} (time_stamp);'.format(context['nearest_db_name']))
    logger.info('Indexing completed')
    # commit
    con.commit()
    logger.info('Committed and complete')


def import_outages(service_name):
    '''
    import the station and store outages and prepare the dict
    '''
    # import data
    with open('data/{}/destination/{}.pk'.format(context['city_code'], service_name), 'rb') as fp:
        outages = pk.load(fp)
    # convert to dict for faster querying
    dict = {d['datetime']:d['operational_ids'] for d in outages}
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
    populate_database(db)
