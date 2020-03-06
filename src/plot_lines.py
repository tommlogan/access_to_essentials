
# user defined variables
state = 'nc' #input('State: ')
service = 'supermarket'#'gas_station'

from config import *
db, context = cfg_init(state)
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')
from matplotlib import cm
import datetime

def resilience_curve(state, service):
    '''
    Plot the resilience curve for each resident
    '''
    con = db['con']
    # import the data distance, time data
    sql = 'SELECT id_orig, time_stamp, distance FROM {} WHERE service = %s'.format(context['nearest_db_name'])
    dist = pd.read_sql(sql, con, params = (service,))
    # import population data
    sql = 'SELECT "H7X001", geoid10 FROM demograph;'
    pop = pd.read_sql(sql, con)
    pop = pop.set_index('geoid10')
    # merge population into blocks
    distxdem = pop.merge(dist, left_on = 'geoid10', right_on = 'id_orig')
    distxdem = distxdem.set_index('id_orig')

    # get list of ids
    ids = np.unique(distxdem[distxdem.H7X001>0].index)

    import code
    code.interact(local=locals())
    fig, ax = plt.subplots()
    # loop through the residents
    for id_b in ids:
        dist_id = distxdem.loc[id_b]
        for i in range(int(pop.loc[id_b].values)):
            dist_id.plot(x='time_stamp', y='distance', alpha=0.3, ax=ax)


# def ecdf(state, service, time_stamps):



if __name__ == "__main__":
    resilience_curve(state, service)
