# user defined variables
state = 'nc'

from config import *
db, context = cfg_init(state)
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('fivethirtyeight')
from matplotlib import cm

def plot(services, time_steps, sim_id, metric, db):
    '''
    plot the ecdf at a certain time
    '''
    con = db['con']
    # fig, ax = plt.subplots()
    for time_step in time_steps:
        for service in services:
            # calculate the ecdf data
            pop = calculate(service, time_step, db, context)
            # plot the cdf
            plt.plot(pop.distance/1000, pop.perc, label = service)
    # ylabel
    plt.ylabel('% residents')
    # xlabel
    plt.xlabel('Distance to open service (km)')
    plt.xlim([0,4])
    plt.ylim([0,None])
    #making the title todays date and the time_stamp
    plt.title('Time: ' + str(time_step), loc='left')
    plt.legend(loc='lower right')
    # present
    plt.show()
    # savefig
    # name = services #just for saving the file purposes
    # fig_out = 'figures/{}/cdf_{}_{}.pdf'.format(state, name, time_stamp)
    # if os.path.isfile(fig_out):
    #     os.remove(fig_out)
    # plt.savefig(fig_out, dpi=dpi, format='pdf', transparent=fig_transparency)#, bbox_inches='tight')
    # plt.clf()


def calculate(service, time_step, db, context):
    '''
    calculate the ecdf at a certain time
    '''
    con = db['con']
    # import the distance to the nearest service for this time
    sql = 'SELECT id_orig, time_stamp, distance FROM {} WHERE time_stamp = %s AND service = %s'.format(context['nearest_db_name'])
    dist = pd.read_sql(sql, con, params = (time_step, service,))
    # import number of people
    sql = 'SELECT "H7X001", "H7X002", "H7X003", geoid10 FROM demograph;'
    pop = pd.read_sql(sql, db['con'])
    pop = pop.loc[pop['H7X001'] != 0]
    # merge population into blocks
    pop = pop.merge(dist, left_on = 'geoid10', right_on = 'id_orig')
    pop['white'] = pop.H7X002
    pop['all'] = pop.H7X001
    pop['black'] = pop.H7X003
    # total population
    pop_total = pop.H7X001.sum()
    # pop_black
    pop_black_total = pop.black.sum()
    pop_white_total = pop.white.sum()
    # sort df by distance (ascending)
    pop = pop.sort_values('distance')
    # column for percent of residents
    pop['perc'] = pop.H7X001.cumsum()/pop_total*100
    pop['black_perc'] = pop.black.cumsum()/pop_black_total*100
    pop['white_perc'] = pop.white.cumsum()/pop_white_total*100
    # return df
    return(pop)
