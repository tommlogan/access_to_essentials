# user defined variables
state = 'fl' #input('State: ')
service = 'gas_station'#'gas_station'

from config import *
db, context = cfg_init(state)
logger = logging.getLogger(__name__)

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style

from matplotlib import cm
import datetime
from math import *

# db connection
con = db['con']
cursor = con.cursor()

# define the plotting style
plt.style.use(['tableau-colorblind10'])#,'dark_background'])
fig_transparency = False
# figure size (cm)
fig_width = 22*2/3#33.5#8.26
golden_mean = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
fig_height = 16*2/3#16#6.43 #fig_width/golden_mean
# font size
font_size = 8
dpi = 500
# additional parameters
params = {'axes.labelsize': font_size, # fontsize for x and y labels (was 10)
          'font.size': font_size, # was 10
          'legend.fontsize': font_size * 2/3, # was 10
          'xtick.labelsize': font_size,
          'font.sans-serif' : 'Corbel',
          # 'ytick.labelsize': 0,
          'lines.linewidth' : 1,
          'figure.autolayout' : True,
          'figure.figsize': [3,1.5],#[fig_width/2.54,fig_height/2.54]
          'axes.spines.top'    : False,
          'axes.spines.right'  : False,
          'axes.xmargin' : 0
}
mpl.rcParams.update(params)

def main():
    '''
    plots
    '''

    services = ['gas_station','supermarket']#,'gas_station']
    # import the service operational ids over time
    operating = {}
    for service in services:
        operating[service] = import_operating(service)
        # service_restoration(service)
    # Plot service restoration
    # for michael i want to end at Nov 25
    date_list = list(operating[service].keys())
    # code.interact(local=locals())
    if state == 'fl':
        date_loop = np.linspace(0,1128,10)
        date_loop = np.linspace(0,750,10)
        # delete the future dates
        for service in services:
            if service == 'supermarket':
                operating[service] = {t:operating[service][t] for t in date_list[0:750]}#1130]}
            else:
                operating[service] = {t:operating[service][t] for t in date_list[0:1130]}
    else:
        date_loop = np.linspace(0,len(date_list)-1,10)
    # for i in date_loop:
    i = date_loop[2]
    time_stamp = date_list[int(i)]
    # code.interact(local=locals())
    for service in services:
        # determine the geoid10's access quintiles
        access_quintiles = determine_quintile(date_list[0], service, operating)
        # plot the resilience curve
        resilience_curve(service, operating, time_stamp, access_quintiles)
        # Plot choropleth
        # plot_ecdf(time_stamp, service, operating)
        # choropleth_city(time_stamp, service, operating)



def determine_quintile(time_stamp, service, operating):
    '''
    calculate the ecdf at a certain time
    '''
    # import the distance to the nearest service for this time
    sql = 'SELECT distance, id_orig FROM {} WHERE time_stamp = %s AND service = %s'.format(context['nearest_db_name'])
    dist = pd.read_sql(sql, con, params = (time_stamp, service,))
    # import number of people
    sql = 'SELECT "H7X001", "H7X002", geoid10 FROM demograph;'
    pop = pd.read_sql(sql, con)
    # merge population into blocks
    pop = pop.merge(dist, left_on = 'geoid10', right_on = 'id_orig')
    pop['total'] = pop.H7X001
    # determine the quintile labels based on distance (weighted so the bins are equal population)
    pop['access_quintile'] = weighted_qcut(values = dist.distance, weights = pop.total, q = 5, labels=False, duplicates='drop')
    # create populations based on access quintle
    pop['access_rich'] = pop.total * (pop.access_quintile == 0)
    pop['access_poor'] = pop.total * (pop.access_quintile == 4)
    # drop a couple columns
    access_quintiles = pop.drop(columns=['H7X001', 'H7X002', 'distance','id_orig','total'])
    # return df
    return(access_quintiles)


def resilience_curve(service, operating, time_stamp, access_quintiles):
    '''
    Plot the resilience curve
    '''
    time_stamp_line = time_stamp
    # percentiles
    percentiles = np.linspace(0,1,11)[1:-1]
    col_list = ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
    # init df
    times = sorted(operating[service].keys())
    df = pd.DataFrame({'time_stamp':times})
    for p in percentiles:
        df['p{0:0.1f}'.format(p)] = None
    df['mean'] = None
    df_names = ['p{0:0.1f}'.format(i) for i in percentiles]
    # loop through time
    for index, row in df.iterrows():
        time_stamp = df.loc[index,'time_stamp']
        pop = calc_ecdf(time_stamp, service, operating)
        # merge with the quinitles
        pop = pop.merge(access_quintiles, on = 'geoid10')
        # code.interact(local=locals())
        resil_values = weighted_quantile(pop.distance.values, percentiles, sample_weight=pop.H7X001.values, values_sorted=True)
        # df.loc[index, 'mean'] = np.average(pop.distance.values, weights = pop.H7X001.values)
        df.loc[index, 'white_025th'] = weighted_quantile(pop.distance.values, 0.025, sample_weight=pop.white.values, values_sorted=True)
        df.loc[index, 'white_975th'] = weighted_quantile(pop.distance.values, 0.975, sample_weight=pop.white.values, values_sorted=True)
        df.loc[index, 'mean_white'] = np.average(pop.distance.values, weights = pop.white.values)
        df.loc[index, 'mean_nonwhite'] = np.average(pop.distance.values, weights = pop.nonwhite.values)
        df.loc[index, 'nonwhite_025th'] = weighted_quantile(pop.distance.values, 0.025, sample_weight=pop.nonwhite.values, values_sorted=True)
        df.loc[index, 'nonwhite_975th'] = weighted_quantile(pop.distance.values, 0.975, sample_weight=pop.nonwhite.values, values_sorted=True)
        df.loc[index,df_names] = resil_values
        # access groups
        df.loc[index, 'mean_best'] = np.average(pop.distance.values, weights = pop.access_rich.values)
        df.loc[index, 'mean_worst'] = np.average(pop.distance.values, weights = pop.access_poor.values)
    if state == 'fl' and service == 'gas_station':
        df['mean_white'] = df['mean_white'].rolling(10).median()
        df['mean_nonwhite'] = df['mean_nonwhite'].rolling(10).median()
        df['mean_best'] = df['mean_best'].rolling(10).median()
        df['mean_worst'] = df['mean_worst'].rolling(10).median()
    # plot
    # import code
    # code.interact(local=locals())
    # for i in range(len(percentiles)-1):
    # plt.fill_between(df.time_stamp.values, df['white_025th']/1000, df['white_975th']/1000, alpha=0.3)
    # plt.fill_between(df.time_stamp.values, df['nonwhite_025th']/1000, df['nonwhite_975th']/1000, alpha=0.3)
    plt.plot(df.time_stamp, df['mean_white']/1000, label = 'white', color = 'k', alpha = 0.3)
    plt.plot(df.time_stamp, df['mean_nonwhite']/1000, label = 'non-white', color = 'k', alpha = 1)
    # distributional effects
    plt.plot(df.time_stamp, df['mean_best']/1000, label = 'access-rich', linestyle = '--')
    plt.plot(df.time_stamp, df['mean_worst']/1000, label = 'access-poor', linestyle = '--')
    # plt.plot(df.time_stamp, np.array(df[df_names[0]], dtype = float)/1000, linestyle = '--', color = 'k')
    # plt.plot(df.time_stamp, np.array(df[df_names[i+1]], dtype = float)/1000, linestyle = '--', color = 'k')
    # plt.axvline(x=time_stamp_line, color = 'k')
    # # land fall
    if state == 'fl':
        plt.axvline(datetime.datetime(2018,10,10,12,0),ls='--', color = 'k', linewidth=0.5)
        # plt.text(datetime(2018,10,10,20,0), 3.5,'landfall', fontsize=5)
        plt.xlim([None, datetime.datetime(2018,11,9,12,0)])
        x_len = df.index[df.time_stamp == datetime.datetime(2018,11,9,0,0)].tolist()[0]
    else:
        plt.axvline(datetime.datetime(2018,9,14,7,0),ls='--', color = 'k', linewidth=0.5)
        # plt.xlim([None, datetime(2018,9,29,0,0)])
        # x_len = df.index[df.time_stamp == datetime(2018,9,29,0,0)].tolist()[0]
        plt.xlim([None, datetime.datetime(2018,10,9,0,0)])
        x_len = df.index[df.time_stamp == datetime.datetime(2018,10,9,0,0)].tolist()[0]
     #   plt.text(datetime(2018,9,11,0,0), 3.5,'landfall', fontsize=5)
    # x ticks
    x_dummy = np.linspace(0,x_len,4)
    x_locs = [df.time_stamp[int(i)].date().strftime("%d-%b-%Y") for i in x_dummy]
    x_locs = [datetime.datetime.strptime(i,"%d-%b-%Y") for i in x_locs]
    x_labels = [df.time_stamp[int(i)].date().strftime("%d-%b") for i in x_dummy]
    # x_labels = [dist_id.time_stamp[int(i)].date().strftime("%d-%b") for i in x_dummy]
    plt.xticks(ticks=x_locs, labels=x_labels, rotation=0)
    # ylabel
    # plt.ylim([3,9])
    if service == 'gas_station':
        # plt.ylabel('Distribution of distance \n to open gas station (m)')
        plt.ylabel('Km to open facility')#service \n station: Wilmington, NC')
        if state == 'fl': plt.ylim([0,5])
        else: plt.ylim([0,4])
        # plt.yticks([1,2])
    else:
        plt.ylabel('Km to open facility')#super \n market: Panama City, FL')
        if state == 'fl': plt.ylim([0,12])
        else: plt.ylim([0,7])
    plt.gca().invert_yaxis()
    # legend
    plt.legend(loc='lower right')
    plt.xlabel('Going to delete this')
    # savefig
    # plt.show()
    fig_out = 'fig/resilience_equity_{}_{}.pdf'.format(state,service,time_stamp_line.strftime("%Y%m%d-%H"))
    if os.path.isfile(fig_out):
        os.remove(fig_out)
    plt.savefig(fig_out, dpi=dpi, format='pdf')#, bbox_inches='tight', transparent=fig_transparency)
    # plt.show()
    plt.clf()


def calc_ecdf(time_stamp, service, operating):
    '''
    calculate the ecdf at a certain time
    '''
    # import the distance to the nearest service for this time
    sql = 'SELECT distance, id_orig FROM {} WHERE time_stamp = %s AND service = %s'.format(context['nearest_db_name'])
    dist = pd.read_sql(sql, con, params = (time_stamp, service,))
    # import number of people
    sql = 'SELECT "H7X001", "H7X002", geoid10 FROM demograph;'
    pop = pd.read_sql(sql, con)
    # merge population into blocks
    pop = pop.merge(dist, left_on = 'geoid10', right_on = 'id_orig')
    pop['white'] = pop.H7X002
    pop['nonwhite'] = pop.H7X001 - pop.H7X002
    # total population
    pop_total = pop.H7X001.sum()
    # pop_nonwhite
    pop_nonwhite_total = pop.nonwhite.sum()
    pop_white_total = pop.white.sum()
    # sort df by distance (ascending)
    pop = pop.sort_values('distance')
    # column for percent of residents
    pop['perc'] = pop.H7X001.cumsum()/pop_total*100
    pop['nonwhite_perc'] = pop.nonwhite.cumsum()/pop_nonwhite_total*100
    pop['white_perc'] = pop.white.cumsum()/pop_white_total*100
    # return df
    return(pop)


def import_operating(service_name):
    '''
    import the station and store outages and prepare the dict
    '''
    # import data
    if state=='fl':
        in_name = 'data/pan/destination/{}.pk'
    else:
        in_name = 'data/wil/destination/{}.pk'
    with open(in_name.format(service_name), 'rb') as fp:
        operating = pk.load(fp)
    # convert to dict for faster querying
    dict = {d['datetime']:d['operational_ids'] for d in operating}
    return(dict)


def weighted_qcut(values, weights, q, **kwargs):
    # thanks - https://stackoverflow.com/questions/45528029/python-how-to-create-weighted-quantiles-in-pandas
    from pandas._libs.lib import is_integer
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
    order = weights[values.argsort()].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    return bins.sort_index()


def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    # https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy/29677616#29677616
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'
    #
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
    #
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with np.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

if __name__ == "__main__":
    main()
