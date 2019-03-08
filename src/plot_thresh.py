import geopandas as gpd
from math import *
import numpy as np
import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as pk
import itertools
import code
import os

state = 'nc'




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
          'figure.figsize': [3.4,1.7],#[fig_width/2.54,fig_height/2.54]
          'axes.spines.top'    : False,
          'axes.spines.right'  : False,
          'axes.xmargin' : 0
}
mpl.rcParams.update(params)



def main():
    '''
    plots
    '''
    df = pd.DataFrame()
    services = ['super_market','gas_station']
    states = ['fl','nc']
    # loop through states
    for state in states:
        input("Press Enter to continue...")
        # connect to database
        if state == 'nc':
            con = psycopg2.connect("host='localhost' dbname='nc' user='postgres' password='' port='5444'")
        else:
            con = psycopg2.connect("host='localhost' dbname='fl' user='postgres' password='resil.florida' port='5444'")
        cursor = con.cursor()
        # landfall
        if state == 'fl':
            landfall = datetime(2018,10,10,12,0)
        else:
            landfall = datetime(2018,9,14,7,0)
        # import the service operational ids over time
        operating = {}
        for service in services:
            operating[service] = import_operating(service, state)
            # service_restoration(service)
        # Plot service restoration
        # for michael i want to end at Nov 25
        date_list = list(operating[service].keys())
        # code.interact(local=locals())
        if state == 'fl':
            # delete the future dates
            for service in services:
                operating[service] = {t:operating[service][t] for t in date_list[0:750]}#1130]}
        # code.interact(local=locals())
        for service in services:
            times = sorted(operating[service].keys())
            for t in times:
                # sort the population and distance
                pop = calc_ecdf(t, service, operating, con)
                for thresh in [800,1600]:
                    idx = np.searchsorted(pop.distance, thresh)[0]
                    new_row = {
                            'time_delta':(t - landfall)/timedelta(days=1),
                            'service':service,
                            'distance':thresh,
                            'state':state,
                            'perc_population':pop.iloc[idx]['perc']
                            }
                    # append to dataframe
                    df = df.append(new_row, ignore_index=True)
                # add the mean
                new_row = {
                        'time_delta':(t - landfall)/timedelta(days=1),
                        'service':service,
                        'distance':np.average(pop.distance.values, weights = pop.H7X001.values),
                        'state':state,
                        'perc_population':'mean'
                        }
                # append to dataframe
                df = df.append(new_row, ignore_index=True)
    # save csv
    df.to_csv('data/suff_access.csv')
    code.interact(local=locals())
    # plot
    # resilience_curve(df, services, states)


def resilience_curve(df, services, states):
    '''
    Plot the resilience curve
    '''

    # smooth the service stations in panama city
    df.loc[(df.state == 'fl') & (df.service == 'gas_station') & (df.perc_population == 'mean'),'distance'] = df[(df.state == 'fl') & (df.service == 'gas_station') & (df.perc_population == 'mean')]['distance'].rolling(10).median()
    df.loc[(df.state == 'fl') & (df.service == 'gas_station') & (df.distance == 800),'perc_population'] = df[(df.state == 'fl') & (df.service == 'gas_station') & (df.distance == 800)]['perc_population'].rolling(10).median()
    df.loc[(df.state == 'fl') & (df.service == 'gas_station') & (df.distance == 1600),'perc_population'] = df[(df.state == 'fl') & (df.service == 'gas_station') & (df.distance == 1600)]['perc_population'].rolling(10).median()

    # mean values
    # subset data to just mean
    df_plot = df[df.perc_population == 'mean']
    for service in services:
        # subset again
        df_plot_s = df_plot[df_plot.service == service]
        for state in states:
            # subset
            df_plot_s_s = df_plot_s[df_plot_s.state == state]
            # plot
            plt.plot(df_plot_s_s.time_delta, df_plot_s_s['distance']/1000, label = state)
        plt.axvline(0,ls='--', color = 'k', linewidth=0.5)
        plt.ylabel('Km to open facility')
        plt.xlabel('Days since landfall')
        plt.gca().invert_yaxis()
        plt.legend(loc='lower right')
        fig_out = 'fig/resilience_meanD_{}.pdf'.format(service)
        if os.path.isfile(fig_out):
            os.remove(fig_out)
        plt.savefig(fig_out, dpi=dpi, format='pdf')
        plt.clf()

    # sufficient access
    for service in services:
        # subset again
        df_plot = df[df.service == service]
        for thresh in [800, 1600]:
            df_plot_s = df_plot[df_plot.distance == thresh]
            for state in states:
                # subset
                df_plot_s_s = df_plot_s[df_plot_s.state == state]
                # plot
                plt.plot(df_plot_s_s.time_delta, df_plot_s_s['perc_population'], label = '{}_{}'.format(state,thresh))
        plt.axvline(0,ls='--', color = 'k', linewidth=0.5)
        plt.ylabel('% residents with \n sufficient access')
        plt.xlabel('Days since landfall')
        # plt.gca().invert_yaxis()
        plt.legend(loc='lower right')
        fig_out = 'fig/resilience_suffAccess_{}_threshs.pdf'.format(service,thresh)
        if os.path.isfile(fig_out):
            os.remove(fig_out)
        plt.savefig(fig_out, dpi=dpi, format='pdf')
        plt.clf()


def calc_ecdf(time_stamp, service, operating, con):
    '''
    calculate the ecdf at a certain time
    '''
    # import the distance to the nearest service for this time
    sql = 'SELECT distance, id_orig FROM nearest_in_time WHERE time_stamp = %s AND service = %s'
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


def import_operating(service_name, state):
    '''
    import the station and store outages and prepare the dict
    '''
    # import data
    if state=='fl':
        in_name = 'data/destinations/{}_operating_FL.pk'
    else:
        in_name = 'data/destinations/{}_operating.pk'
    with open(in_name.format(service_name), 'rb') as fp:
        operating = pk.load(fp)
    # convert to dict for faster querying
    dict = {d['datetime']:d['operational_ids'] for d in operating}
    return(dict)


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
