'''
Create the stylized pdf and cdf figures as an example of how the pdf maps to the cdf,
and what the pdf/cdf looks like for each phase in the hazard cycle
'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import *
import seaborn as sns
import os

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
          'figure.figsize': [2.33,1.5],#[fig_width/2.54,fig_height/2.54]
          'axes.spines.top'    : False,
          'axes.spines.right'  : False,
          'axes.xmargin' : 0
}
mpl.rcParams.update(params)

nsim = int(1e6)
# let's say I want to get 800 meters as the threshold
thresh = 800
# the transformed will be mean=400, std=100
tr = np.random.normal(400,100,nsim)
# the prepared
pr = np.random.normal(700,150,nsim)
haz = np.random.normal(1200,250,nsim)
rec = np.random.normal(1000,200,nsim)

# plot the distribution
sns.distplot(tr, hist=False, rug=False);
sns.distplot(pr, hist=False, rug=False);
sns.distplot(haz, hist=False, rug=False);
sns.distplot(rec, hist=False, rug=False);

plt.xlim(0,None)
plt.ylabel('% population \n at x distance')
plt.xlabel('distance')
plt.xticks([], []); plt.yticks([], [])
fig_out = 'fig/pdf.pdf'
# if os.path.isfile(fig_out):
#     os.remove(fig_out)
#
# plt.savefig(fig_out, dpi=dpi, format='pdf', transparent=fig_transparency)#, bbox_inches='tight')
# plt.show()

# plot the cdf
sns.distplot(tr, kde_kws=dict(cumulative=True));
sns.distplot(pr, kde_kws=dict(cumulative=True));
sns.distplot(haz, kde_kws=dict(cumulative=True));
sns.distplot(rec, kde_kws=dict(cumulative=True));

plt.xlim(0,None)
plt.ylabel('% population \n within x distance')
plt.xlabel('distance')
plt.xticks([], []); plt.yticks([], [])
plt.axvline(x=thresh, ls='--', color = 'k')
fig_out = 'fig/cdf.pdf'
# if os.path.isfile(fig_out):
#     os.remove(fig_out)
# plt.savefig(fig_out, dpi=dpi, format='pdf', transparent=fig_transparency)#, bbox_inches='tight')
plt.show()

# create a resilience line in illustrator once I've put these other figures there.
