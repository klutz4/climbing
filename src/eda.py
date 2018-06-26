import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'small',
    'legend.fontsize'     : 'medium',
})

boulders = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/boulders.csv')
routes = pd.read_csv('/Users/Kelly/galvanize/capstones/mod1/data/routes.csv')

def plot_routes():
    rpg = sns.countplot(x='usa_routes', hue ='sex', data=routes)
    handles, _ = rpg.get_legend_handles_labels()
    rpg.legend(handles, ["Male", "Female"])
    rpg.set_xticklabels(['5.4','5.5','5.6','5.7','5.8','5.9','5.10a','5.10b','5.10c','5.10d','5.11a','5.11b','5.11c','5.12a','5.12b','5.12c','5.12d','5.13a','5.13b','5.13c','5.13d','5.14a','5.14b','5.14c','5.14d','5.15a'])
    rpg.set_title('Routes per Gender')
    for tick in rpg.get_xticklabels():
        tick.set_rotation(90)
    plt.savefig('images/routes_per_gender.png')
    plt.clf()

def routes_scatter(col,filename):
    fig2 = plt.figure(figsize=(15,8))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(x=routes[col],y=routes.usa_routes,color='green',alpha=0.5)
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start, end, .01))
    ax2.set_yticklabels([0,'5.4','5.5','5.6','5.7','5.8','5.9','5.10a','5.11a','5.12a','5.13a','5.14a','5.15a'])
    ax2.set_xlabel(col)
    ax2.set_ylabel('Route grade')
    ax2.set_title('Routes per {}'.format(col))
    plt.savefig(filename)
    plt.clf()

def plot_boulders():
    bpg = sns.countplot(x='usa_boulders', hue = 'sex', data=boulders)
    handles, _ = bpg.get_legend_handles_labels()
    bpg.legend(handles, ["Male", "Female"])
    bpg.set_xticklabels(['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16'])
    bpg.set_title('Boulders per Gender')
    plt.savefig('images/boulders_per_gender.png')
    plt.clf()

def boulders_scatter(col,filename):
    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(111)
    ax.scatter(x=boulders[col],y=boulders.usa_boulders,color='green',alpha=0.5)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, 1))
    ax.set_yticklabels(('V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16'))
    ax.set_xlabel(col)
    ax.set_ylabel('Boulder grade')
    ax.set_title('Boulders per {}'.format(col))
    plt.savefig(filename)
    plt.clf()
