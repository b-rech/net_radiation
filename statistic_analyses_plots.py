# -*- coding: utf-8 -*-
"""
FEDERAL UNIVERSITY OF SANTA CATARINA
TECHNOLOGICAL CENTER
DEPT. OF SANITARY AND ENVIRONMENTAL ENGINEERING
LABORATORY OF MARINE HYDRAULICS

Created on 2022/11/06
Author: Bruno Rech (b.rech@outlook.com)

SCRIPT 3/3 - STATISTICS AND PLOTS
"""

# %% LIBRARIES AND SCRIPTS

# Required libraries
import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import geemap
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import matplotlib.ticker as tkr
plt.rcParams['figure.dpi'] = 300


# %% DATA UPLOAD

# Images metadata
metadata = pd.read_csv('generated_data\\images_metadata.csv', sep=';',
                            decimal=',')
metadata['date'] = pd.to_datetime(metadata.date)


# Spectral data from samples
spectral_data = pd.read_csv('generated_data\\spectral_data.csv', sep=';',
                            decimal=',')

# Radiation data from samples
radiation_data = pd.read_csv('generated_data\\radiation_data.csv', sep=';',
                            decimal=',')


# %% PLOT - TEMPORAL DISTRIBUTION OF IMAGES

# Create figure
sns.set_style('white')
plot, ax = plt.subplots(nrows=10, figsize=(10, 6), dpi=300)

# Iterate on each axis (year)
for year, axis in zip(
        range(metadata.year.min(), metadata.year.max() + 1),
        range(0, 10)):

    # Styling
    ax[axis].set(xlim=(np.datetime64(f'{year}-01-01'),
                       np.datetime64(f'{year}-12-31')),
                 ylim=(-0.1, 0.1),
                 xticklabels=[], yticklabels=[],
                 ylabel = year, xlabel=' ')
    ax[axis].grid(visible=True, which='major', axis='both')

    # Total available images (with good cloud cover)
    sns.scatterplot(x='date', y=0,
                    data=metadata[metadata.year == year],
                    ax=ax[axis], legend=False, marker='s', hue='season',
                    palette=['#ff7f00', '#de2d26', '#33a02c', '#3182bd'],
                    hue_order=['Spring', 'Summer', 'Fall', 'Winter'],
                    edgecolor='black')

# Legend handles
handles = [Rectangle(xy=(0, 0), height=1, width=1, facecolor=color,
                     edgecolor='black', linewidth=.5)
           for color in ['#ff7f00', '#de2d26', '#33a02c', '#3182bd']]

# Add legend
plot.legend(handles=handles, labels=['Primavera', 'Verão', 'Outono', 'Inverno'],
            loc='lower center', handlelength=.8, handleheight=.8, ncol=4)

# Months (x axis)
ax[9].set(xticklabels=['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago',
                       'Set', 'Out', 'Nov', 'Dez'], label='Mês')

# plt.savefig('temporal_availability.tif', dpi=300)


# %% PLOT - SPECTRAL SIGNATURES

sns.set_theme(style='whitegrid')

classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

# Columns to melt
spectral_cols = ['B' + str(x) for x in range(2, 8)]

# Dataset with all pixels
spectral_to_plot = spectral_data.melt(id_vars=['date', 'classes'],
                                      value_vars=spectral_cols,
                                      var_name='bands',
                                      value_name='reflect')

# Create grid
spectral_grid = sns.FacetGrid(data=spectral_to_plot, col='classes',
                              col_wrap=3, col_order=classes,
                              sharex=False, sharey=False)


# Mean points
spectral_grid.map(sns.pointplot, 'bands', 'reflect', markers='.', join=False,
                  errorbar=('pi', 95), order=spectral_cols, errwidth=1,
                  capsize=.4, n_boot=500000, color='black')


# Configure axes labels
spectral_grid.set_xlabels('Banda')
spectral_grid.set_ylabels('Reflectância')

# Add classes titles and format y ticks
for ax, title in zip(spectral_grid.axes.flatten(), classes):
    ax.set_title(title, fontweight='bold')
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda y, p: f'{y:.2f}'))

plt.tight_layout()

# Save plot as tif
# plt.savefig('spectral_signatures.tif', dpi=300)


# %% SEASONAL BOXPLOT FUNCTION

# Define function
def boxplot_seasons(param, ylabel):

    sns.set_style('whitegrid')

    classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

    # Facecolor and edgecolor
    rn_boxplot_fc = 'black'
    rn_boxplot_ec = 'black'

    # Create plot - boxplot by season
    season_boxplot = sns.FacetGrid(data=radiation_data, col='classes',
                                   sharex=False, sharey=False,col_wrap=2,
                                   col_order=classes, aspect=1.5)

    # Dictionary of colors for seasons
    season_colors = {'Spring':'#ff7f00', 'Summer':'#de2d26',
                     'Fall':'#33a02c', 'Winter':'#3182bd'}

    # Seasons in order to be displayed
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']

    # Map plots to grid
    season_boxplot.map(sns.boxplot, 'season', param,'season', showmeans=True,
                       hue_order=season_order, order=season_order, dodge=False,
                       palette=season_colors,
                       flierprops={'marker':'.',
                                   'markerfacecolor':rn_boxplot_ec,
                                   'markeredgecolor':rn_boxplot_ec},
                       boxprops={'edgecolor':rn_boxplot_ec, 'linewidth':1},
                       whiskerprops={'color':rn_boxplot_ec, 'linewidth':1},
                       capprops={'color':rn_boxplot_ec, 'linewidth':1},
                       medianprops={'color':rn_boxplot_ec, 'linewidth':1},
                       meanprops={'marker':'s',
                                  'markerfacecolor':rn_boxplot_ec,
                                  'markeredgecolor':rn_boxplot_ec,
                                  'markersize':4})


    # Configure axes labels
    season_boxplot.set_xlabels(' ')
    season_boxplot.set_ylabels(ylabel, fontsize=10)

    # Add classes titles
    for ax, title in zip(season_boxplot.axes.flatten(), classes):
        ax.set_title(title)

    # Set x tick labels
    season_boxplot.set_xticklabels(['Primavera', 'Verão', 'Outono', 'Inverno'],
                                   fontsize=11)

    return season_boxplot

# %% SEASONAL BOXPLOTS

# Short wave radiation budget
boxplot_seasons(param='rns', ylabel='Radiação ($Wm^{-2}$)')

# Long wave radiation budget
boxplot_seasons(param='rnl', ylabel='Radiação ($Wm^{-2}$)')

# All wave radiation budget
boxplot_seasons(param='rn', ylabel='Radiação ($Wm^{-2}$)')


# %% TIME SERIES

# Data to long format
rad_plot = (radiation_data.melt(id_vars=['date', 'classes', 'season'],
                                value_vars=['rnl', 'rns', 'rn'],
                                var_name='type', value_name='rad'))

# Dataframe with mean values
radiation_means = (radiation_data.groupby(['date', 'classes', 'season'])
                   .mean().reset_index(drop=False))
radiation_means['date'] = pd.to_datetime(radiation_means.date)

# Dictionary of colors for seasons
season_colors = {'Spring':'#ff7f00', 'Summer':'#de2d26',
                 'Fall':'#33a02c', 'Winter':'#3182bd'}

# Seasons in order to be displayed
season_order = ['Spring', 'Summer', 'Fall', 'Winter']

# Crete grid
rad_time_grid = sns.FacetGrid(data=radiation_means, row='classes', aspect=6,
                              sharey=False, sharex=False, height=1.6)

# Scatterplot
rad_time_grid.map(sns.scatterplot, 'date', 'rn', 'season',
                  palette=season_colors, legend=True)

# Lineplot
rad_time_grid.map(sns.lineplot, 'date', 'rn', alpha=0.3, color='black',
                  zorder=0)

rad_time_grid.add_legend(ncol=4, loc='lower center', label_order=season_order,
                         labels=['Spring', 'Summer', 'Fall', 'Winter'])

rad_time_grid.set_xlabels(' ')
rad_time_grid.set_ylabels('Radiação ($Wm^{-2}$)')
