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
import numpy as np
import pandas as pd
from scipy import stats
import scikit_posthocs as sp
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
radiation_data['date'] = pd.to_datetime(radiation_data.date)

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

# Save figure
# plt.savefig('temporal_availability.tif', dpi=300)


# %% PLOT AND TABLE - SPECTRAL SIGNATURES

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

# Save figure
# plt.savefig('spectral_signatures.tif', dpi=300)

### TABLE

# Calculate means
spectral_signatures = (spectral_to_plot.groupby(['classes', 'bands'])
                       .mean().reset_index())

# Tranform to long format
spectral_signatures = spectral_signatures.pivot(index='classes',
                                                columns='bands',
                                                values='reflect').reset_index()

# Save to csv
# spectral_signatures.to_csv('generated_data\\spectral_signatures.csv', sep=';',
#                            decimal=',', index=False)


# %% TABLE - NUMBER OF OBSERVATIONS

# Calculate the sum of observations
observations = (radiation_data
                .groupby(['classes', 'season'])
                .count()
                .reset_index())

# Format table (totals per season)
observations = observations.pivot(index='classes',columns='season',
                                  values='date')

# Totals
observations['total'] = observations.sum(axis=1)
observations['total_percent'] = 100*observations.total/observations.total.sum()

observations = observations.reset_index()

# observations.to_csv('generated_data\\observations_count.csv', sep=';',
#                     decimal=',', index=False)


# %% TABLE - MEAN ALBEDOS

# Calculate means
albedo = radiation_data.groupby('classes').mean()

# Calculate standard deviation
albedo['sd'] = radiation_data.groupby('classes').std()['albedo']

# Create dataframe
albedo = albedo[['albedo', 'sd']]

# Calculate coefficient of variation
albedo['var_coef'] = albedo.sd/albedo.albedo

# Save to csv
# albedo.to_csv('generated_data\\albedo_means.csv', decimal=',',
#               sep=';')

# %% PLOT - ALBEDO


sns.set_theme(style='whitegrid')

classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

# Seasons in order to be displayed
season_order = ['Spring', 'Summer', 'Fall', 'Winter']

# Create plot
sns.pointplot(data=radiation_data, x='classes', y='albedo', markers='.',
              join=False, errorbar=('pi', 95), order=classes, errwidth=1,
                  capsize=.4, n_boot=500000, color='black')

sns.catplot(data=radiation_data, x='classes', y='albedo', kind='bar',
              errorbar=('pi', 95), order=classes,
                  n_boot=500000, facecolor='#d9d9d9', edgecolor='black',
                  capsize=.4, errwidth=1)



# Configure axes labels
spectral_grid.set_xlabels('Banda')
spectral_grid.set_ylabels('Reflectância')

rad_mean_scenes = (radiation_data.groupby(['date', 'classes'])
                   .mean().reset_index())

albedo_grid = sns.FacetGrid(data=radiation_data, row='classes', aspect=6,
                            sharey=False, sharex=False, height=1.6)

albedo_grid.map(sns.lineplot, 'date', 'albedo', errorbar=('pi', 90), marker='.')

# Lineplot
albedo_grid.map(sns.lineplot, 'date', 'albedo', alpha=0.3, color='black',
                  zorder=0)


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
        ax.set_title(title, fontweight='bold')

    # Set x tick labels
    season_boxplot.set_xticklabels(['Primavera', 'Verão', 'Outono', 'Inverno'],
                                   fontsize=11)

    return season_boxplot


# %% SEASONAL BOXPLOTS

# Short wave radiation budget
boxplot_seasons(param='rns', ylabel='Radiação ($Wm^{-2}$)')

plt.savefig('boxplot_rns.tif', dpi=300)

# Long wave radiation budget
boxplot_seasons(param='rnl', ylabel='Radiação ($Wm^{-2}$)')

# All wave radiation budget
boxplot_seasons(param='rn', ylabel='Radiação ($Wm^{-2}$)')
plt.savefig('boxplot_rn.tif', dpi=300)

# %% TABLE - SHORTWAVE NET RADIATION PER SEASON

# Get means
rad_seasonal_means = (radiation_data.groupby(['classes', 'season'])
                      .mean().reset_index())

# Transform table
sw_rad_seasonal_means = (rad_seasonal_means[['classes', 'season', 'rns']]
                         .pivot(index='season',columns='classes',
                                values='rns'))

# # Save to csv
# sw_rad_seasonal_means.to_csv('generated_data\\mean_net_shortwave.csv',
#                               sep=';', decimal=',')


# %% TABLE - LONGWAVE NET RADIATION PER SEASON

# Get means
rad_seasonal_means = (radiation_data.groupby(['classes', 'season'])
                      .mean().reset_index())

# Transform table
lw_rad_seasonal_means = (rad_seasonal_means[['classes', 'season', 'rnl']]
                         .pivot(index='season',columns='classes',
                                values='rnl'))

# # Save to csv
# lw_rad_seasonal_means.to_csv('generated_data\\mean_net_longwave.csv',
#                               sep=';', decimal=',')


# %% TABLE - ALL-WAVE NET RADIATION PER SEASON

# Get means
rad_seasonal_means = (radiation_data.groupby(['classes', 'season'])
                      .mean().reset_index())

# Transform table
rn_rad_seasonal_means = (rad_seasonal_means[['classes', 'season', 'rn']]
                         .pivot(index='season',columns='classes',
                                values='rn'))

# # Save to csv
# rn_rad_seasonal_means.to_csv('generated_data\\mean_net_allwave.csv',
#                              sep=';', decimal=',')



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
                  palette=season_colors, legend=True, edgecolor='black')

# Lineplot
rad_time_grid.map(sns.lineplot, 'date', 'rn', alpha=0.3, color='black',
                  zorder=0)

rad_time_grid.add_legend(ncol=4, loc='lower center', label_order=season_order)


new_labels = ['Primavera', 'Verão', 'Outono', 'Inverno']
for t, l in zip(rad_time_grid._legend.texts, new_labels):
    t.set_text(l)

rad_time_grid.set_xlabels(' ')
rad_time_grid.set_ylabels('Radiação ($Wm^{-2}$)')


# %%

# Define function
def hist_seasons(param, ylabel):

    sns.set_style('whitegrid')

    classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

    # Facecolor and edgecolor
    rn_boxplot_fc = 'black'
    rn_boxplot_ec = 'black'

    # Create plot - boxplot by season
    season_hist = sns.FacetGrid(data=radiation_data, col='classes',
                                sharex=False, sharey=False, row='season')

    # Dictionary of colors for seasons
    season_colors = {'Spring':'#ff7f00', 'Summer':'#de2d26',
                     'Fall':'#33a02c', 'Winter':'#3182bd'}

    # Seasons in order to be displayed
    season_order = ['Spring', 'Summer', 'Fall', 'Winter']

    # Map plots to grid
    season_hist.map(sns.histplot, param)


    return season_hist


# %% KRUSKAL-WALIS TESTS

# Classes
classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

# Radiation data means per scene
mean_rad_data = (radiation_data.groupby(['date', 'classes', 'season'])
                 .mean().reset_index())


statistics = []
pvalues = []

for classe in classes:

    # Filtered data for the selected class
    data = mean_rad_data[mean_rad_data.classes==classe]

    # Spring samples
    spring = data[data.season=='Spring']['rn'].tolist()

    # Summer samples
    summer = data[data.season=='Summer']['rn'].tolist()

    # Fall samples
    fall = data[data.season=='Fall']['rn'].tolist()

    # Winter samples
    winter = data[data.season=='Winter']['rn'].tolist()

    stats.kruskal(spring, summer, fall, winter)

    # Performs test
    kw_test = stats.kruskal(spring, summer, fall, winter, nan_policy='raise')

    print(kw_test)
    # Retrieve info
    statistics.append(kw_test.statistic)
    pvalues.append(kw_test.pvalue)

# Create dataframe with results
kw_test_results = pd.DataFrame({'classes':classes,
                                'statistic':statistics,
                                'pvalue':pvalues})

# Save to csv
# kw_test_results.to_csv('generated_data\\kruskal_wallis_results.csv', sep=';',
#                         decimal=',', index=False)

dunn_test_results = pd.DataFrame(
    columns=['Spring', 'Summer', 'Fall', 'Winter', 'classes'])

for classe in classes:

    data = mean_rad_data[mean_rad_data.classes==classe]

    test_output = (sp.posthoc_dunn(data, val_col='rn', group_col='season',
                                  p_adjust='bonferroni')
                   .reindex(['Spring', 'Summer', 'Fall', 'Winter'])
                   .reset_index())

    test_output['classes'] = [classe]*4

    dunn_test_results = pd.concat([dunn_test_results, test_output])

# dunn_test_results.to_csv('generated_data\\dunn_test_results.csv', sep=';',
#                           decimal=',', index=False)
