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
metadata['year'] = metadata.date.dt.year

# Spectral data from samples
spectral_data = pd.read_csv('generated_data\\spectral_data.csv', sep=';',
                            decimal=',')

# Radiation data from samples
radiation_data = pd.read_csv('generated_data\\radiation_data.csv', sep=';',
                            decimal=',')
radiation_data['date'] = pd.to_datetime(radiation_data.date)

# Removal of a single observation with Rn < 0 (error)
radiation_data = radiation_data[radiation_data.rn > 0]

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
                    palette=['#de2d26', '#33a02c', '#3182bd', '#ff7f00'],
                    hue_order=['Summer', 'Fall', 'Winter', 'Spring'],
                    edgecolor='black')

# Legend handles
handles = [Rectangle(xy=(0, 0), height=1, width=1, facecolor=color,
                     edgecolor='black', linewidth=.5)
           for color in ['#de2d26', '#33a02c', '#3182bd', '#ff7f00']]

# Add legend
plot.legend(handles=handles, labels=['Verão', 'Outono', 'Inverno', 'Primavera'],
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


# Median points
spectral_grid.map(sns.pointplot, 'bands', 'reflect', markers='.', join=False,
                  estimator='median', errorbar=('pi', 95),
                  order=spectral_cols, errwidth=1,
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

# Calculate medians
spectral_signatures = (spectral_to_plot.groupby(['classes', 'bands'])
                       .median().reset_index())

# Tranform to long format
spectral_signatures = spectral_signatures.pivot(index='classes',
                                                columns='bands',
                                                values='reflect').reset_index()

# Save to csv
# spectral_signatures.to_csv('generated_data\\spectral_signatures.csv', sep=';',
#                             decimal=',', index=False)


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


# %% TABLE - MEAN EMISSIVITY

# Calculate means
emiss = radiation_data.groupby('classes').mean()

# Calculate standard deviation
emiss['sd'] = radiation_data.groupby('classes').std()['emiss']

# Create dataframe
emiss = emiss[['emiss', 'sd']]

# Calculate coefficient of variation
emiss['var_coef'] = emiss.sd/emiss.emiss

# Save to csv
# emiss.to_csv('generated_data\\emiss_means.csv', decimal=',',
#               sep=';')


# %% TABLE - MEAN TEMPERATURE

# Calculate means
temp = radiation_data.groupby('classes').mean() - 273.15

# Calculate standard deviation
temp['sd'] = radiation_data.groupby('classes').std()['temp']

# Create dataframe
temp = temp[['temp', 'sd']]

# Calculate coefficient of variation
temp['var_coef'] = temp.sd/temp.temp

# Save to csv
# temp.to_csv('generated_data\\temperature_means.csv', decimal=',',
#             sep=';')


# %% SEASONAL BOXPLOT FUNCTION

# Define function
def boxplot_seasons(param, ylabel):

    sns.set_style('whitegrid')

    classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

    # Facecolor and edgecolor
    rn_boxplot_ec = 'black'

    # Create plot - boxplot by season
    season_boxplot = sns.FacetGrid(data=radiation_data, col='classes',
                                   sharex=True, sharey=True,col_wrap=3,
                                   col_order=classes, aspect=1, height=2.5)

    # Dictionary of colors for seasons
    season_colors = {'Spring':'#ff7f00', 'Summer':'#de2d26',
                     'Fall':'#33a02c', 'Winter':'#3182bd'}

    # Seasons in order to be displayed
    season_order = ['Summer', 'Fall', 'Winter', 'Spring']

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
                                  'markersize':3})


    # Configure axes labels
    season_boxplot.set_xlabels(' ')
    season_boxplot.set_ylabels(ylabel, fontsize=10)

    # Add classes titles
    for ax, title in zip(season_boxplot.axes.flatten(), classes):
        ax.set_title(title, fontweight='bold')
        ax.get_yaxis().set_minor_locator(tkr.AutoMinorLocator())
        ax.grid(visible=True, which='minor', linewidth=0.4)

    # Set x tick labels
    season_boxplot.set_xticklabels([None]*4)

    season_boxplot.add_legend(ncol=1, loc=(.8, .10), title='Estações')

    new_labels = ['Verão', 'Outono', 'Inverno', 'Primavera']
    for t, l in zip(season_boxplot._legend.texts, new_labels):
        t.set_text(l)

    return season_boxplot


# %% SEASONAL BOXPLOTS

# Short wave radiation budget
boxplot_seasons(param='rns', ylabel='$R_n^S ~ (Wm^{-2})$')
plt.tight_layout()
plt.savefig('boxplot_rns.tif', dpi=300)

# Long wave radiation budget
boxplot_seasons(param='rnl', ylabel='$R_n^L ~ (Wm^{-2})$')
plt.tight_layout()
plt.savefig('boxplot_rnl.tif', dpi=300)

# All wave radiation budget
boxplot_seasons(param='rn', ylabel='$R_n ~ (Wm^{-2})$')
plt.tight_layout()
plt.savefig('boxplot_rn.tif', dpi=300)

# %% TABLE - SHORTWAVE NET RADIATION PER SEASON

# Get medians
rad_seasonal_medians = (radiation_data.groupby(['classes', 'season'])
                      .median().reset_index())

# Transform table
sw_rad_seasonal_medians = (rad_seasonal_medians[['classes', 'season', 'rns']]
                           .pivot(index='season',columns='classes',
                                  values='rns'))

# # Save to csv
# sw_rad_seasonal_medians.to_csv('generated_data\\median_net_shortwave.csv',
#                                 sep=';', decimal=',')


# %% TABLE - LONGWAVE NET RADIATION PER SEASON

# Get medians
rad_seasonal_medians = (radiation_data.groupby(['classes', 'season'])
                        .median().reset_index())

# Transform table
lw_rad_seasonal_medians = (rad_seasonal_medians[['classes', 'season', 'rnl']]
                           .pivot(index='season',columns='classes',
                                  values='rnl'))

# Save to csv
# lw_rad_seasonal_medians.to_csv('generated_data\\median_net_longwave.csv',
#                                 sep=';', decimal=',')


# %% TABLE - ALL-WAVE NET RADIATION PER SEASON

# Get medians
rad_seasonal_medians = (radiation_data.groupby(['classes', 'season'])
                        .median().reset_index())

# Transform table
rn_rad_seasonal_medians = (rad_seasonal_medians[['classes', 'season', 'rn']]
                           .pivot(index='season',columns='classes',
                                  values='rn'))

# Save to csv
# rn_rad_seasonal_medians.to_csv('generated_data\\median_net_allwave.csv',
#                                 sep=';', decimal=',')


# %% TABLE - ANNUAL WEIGHTED MEANS

# Define function
def weighted_means(data, param, weights):

    data['weight'] = [1/weights[x] for x in data.season]

    data['param_weighted'] = data[param]*data.weight

    rad_weight_means = (data.groupby(['classes']).sum().reset_index())

    rad_weight_means['weighted_mean'] = (rad_weight_means.param_weighted/
                                         rad_weight_means.weight)

    return rad_weight_means[['classes', 'weighted_mean']]

# Weights are the number os images
weights = {'Summer':13, 'Fall':36, 'Winter':37, 'Spring':7}

# Shortwave means
shortwave_means = weighted_means(radiation_data, 'rns', weights)

# Longwave means
longwave_means = weighted_means(radiation_data, 'rnl', weights)

# All wave means
allwave_means = weighted_means(radiation_data, 'rn', weights)

# Dataframe with all means
annual_means = pd.DataFrame({'classes':shortwave_means.classes,
                             'rns':shortwave_means.weighted_mean,
                             'rnl':longwave_means.weighted_mean,
                             'rn':allwave_means.weighted_mean})

# Save to csv
# annual_means.to_csv('generated_data\\annual_weighted_means.csv',
#                     sep=';', decimal=',')

# %% TIME SERIES

def time_series_plot(data, param, ylabel):

    sns.set_style('whitegrid')

    classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

    # Dataframe with median values
    radiation_medians = (data.groupby(['date', 'classes', 'season'])
                         .median().reset_index(drop=False))
    radiation_medians['date'] = pd.to_datetime(radiation_medians.date)

    # Dictionary of colors for seasons
    season_colors = {'Spring':'#ff7f00', 'Summer':'#de2d26',
                     'Fall':'#33a02c', 'Winter':'#3182bd'}

    # Seasons in order to be displayed
    season_order = ['Summer', 'Fall', 'Winter', 'Spring']

    # Crete grid
    rad_time_grid = sns.FacetGrid(data=radiation_medians, row='classes',
                                  aspect=3.5, sharey=True,
                                  sharex=False, height=1.6)

    # Scatterplot
    rad_time_grid.map(sns.scatterplot, 'date', param, 'season',
                      palette=season_colors, legend=True, edgecolor='black')

    rad_time_grid.add_legend(ncol=4, loc='lower center',
                             label_order=season_order)

    # Add classes titles
    for ax, title in zip(rad_time_grid.axes.flatten(), classes):
        ax.set_title(title, fontweight='bold')

    new_labels = ['Verão', 'Outono', 'Inverno', 'Primavera']
    for t, l in zip(rad_time_grid._legend.texts, new_labels):
        t.set_text(l)

    rad_time_grid.set_xlabels(' ')
    rad_time_grid.set_ylabels(ylabel)

    return rad_time_grid

# Data to long format
rad_plot = (radiation_data.melt(id_vars=['date', 'classes', 'season'],
                                value_vars=['rnl', 'rns', 'rn'],
                                var_name='type', value_name='rad'))

time_series_plot(data=radiation_data, param='rns',
                 ylabel='$R_n^S ~ (Wm^{-2})$')
plt.tight_layout()
# plt.savefig('time_series_rns.tif', dpi=300)

time_series_plot(data=radiation_data, param='rnl',
                 ylabel='$R_n^L ~ (Wm^{-2})$')
plt.tight_layout()
# plt.savefig('time_series_rnl.tif', dpi=300)

time_series_plot(data=radiation_data, param='rn',
                 ylabel='$R_n ~ (Wm^{-2})$')
plt.tight_layout()
# plt.savefig('time_series_rn.tif', dpi=300)


# %% RELATION BETWEEN SHORT AND LONG BUDGETS

sns.set_style('whitegrid')

# Median values per scene
rad_image_medians = (radiation_data.groupby(['date', 'classes', 'season'])
                     .median().reset_index())

# Classes
classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

# List to save correlations
corr = []

# Spearman correlations
for classe in classes:

    # Select class
    data = rad_image_medians[rad_image_medians.classes==classe]

    # Append correlation to list
    corr.append(data[['rns', 'rnl']].corr(method='spearman').iloc[0, 1])

# Correlation grid
corr_grid = sns.FacetGrid(data=rad_image_medians, col='classes', col_wrap=3,
                          col_order=classes, sharex=False, sharey=False)

# Map plots
corr_grid.map(sns.regplot, 'rnl', 'rns', ci=None,
              line_kws={'color':'#e31a1c', 'linewidth':1, 'linestyle':'--'},
              scatter_kws={'facecolor':'#deebf7', 'edgecolor':'#084594'})

# Set xy labels
corr_grid.set_xlabels('$R_n^L ~ (Wm^{-2})$')
corr_grid.set_ylabels('$R_n^S ~ (Wm^{-2})$')

# Add classes titles and correlation coefficients
for ax, title, cor in zip(corr_grid.axes.flatten(), classes, corr):
    ax.set_title(title, fontweight='bold')
    ax.annotate('$r_S$ = ' + str(format(cor, '.2f')),
                xy=(0.7, 0.85), xycoords='axes fraction',
                bbox=dict(boxstyle="square", fc="w", ec='black'))

plt.tight_layout()

# plt.savefig('correlation_rad.tif', dpi=300)

# %% KRUSKAL-WALIS TESTS

# Function for Kruskal-Wallis test
def kruskal_wallis(input_data, param):

    # Classes
    classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

    # Radiation data medians per scene
    median_rad_data = (input_data.groupby(['date', 'classes', 'season'])
                     .median().reset_index())

    statistics = []
    pvalues = []

    for classe in classes:

        # Filtered data for the selected class
        data = median_rad_data[median_rad_data.classes==classe]

        # Spring samples
        spring = data[data.season=='Spring'][param].tolist()

        # Summer samples
        summer = data[data.season=='Summer'][param].tolist()

        # Fall samples
        fall = data[data.season=='Fall'][param].tolist()

        # Winter samples
        winter = data[data.season=='Winter'][param].tolist()

        # Performs test
        kw_test = stats.kruskal(spring, summer, fall, winter, nan_policy='raise')

        # Retrieve info
        statistics.append(kw_test.statistic)
        pvalues.append(kw_test.pvalue)

    # Create dataframe with results
    kw_test_results = pd.DataFrame({'classes':classes,
                                    'statistic':statistics,
                                    'pvalue':pvalues})

    return kw_test_results


# Function for Dunn test
def dunn_test(input_data, param):

    # Classes
    classes = ['DUN','FOD', 'LCP', 'LCR', 'RAA', 'SIL', 'URB', 'VHE']

    dunn_test_results = pd.DataFrame(
        columns=['Spring', 'Summer', 'Fall', 'Winter', 'classes'])

    # Radiation data medians per scene
    median_rad_data = (input_data.groupby(['date', 'classes', 'season'])
                     .median().reset_index())

    for classe in classes:

        data = median_rad_data[median_rad_data.classes==classe]

        test_output = (sp.posthoc_dunn(data, val_col=param, group_col='season',
                                      p_adjust='bonferroni')
                       .reindex(['Spring', 'Summer', 'Fall', 'Winter'])
                       .reset_index())

        test_output['classes'] = [classe]*4

        dunn_test_results = pd.concat([dunn_test_results, test_output])

    return dunn_test_results


# Kruskal-Wallis: shortwave radiation
kw_test_rns = kruskal_wallis(radiation_data, 'rns')
# kw_test_rns.to_csv('generated_data\\kw_test_rns.csv', sep=';',
#                    decimal=',', index=False)

# Dunn: shortwave radiation
dunn_test_rns = dunn_test(radiation_data, 'rns')
# dunn_test_rns.to_csv('generated_data\\dunn_test_rns.csv', sep=';',
#                      decimal=',', index=False)


# Kruskal-Wallis: longwave radiation
kw_test_rnl = kruskal_wallis(radiation_data, 'rnl')
# kw_test_rnl.to_csv('generated_data\\kw_test_rnl.csv', sep=';',
#                     decimal=',', index=False)

# Dunn: longwave radiation
dunn_test_rnl = dunn_test(radiation_data, 'rnl')
# dunn_test_rnl.to_csv('generated_data\\dunn_test_rnl.csv', sep=';',
#                       decimal=',', index=False)


# Kruskal-Wallis: allwave radiation
kw_test_rn = kruskal_wallis(radiation_data, 'rn')
# kw_test_rn.to_csv('generated_data\\kw_test_rn.csv', sep=';',
#                    decimal=',', index=False)

# Dunn: allwave radiation
dunn_test_rn = dunn_test(radiation_data, 'rn')
# dunn_test_rn.to_csv('generated_data\\dunn_test_rn.csv', sep=';',
#                     decimal=',', index=False)
