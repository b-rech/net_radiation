# -*- coding: utf-8 -*-
"""
FEDERAL UNIVERSITY OF SANTA CATARINA
TECHNOLOGICAL CENTER
DEPT. OF SANITARY AND ENVIRONMENTAL ENGINEERING
LABORATORY OF MARINE HYDRAULICS

Created on Sun Sep  4 13:30:30 2022
Author: Bruno Rech (b.rech@outlook.com)
"""

# %% INITIALIZATION

# Required libraries
import ee
import pandas as pd
import numpy as np

# GEE initialization
ee.Initialize()


# %% FUNCTION 01

# Function to generate a metadata dataframe
def list_info_df(properties_list):

    """
    This function selects important attributes from a metadata list and
    transforms it in a Pandas dataframe.
    """

    # Variable initialization
    ids = []
    cloudiness = []
    algorithm_sr = []
    algorithm_st = []
    time = []
    software_version = []
    path = []
    row = []

    # Information retrieval
    for j in range(0, len(properties_list)):

        ids.append(
            properties_list[j]['properties']['system:index'])

        cloudiness.append(
            properties_list[j]['properties']['CLOUDINESS'])

        algorithm_sr.append(
            properties_list[j]['properties']
            ['ALGORITHM_SOURCE_SURFACE_REFLECTANCE'])

        algorithm_st.append(
            properties_list[j]['properties']
            ['ALGORITHM_SOURCE_SURFACE_TEMPERATURE'])

        time.append(
            properties_list[j]['properties']['system:time_start'])

        software_version.append(
            properties_list[j]['properties']['L1_PROCESSING_SOFTWARE_VERSION'])

        path.append(
            properties_list[j]['properties']['WRS_PATH'])

        row.append(
            properties_list[j]['properties']['WRS_ROW'])


    # Dataframe's generation
    infos = pd.DataFrame({
        'id':ids,
        'cloudiness':cloudiness,
        'date':time,
        'software_version':software_version,
        'algorithm_sr':algorithm_sr,
        'algorithm_st':algorithm_st,
        'path':path,
        'row':row})

    infos['date'] = pd.to_datetime(infos['date'], unit='ms')

    return infos


# %% FUNCTION 02

# Function to identify temporal gaps
def month_gaps(dataframe, date_col, start, end):

    """
    Return the months without scenes.
    date_col: the name of the column with dates.
    """

    month_lack = []

    # Generates a list of lists [month, year] with observed values
    months_observed = np.dstack(
        [pd.DatetimeIndex(dataframe[date_col]).month.to_list(),
         pd.DatetimeIndex(dataframe[date_col]).year.to_list()]).tolist()


    # Generates a list of lists [month, year] with desired values
    months_desired = np.dstack(
        [pd.date_range(start=start, end=end, freq='m', inclusive='both')
         .month.to_list(),
         pd.date_range(start=start, end=end, freq='m', inclusive='both')
         .year.to_list()]).tolist()

    # Verifies wether the desired values are within the observed ones
    for n in range(0, len(months_desired[0])):

        if months_desired[0][n] not in months_observed[0]:

            month_lack.append(months_desired[0][n])

    print(f'\nLacks in {len(month_lack)} months')

    return month_lack
