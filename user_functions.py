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


# %% FUNCTION 03

# Calculates parameters B, E and declination

def declination(image):

    """
    This function calculates B, E and declination values and adds
    to the image metadata
    """

    # Retrieval of month number
    month = image.date().get('month')

    # Retrieval of day number
    day = image.date().get('day')

    # Calculate the day of year (generates ee.Number)
    day_of_year = ee.List(
        [ee.Algorithms.If(month.eq(1), day),
         ee.Algorithms.If(month.eq(2), day.add(31)),
         ee.Algorithms.If(month.eq(3), day.add(59)),
         ee.Algorithms.If(month.eq(4), day.add(90)),
         ee.Algorithms.If(month.eq(5), day.add(120)),
         ee.Algorithms.If(month.eq(6), day.add(151)),
         ee.Algorithms.If(month.eq(7), day.add(181)),
         ee.Algorithms.If(month.eq(8), day.add(212)),
         ee.Algorithms.If(month.eq(9), day.add(243)),
         ee.Algorithms.If(month.eq(10), day.add(273)),
         ee.Algorithms.If(month.eq(11), day.add(304)),
         ee.Algorithms.If(month.eq(12), day.add(334))]
        ).getNumber(month.subtract(1))

    # Calculate B (in radians)
    B = day_of_year.subtract(1).multiply(2*np.pi/365)

    # Calculate E
    E = ee.Number(229.2).multiply(
        ee.Number(0.000075)
        .add(B.cos().multiply(0.001868))
        .subtract(B.sin().multiply(0.032077))
        .subtract(B.multiply(2).cos().multiply(0.014615))
        .subtract(B.multiply(2).sin().multiply(0.04089)))

    # Calculate declination (Spencer Equation, in radians)
    declination = (ee.Number(0.006918)
                   .subtract(B.cos().multiply(0.399912))
                   .add(B.sin().multiply(0.070257))
                   .subtract(B.multiply(2).cos().multiply(0.006758))
                   .add(B.multiply(2).sin().multiply(0.000907))
                   .subtract(B.multiply(3).cos().multiply(0.002697))
                   .add(B.multiply(3).sin().multiply(0.00148)))

    return image.set({'DAY_OF_YEAR':day_of_year, 'B':B, 'E':E,
                      'DECLINATION':declination})


# %% FUNCTION 04

# Reproject L8 scenes to WGS84 (EPSG 4326)
def to_4326(image):

    return image.reproject(crs='EPSG:4326', scale=30)


# %% FUNCTION 05

# Apply scale factors to L8 reflectance and thermal bands
def scale_L8(image):

    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_band = image.select('ST_B10').multiply(0.00341802).add(149)

    return (image.addBands(optical_bands, overwrite=True)
            .addBands(thermal_band, overwrite=True))


# %% FUNCTION 06

# Create bands with pixel coordinates
def pixels_coords(image):

    coords = image.pixelCoordinates('EPSG:4326').rename(['long', 'lat'])

    return image.addBands(coords)
