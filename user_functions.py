# -*- coding: utf-8 -*-
"""
FEDERAL UNIVERSITY OF SANTA CATARINA
TECHNOLOGICAL CENTER
DEPT. OF SANITARY AND ENVIRONMENTAL ENGINEERING
LABORATORY OF MARINE HYDRAULICS

Created on Sun Sep  4 13:30:30 2022
Author: Bruno Rech (b.rech@outlook.com)

SCRIPT 1/3 - AUXILIAR FUNCTIONS
"""

# %% INITIALIZATION

# Required libraries
import ee
import pandas as pd
import numpy as np

# GEE initialization
ee.Initialize()


# %% FUNCTION 01: GENERATE METADATA DATAFRAME

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


# %% FUNCTION 02: IDENTIFY TIME GAPS

# Identify temporal gaps
def month_gaps(dataframe, date_col, start, end):

    """
    Return the months without scenes.
    date_col: the name of the column with dates.
    """

    month_lack = []

    # Generate list of lists [month, year] with observed values
    months_observed = np.dstack(
        [pd.DatetimeIndex(dataframe[date_col]).month.to_list(),
         pd.DatetimeIndex(dataframe[date_col]).year.to_list()]).tolist()


    # Generate list of lists [month, year] with desired values
    months_desired = np.dstack(
        [pd.date_range(start=start, end=end, freq='m', inclusive='both')
         .month.to_list(),
         pd.date_range(start=start, end=end, freq='m', inclusive='both')
         .year.to_list()]).tolist()

    # Verify wether the desired values are within the observed ones
    for n in range(0, len(months_desired[0])):

        if months_desired[0][n] not in months_observed[0]:

            month_lack.append(months_desired[0][n])

    print(f'\nLacks in {len(month_lack)} months')

    return month_lack


# %% FUNCTION 03: CLOUD MASK

# Apply a cloud mask to the images
def cloud_mask(image):

    # Select cloud band
    # It is assigned 1 to the selected pixels and 0 to the others
    # Bit 6: 1 to clear sky and 0 to cloud or dilated cloud
    clear = image.select('QA_PIXEL').bitwiseAnd(1<<6).neq(0)

    return image.updateMask(clear)


# %% FUNCTION 04: RETRIEVE DAY OF YEAR B, E AND DECLINATION

# Calculate parameters B, E and declination
def declination(image):

    """
    This function calculates B, E and declination values and adds
    to the image metadata
    """

    # Retrieve month
    month = image.date().get('month')

    # Retrieve day
    day = image.date().get('day')

    # Calculate day of year (generates ee.Number)
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

    # Calculate E (in minutes)
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

    # Set metadata
    return image.set({'DAY_OF_YEAR':day_of_year, 'B':B, 'E':E,
                      'DECLINATION':declination})


# %% FUNCTION 05: REPROJECT TO SIRGAS 2000 UTM ZONE 22S

# Reproject L8 scenes
def to_31982(image):

    return image.reproject(crs='EPSG:31982', scale=30)


# %% FUNCTION 06: APPLY SCALE AND OFFSET FACTORS

# Apply scale factors to L8 reflectance and thermal bands
def scale_L8(image):

    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_band = image.select('ST_B10').multiply(0.00341802).add(149)

    return (image.addBands(optical_bands, overwrite=True)
            .addBands(thermal_band, overwrite=True))


# %% FUNCTION 07: CREATE PIXEL LAT/LONG BANDS

# Create bands with pixel coordinates
def pixels_coords(image):

    coords = image.pixelLonLat().rename(['long', 'lat'])

    return image.addBands(coords)


# %% FUNCTION 08: RETRIEVE SOLAR ZENITH ANGLE OVER A HORIZONTAL SURFACE

# Calculate theta_hor
def theta_hor(image):

    # Band with pixel latitudes
    lat = image.select('lat')

    # Declination
    declination = image.getNumber('DECLINATION')

    # Hour angle
    hour_angle = image.getNumber('HOUR_ANGLE')

    # Calculate theta
    theta_hor = ((lat.sin().multiply(declination.sin()))
                 .add(lat.cos()
                     .multiply(declination.cos())
                     .multiply(hour_angle.cos())))

    return image.addBands(theta_hor.rename('theta_hor'))


# %% FUNCTION 09: RETRIEVE SOLAR INCIDENCE ANGLE

# Calculate theta_rel
def theta_rel(image):

    # Band with pixel latitudes
    lat = image.select('lat')

    # Declination
    declination = image.getNumber('DECLINATION')

    # Hour angle
    hour_angle = image.getNumber('HOUR_ANGLE')

    # Slope band
    slope = image.select('slope')

    # Aspect band
    aspect = image.select('aspect')

    # Calculate theta_rel
    theta_rel = ((lat.sin().multiply(slope.cos()).multiply(declination.sin()))
                 .subtract(lat.cos().multiply(slope.sin())
                            .multiply(aspect.cos()).multiply(declination.sin()))
                 .add(lat.cos().multiply(slope.cos()).multiply(hour_angle.cos())
                      .multiply(declination.cos()))
                 .add(lat.sin().multiply(slope.sin()).multiply(aspect.cos())
                      .multiply(hour_angle.cos()).multiply(declination.cos()))
                 .add(aspect.sin().multiply(slope.sin())
                      .multiply(hour_angle.sin()).multiply(declination.cos())))

    return image.addBands(theta_rel.rename('theta_rel'))


# %% FUNCTION 10: RETRIEVE ATMOSPHERIC PRESSURE

# Calculate atmospheric pressure (kPa)
def atm_pressure(image):

    elev = image.select('elevation')

    atm_pressure = (((elev.multiply(-0.0065).add(293))
                     .divide(293)).pow(5.26)).multiply(101.3)

    return image.addBands(atm_pressure.rename('p_atm'))


# %% FUNCTION 11: ALBEDO RETRIEVAL

# Calculate albedo using model proposed by Angelini et al. (2021)
def albedo(image):

    # Select required bands
    b2 = image.select('SR_B2')
    b3 = image.select('SR_B3')
    b4 = image.select('SR_B4')
    b5 = image.select('SR_B5')
    b6 = image.select('SR_B6')
    b7 = image.select('SR_B7')

    # Apply model (equation)
    albedo = (b2.multiply(0.4739)
              .add(b3.multiply(-0.4372))
              .add(b4.multiply(0.1652))
              .add(b5.multiply(0.2831))
              .add(b6.multiply(0.1072))
              .add(b7.multiply(0.1029))
              .add(0.0366))

    return image.addBands(albedo.rename('albedo'))


# %% FUNCTION 12: SAVI AND LAI

# Calculate Soil-Adjusted Vegetation Index
def savi_lai(image):

    # Select required bands
    red = image.select('SR_B4')
    nir = image.select('SR_B5')

    # Set the value for L
    L = 0.5

    # Calculate SAVI
    savi = nir.subtract(red).divide(nir.add(red).add(L)).multiply(1 + L)

    # Calculate LAI
    raw_lai = savi.multiply(-1).add(0.69).divide(0.59).log().divide(-0.91)

    # LAI <= 3
    lai_lte3 = lai.lte(3)

    # Apply mask to keep the pixels <= 3 and attribute 3 to masked pixels
    lai = raw_lai.updateMask(lai_lte3).unmask(3)
