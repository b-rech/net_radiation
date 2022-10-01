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
import geemap

# GEE initialization
ee.Initialize()


# %% CLOUD MASK

# Apply a cloud mask to the images
def cloud_mask(image):

    # Select cloud band
    # It is assigned 1 to the selected pixels and 0 to the others
    # Bit 6: 1 to clear sky and 0 to cloud or dilated cloud
    clear = image.select('QA_PIXEL').bitwiseAnd(1<<6)

    return image.updateMask(clear)


# %% APPLY SCALE AND OFFSET FACTORS

# Apply scale factors to L8 reflectance and thermal bands
def scale_L8(image):

    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal_band = image.select('ST_B10').multiply(0.00341802).add(149)

    return (image.addBands(optical_bands, overwrite=True)
            .addBands(thermal_band, overwrite=True))


# %% CREATE PIXEL LAT/LONG BANDS

# Create bands with pixel coordinates (radians)
def pixels_coords(image):

    coords = image.pixelLonLat().multiply(np.pi/180).rename(['long', 'lat'])

    return image.addBands(coords)


# %% RETRIEVE DAY OF YEAR B, E AND DECLINATION

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


# %% RETRIEVE SOLAR ZENITH ANGLE COSINE OVER A HORIZONTAL SURFACE

# Calculate cos_theta_hor
def cos_theta_hor(image):

    # Band with pixel latitudes (radians)
    lat = image.select('lat')

    # Declination
    declination = image.getNumber('DECLINATION')

    # Hour angle
    hour_angle = image.getNumber('HOUR_ANGLE')

    # Calculate cosine of theta
    cos_theta_hor = (lat.sin().multiply(declination.sin())
                     .add(lat.cos()
                     .multiply(declination.cos())
                     .multiply(hour_angle.cos())))

    return image.addBands(cos_theta_hor.rename('cos_theta_hor'))


# %% RETRIEVE SOLAR INCIDENCE ANGLE COSINE

# Calculate theta_rel
def cos_theta_rel(image):

    # Band with pixel latitudes (radians)
    lat = image.select('lat')

    # Declination
    declination = image.getNumber('DECLINATION')

    # Hour angle
    hour_angle = image.getNumber('HOUR_ANGLE')

    # Slope band
    slope = image.select('slope')

    # Aspect band
    aspect = image.select('aspect')

    # Calculate cos_theta_rel
    cos_theta_rel = ((lat.sin().multiply(slope.cos()).multiply(declination.sin()))
                 .subtract(lat.cos().multiply(slope.sin())
                            .multiply(aspect.cos()).multiply(declination.sin()))
                 .add(lat.cos().multiply(slope.cos()).multiply(hour_angle.cos())
                      .multiply(declination.cos()))
                 .add(lat.sin().multiply(slope.sin()).multiply(aspect.cos())
                      .multiply(hour_angle.cos()).multiply(declination.cos()))
                 .add(aspect.sin().multiply(slope.sin())
                      .multiply(hour_angle.sin()).multiply(declination.cos())))

    return image.addBands(cos_theta_rel.rename('cos_theta_rel'))


# %% RETRIEVE ATMOSPHERIC PRESSURE

# Calculate atmospheric pressure (kPa)
def atm_pressure(image):

    # Select elevation band
    elev = image.select('elevation')

    # Calculate pressure (kPa)
    atm_pressure = (((elev.multiply(-0.0065).add(293))
                     .divide(293)).pow(5.26)).multiply(101.3)

    return image.addBands(atm_pressure.rename('p_atm'))


# %% PRECIPITABLE WATER

# Function to calculate precipitable water
def prec_water(image):

    # Retrieve saturation vapor pressure and relative humidity
    sat_vp = image.getNumber('SAT_VP')
    rel_hum = image.getNumber('REL_HUM')

    # Retrieve atmospheric pressure
    p = image.select('p_atm')

    # Calculate actual vapor pressure
    act_vp = (p.multiply(3.15E-5)
              .add(p.pow(-1).multiply(-0.074))
              .add(1.0016)
              .multiply(sat_vp).multiply(rel_hum))

    # Calculate precipitable water
    prec_w = act_vp.multiply(p).multiply(0.14).add(2.1)

    # Add band 'prec_water' to the image
    return image.addBands(prec_w.rename('prec_water'))


# %% ATMOSPHERIC TRANSMISSIVITY

# Calculates atmospheric transmissivity
def atm_trans(image):

    # Retrieve required parameters:

    # Atmospheric pressure
    p = image.select('p_atm')

    # Precipitable water
    w = image.select('prec_water')

    # Cosine of solar zenith angle over a horizontal surface
    c_theta_hor = image.select('cos_theta_hor')

    # Turbidity coefficient assumed to be 1
    kt = 1

    # Retrieve atmospheric transmissivity
    atm_trans = ((p.multiply(-0.00146).divide(c_theta_hor.multiply(kt))
                  .add(w.divide(c_theta_hor).pow(0.4).multiply(-0.075))).exp()
                 .multiply(0.627)).add(0.35)

    # Add calculated band to image
    return image.addBands(atm_trans.rename('atm_trans'))


# %% DOWNWARD SHORTWAVE RADIATION

# Calculates downward shortwave radiation
def dw_sw_rad(image):

    # Solar constant
    G = 1367 # W/m²

    # Cosine of solar incidence angle
    c_theta_rel = image.select('cos_theta_rel')

    # Atmospheric transmissivity
    trans = image.select('atm_trans')

    # Earth-Sun distance
    d = image.getNumber('EARTH_SUN_DISTANCE')

    # Calculate incident radiation
    dw_sw_rad = c_theta_rel.multiply(trans).multiply(G).divide(d.pow(2))

    return image.addBands(dw_sw_rad.rename('dw_sw_rad'))


# %% ALBEDO RETRIEVAL

# Calculate albedo using model proposed by Angelini et al. (2021)
def get_albedo(image):

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


# %% UPWARD SHORTWAVE RADIATION

# Calculates upward shortwave radiation
def up_sw_rad(image):

    # Retrieve reflected radiation
    up_sw_rad = image.select('dw_sw_rad').multiply(image.select('albedo'))

    return image.addBands(up_sw_rad.rename('up_sw_rad'))


# %% NET SHORTWAVE RADIATION

# Calculates shortwave radiation budget
def net_sw_rad(image):

    # Upward and downward fluxes
    up = image.select('up_sw_rad')
    down = image.select('dw_sw_rad')

    # Net shortwave radiation
    budget = down.subtract(up)

    return image.addBands(budget.rename('net_sw_rad'))


# %% ATMOSPHERIC EMISSIVITY

# Calculates atmospheric emissivity
def atm_emiss(image):

    # Select atmospheric transmissivity
    trans = image.select('atm_trans')

    # Retrieve atmospheric emissivity
    atm_emiss = trans.log().multiply(-1).pow(0.09).multiply(0.85)

    return image.addBands(atm_emiss.rename('atm_emiss'))


# %% DOWNWARD LONGWAVE RADIATION

# Calculates incident longwave radiation flux
def dw_lw_rad(image):

    # Select temperature band (scaled to K)
    temp = image.select('ST_B10')

    # Select atmospheric emissivity band
    emiss = image.select('atm_emiss')

    # Apply Stefan-Boltzmann equation
    dw_lw_rad = temp.pow(4).multiply(emiss).multiply(5.67E-8)

    # Add band to scene
    return image.addBands(dw_lw_rad.rename('dw_lw_rad'))


# %% UPWARD LONGWAVE RADIATION

# Calculates the upward longwave radiation
def up_lw_rad(image):

    # Select temperature band (scaled to K)
    temp = image.select('ST_B10')

    # Select emissivity band
    emiss = image.select('emiss')

    # Apply Stefan-Boltzmann equation
    up_lw_rad = temp.pow(4).multiply(emiss).multiply(5.67E-8)

    # Add band to scene
    return image.addBands(up_lw_rad.rename('up_lw_rad'))


# %% NET LONGWAVE RADIATION

# Calculates longwave radiation budget
def net_lw_rad(image):

    # Upward and downward fluxes
    up = image.select('up_lw_rad')
    down = image.select('dw_lw_rad')

    # Net longwave radiation
    budget = down.subtract(up)

    return image.addBands(budget.rename('net_lw_rad'))


# %% ALL-WAVE NET RADIATION

# Calculate all-wave radiation budget
def all_wave_rn(image):

    # Downward shortwave radiation
    dw_sw = image.select('dw_sw_rad')

    # Upward shortwave radiation
    up_sw = image.select('up_sw_rad')

    # Downward longwave radiation
    dw_lw = image.select('dw_lw_rad')

    # Upward longwave radiation
    up_lw = image.select('up_lw_rad')

    # Surface emissivity
    emiss = image.select('emiss')

    # Instantaneous all-wave net radiation
    Rn = dw_sw.subtract(up_sw).add(emiss.multiply(dw_lw)).subtract(up_lw)

    return image.addBands(Rn.rename('Rn'))


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
