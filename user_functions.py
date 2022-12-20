# -*- coding: utf-8 -*-
"""
FEDERAL UNIVERSITY OF SANTA CATARINA
TECHNOLOGICAL CENTER
DEPT. OF SANITARY AND ENVIRONMENTAL ENGINEERING
LABORATORY OF MARINE HYDRAULICS

Created on 2022/09/03
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


# %% CLOUD MASK

def cloud_mask(image):

    # Select cloud band
    # It is assigned 1 to the selected pixels and 0 to the others
    # Bit 6: 1 to clear sky and 0 to cloud or dilated cloud
    clear = image.select('QA_PIXEL').bitwiseAnd(1<<6)

    return image.updateMask(clear)


# %% APPLY SCALE AND OFFSET FACTORS

def scale_L8(image):

    # Scale and offset optical bands
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)

    # Scale and offset thermal band (LST)
    thermal_band = image.select('ST_B10').multiply(0.00341802).add(149)

    return (image.addBands(optical_bands, overwrite=True)
            .addBands(thermal_band, overwrite=True))


# %% CREATE PIXEL LAT/LONG BANDS

def pixels_coords(image):

    # Generate a band with pixel coordinates (radians)
    coords = image.pixelLonLat().multiply(np.pi/180).rename(['long', 'lat'])

    return image.addBands(coords)


# %% RETRIEVE DAY OF YEAR B, E AND DECLINATION

def declination(image):

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

def atm_pressure(image):

    # Select elevation band
    elev = image.select('elevation')

    # Calculate pressure (kPa)
    atm_pressure = (((elev.multiply(-0.0065).add(293))
                     .divide(293)).pow(5.26)).multiply(101.3)

    return image.addBands(atm_pressure.rename('p_atm'))


# %% PRECIPITABLE WATER

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

    # Calculate precipitable water (mm)
    prec_w = act_vp.multiply(p).multiply(0.14).add(2.1)

    # Add band 'prec_water' to the image
    return image.addBands(prec_w.rename('prec_water'))


# %% ATMOSPHERIC TRANSMISSIVITY

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

def up_sw_rad(image):

    # Retrieve reflected radiation
    up_sw_rad = image.select('dw_sw_rad').multiply(image.select('albedo'))

    return image.addBands(up_sw_rad.rename('up_sw_rad'))


# %% NET SHORTWAVE RADIATION

def net_sw_rad(image):

    # Upward and downward fluxes
    up = image.select('up_sw_rad')
    down = image.select('dw_sw_rad')

    # Net shortwave radiation
    budget = down.subtract(up)

    return image.addBands(budget.rename('net_sw_rad'))


# %% ATMOSPHERIC EMISSIVITY

def atm_emiss(image):

    # Select atmospheric transmissivity
    trans = image.select('atm_trans')

    # Retrieve atmospheric emissivity
    atm_emiss = trans.log().multiply(-1).pow(0.09).multiply(0.85)

    return image.addBands(atm_emiss.rename('atm_emiss'))


# %% DOWNWARD LONGWAVE RADIATION

def dw_lw_rad(image):

    # Select temperature band (scaled to K)
    temp = image.select('ST_B10')

    # Select atmospheric emissivity band
    emiss = image.select('atm_emiss')

    # Apply Stefan-Boltzmann equation (W/m²)
    dw_lw_rad = temp.pow(4).multiply(emiss).multiply(5.67E-8)

    # Add band to scene
    return image.addBands(dw_lw_rad.rename('dw_lw_rad'))


# %% UPWARD LONGWAVE RADIATION

def up_lw_rad(image):

    # Select temperature band (scaled to K)
    temp = image.select('ST_B10')

    # Select emissivity band
    emiss = image.select('emiss')

    # Apply Stefan-Boltzmann equation (W/m²)
    up_lw_rad = temp.pow(4).multiply(emiss).multiply(5.67E-8)

    # Add band to scene
    return image.addBands(up_lw_rad.rename('up_lw_rad'))


# %% NET LONGWAVE RADIATION

def net_lw_rad(image):

    # Upward and downward fluxes
    up = image.select('up_lw_rad')
    down = image.select('dw_lw_rad')

    # Emissivity
    emiss = image.select('emiss')

    # Net longwave radiation (W/m²)
    budget = emiss.multiply(down).subtract(up)

    return image.addBands(budget.rename('net_lw_rad'))


# %% ALL-WAVE NET RADIATION

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

    # Instantaneous all-wave net radiation (W/m²)
    Rn = dw_sw.subtract(up_sw).add(emiss.multiply(dw_lw)).subtract(up_lw)

    return image.addBands(Rn.rename('Rn'))


# %% SET SEASON

# Includes season in image metadata
def set_season(image):

    # Select month
    month = image.date().get('month')

    # Select day
    day = image.date().get('day')

    # Retrieve the season
    season = ee.List(
        [ee.Algorithms.If(month.eq(1), 'Summer'),
         ee.Algorithms.If(month.eq(2), 'Summer'),
         ee.Algorithms.If(month.eq(3).And(day.lt(21)), 'Summer', 'Fall'),
         ee.Algorithms.If(month.eq(4), 'Fall'),
         ee.Algorithms.If(month.eq(5), 'Fall'),
         ee.Algorithms.If(month.eq(6).And(day.lt(21)), 'Fall', 'Winter'),
         ee.Algorithms.If(month.eq(7), 'Winter'),
         ee.Algorithms.If(month.eq(8), 'Winter'),
         ee.Algorithms.If(month.eq(9).And(day.lt(23)), 'Winter', 'Spring'),
         ee.Algorithms.If(month.eq(10), 'Spring'),
         ee.Algorithms.If(month.eq(11), 'Spring'),
         ee.Algorithms.If(month.eq(12).And(day.lt(21)), 'Spring', 'Summer')]
        ).get(month.subtract(1))

    # Set to metadata
    return image.set({'SEASON':season})


# %% GENERATE METADATA DATAFRAME

def list_info_df(properties_list):

    # Variable initialization
    ids = []
    date = []
    time = []
    season = []
    cloudiness = []
    declination = []
    earth_sun_dist = []
    air_temp = []
    rel_hum = []

    # Information retrieval
    for j in range(0, len(properties_list)):

        ids.append(
            properties_list[j]['properties']['system:index'])

        date.append(
            properties_list[j]['properties']['system:time_start'])

        time.append(
            properties_list[j]['properties']['SCENE_CENTER_TIME'])

        season.append(
            properties_list[j]['properties']['SEASON'])

        cloudiness.append(
            properties_list[j]['properties']['CLOUDINESS'])

        declination.append(
            properties_list[j]['properties']['DECLINATION'])

        earth_sun_dist.append(
            properties_list[j]['properties']['EARTH_SUN_DISTANCE'])

        air_temp.append(
            properties_list[j]['properties']['AIR_TEMP'])

        rel_hum.append(
            properties_list[j]['properties']['REL_HUM'])

    # Dataframe's generation
    infos = pd.DataFrame({
        'id':ids,
        'date':date,
        'time':time,
        'season':season,
        'cloudiness':cloudiness,
        'declination':declination,
        'earth_sun_dist':earth_sun_dist,
        'air_temp':air_temp,
        'rel_hum':rel_hum})

    infos['date'] = pd.to_datetime(infos['date'], unit='ms')

    return infos


# %% TRANSFORM SHAPEFILES TO EE.GEOMETRIES

def shape_to_feature_coll(shapefile):

    '''
    Transforms a multipolygon shapefile (geopandas Dataframe) into
    an ee.FeatureCollection
    The shapefile must have a field called "classes"
    '''

    # Empty list to save the polygons
    samples_list = ee.List([])

    # Iterate over each polygon
    for pol in range(0, len(shapefile)):

        # Select the feature of interest
        polygon = (np.dstack(shapefile.geometry[pol].geoms[0]
                             .exterior.coords.xy).tolist())

        # Create a ee.Feature with the external coordinates
        geometry = ee.Feature(ee.Geometry.Polygon(polygon))

        geometry = geometry.set({'classes':shapefile.classes[pol]})

        # Append the geometry to the list
        samples_list = samples_list.add(geometry)

    return ee.FeatureCollection(samples_list)
