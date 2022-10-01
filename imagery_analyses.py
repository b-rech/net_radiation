# -*- coding: utf-8 -*-
"""
FEDERAL UNIVERSITY OF SANTA CATARINA
TECHNOLOGICAL CENTER
DEPT. OF SANITARY AND ENVIRONMENTAL ENGINEERING
LABORATORY OF MARINE HYDRAULICS

Created on Sat Sep  3 15:41:01 2022
Author: Bruno Rech (b.rech@outlook.com)

SCRIPT 2/3 - IMAGERY ANALYSES
"""

# %% 1 DATA PREPARATION #######################################################

# This part of the code prepares the data for the subsequent calculations.

# -----------------------------------------------------------------------------
# 1.1 LIBRARIES AND SCRIPTS

# Required libraries
import ee
import numpy as np
import pandas as pd
import geopandas as gpd
import geemap
from matplotlib import pyplot as plt
import seaborn as sns

# Required scripts
from user_functions import *

# GEE authentication and initialization
#ee.Authenticate()
ee.Initialize()

# -----------------------------------------------------------------------------
# 1.2 VECTOR LAYERS UPLOAD

# Upload contours of the basin and lagoon
basin = gpd.read_file('vectors\\vector_layers.gpkg', layer='basin_area')
lagoon = gpd.read_file('vectors\\vector_layers.gpkg', layer='lagoon')

# Extract coordinates
basin_coord = np.dstack(basin.geometry[0].geoms[0]
                        .exterior.coords.xy).tolist()

lagoon_coord = np.dstack(lagoon.geometry[0].geoms[0]
                         .exterior.coords.xy).tolist()

# Create ee.Geometry with external coordinates
basin_geom = ee.Geometry.Polygon(basin_coord)
lagoon_geom = ee.Geometry.Polygon(lagoon_coord)

# Create rectangle for clipping images
rect = ee.Geometry.Rectangle([-48.37428696492995, -27.68296346714984,
                            -48.56174151517356, -27.42983615496652])

# -----------------------------------------------------------------------------
# 1.3 METEOROLOGICAL DATA UPLOAD

# Upload meteorological data from weather station
met_data = pd.read_csv('data_station_inmet.csv', sep=';', decimal=',',
                       skiprows=10)

# Select and rename attributes
met_data = (met_data.iloc[:, [0, 1, 3, 7, 9, 18]]
            .dropna().reset_index(drop=True))
met_data.columns = ['date', 'hour', 'p_atm', 'rad', 'air_temp', 'rel_hum']

# Convert data
met_data['hour'] = met_data.loc[:, 'hour']/100
met_data['rel_hum'] = met_data.loc[:, 'rel_hum']/100

# Select dates between 11h and 15h only (L8 passes usually around 13h)
met_data = met_data[[11 <= h <= 15 for h in met_data.hour]]

# Atmospheric pressure from hPa to kPa
met_data['p_atm'] = met_data.loc[:, 'p_atm']/10

# Units: atmospheric pressure in kPa, global radiation in kJ/m²,
# air temperature (dry-bulb) in °C.

# -----------------------------------------------------------------------------
# 1.4 SELECTION OF LANDSAT DATA

# Filter Landsat 8 imagery:
    # by bounds;
    # by processing level (L2SP = both optical and thermal bands);
    # by sensors quality (0=worst, 9=best);
landsat8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(basin_geom)
            .filter(ee.Filter.contains('.geo', basin_geom))
            .filter(ee.Filter.eq('IMAGE_QUALITY_OLI', 9))
            .filter(ee.Filter.eq('IMAGE_QUALITY_TIRS', 9)))

# -----------------------------------------------------------------------------
# 1.5 CLOUD COVER ASSESSMENT


# Function to retrieve cloud cover over the area of interest
def get_cloud_percent(image):

    """
    This function calculates the cloud cover proportion within the geometry and
    adds it to the metadata as a new attribute called CLOUDINESS.
    """

    # Select cloud band
    # It is assigned 1 to the selected pixels and 0 to the others
    # Bit 6: 1 to clear sky and 0 to cloud or dilated cloud
    cloud_band = (image.select(['QA_PIXEL']).bitwiseAnd(1<<6).eq(0))

    # Generate cloud proportion
    # Since the values are just 0 e 1, the mean is equal to the proportion
    # An ee.Dictionary is generated with the key renamed to "cloudiness"
    cloud_percent = cloud_band.reduceRegion(**{
        'reducer':ee.Reducer.mean(),
        'geometry':basin_geom,
        'scale':30}).rename(['QA_PIXEL'], ['CLOUDINESS'], True)

    # Add information to image metadata
    return image.set(cloud_percent)


# Map function over the collection
dataset1 = landsat8.map(get_cloud_percent)

# Filter images (cloudiness limit of 5%)
# and apply cloud mask (from user_functions)
dataset2 = (dataset1.filter(ee.Filter.lte('CLOUDINESS', 0.05))
            .map(cloud_mask).sort('system:time_start'))

# -----------------------------------------------------------------------------
# 1.6 ELEVATION DATA

# Digital elevation model from NASADEM
dem = ee.Image("NASA/NASADEM_HGT/001")

# Calculate slope. Units are degrees, range is [0, 90)
# Convert slope to radians
dem = dem.addBands(ee.Terrain.slope(dem).multiply(np.pi/180))

# Calculate aspect. Units are degrees where 0=N, 90=E, 180=S, 270=W
# Transform data to 0=S, -90=E, 90=W, -180=N by subtracting 180
# Convert to radians
dem = dem.addBands(ee.Terrain.aspect(dem).subtract(180).multiply(np.pi/180))

# -----------------------------------------------------------------------------
# 1.7 FINAL SCENE ADJUSTMENTS

# Map functions:
# Scale the bands (from user_functions)
# Add DEM, slope and aspect
# Add bands of lat and long coords (from user_functions)
# Clip to the rectangle
# Reproject to SIRGAS 2000 UTM zone 22S
dataset3 = (dataset2
            .map(scale_L8)
            .map(lambda img : img.addBands(
                dem, names=['elevation', 'slope', 'aspect']))
            .map(pixels_coords)
            .map(lambda img : img.clip(rect))
            .map(lambda img : img.reproject(crs='EPSG:31982', scale=30)))

# %% 2 SHORTWAVE RADIATION ####################################################

# This part of the code provides the calculation of shortwave radiation.

# -----------------------------------------------------------------------------
# 2.1 RETRIEVAL OF ANGULAR PARAMETERS

# Calculate declination, B, E and day of year (from user_functions)
dataset4 = dataset3.map(declination)

# Get centroid longitude (degrees absolute) for the area of interest (basin)
basin_long = basin_geom.centroid().coordinates().getNumber(0).abs()


# Function to calculate hour angle (assumed constant over each scene)
def hour_angle(image):

    """
    This function calculates the hour angle of the scene.
    """

    # Retrieve parameter E (convert to seconds)
    E = image.getNumber('E').multiply(60)

    # Retrieve local time (seconds)
    local_time = image.date().getRelative('second', 'day')

    # Local time zone standard meridian (degrees)
    # UTC -3, 15 degrees each hour
    meridian = ee.Number(3*15)

    # Calculate solar time (seconds)
    solar_time = local_time.add(
        ee.Number(4*60).multiply(meridian.subtract(basin_long))).add(E)

    # Calculate hour angle (radians)
    hour_angle = solar_time.subtract(60*60*12).divide(60*60).multiply(np.pi/12)

    # Calculate solar time in hours, minutes and seconds
    h = solar_time.divide(3600).floor().int()
    m = solar_time.mod(3600).divide(60).floor().int()
    s = solar_time.mod(3600).mod(60).floor().int()

    solar_hms = ee.Date(
        ee.String('0T').cat(ee.List([h, m, s]).join(':'))).format('HH:mm:ss')

    # Set values to metadata
    return image.set({'SOLAR_TIME':solar_hms, 'HOUR_ANGLE':hour_angle})


# Map function over collection
dataset5 = dataset4.map(hour_angle)

# Calculate solar zenith over a horizontal surface (cos_theta_hor) and solar
# incidence angle (cos_theta_rel) cosines (from user_functions)
dataset6 = dataset5.map(cos_theta_hor).map(cos_theta_rel)

# -----------------------------------------------------------------------------
# 2.2 ATMOSPHERIC TRANSMISSIVITY

# Calculate atmospheric pressure from elevation (from user_functions)
dataset7 = dataset6.map(atm_pressure)

# Calculate vapor pressure from meteorological data (weather station):

# Generate yyyy-mm-dd-Hhh IDs for information matching
met_data['ids'] = met_data.date + '-H' + met_data.hour.astype(int).astype(str)

# Retrieve saturation vapor pressure (kPa)
met_data['sat_vp'] = (0.6112*np.exp(
    17.62*met_data.air_temp/(243.12+met_data.air_temp)))

# Create ee.Dictionary with date/hour ids and associated vapor pressure
vp_values = ee.Dictionary.fromLists(keys=met_data.ids.values.tolist(),
                                    values=met_data.sat_vp.values.tolist())

# Create ee.Dictionary with date/hour ids and associated relative humidity
hum_values = ee.Dictionary.fromLists(keys=met_data.ids.values.tolist(),
                                    values=met_data.rel_hum.values.tolist())


# Function to assign saturation vapor pressure and relative humidity
# values to each image
def vp_hum(image):

    # Image's hour of acquisition
    hour = (image.date().get('hour')
            .add(image.date().get('minute').divide(60))
            .round().format('%.0f'))

    # ID in the format yyyy-mm-dd-Hhh
    date_hour = ee.String(
        image.get('DATE_ACQUIRED')).cat(ee.String('-H').cat(hour))

    return image.set({'SAT_VP':vp_values.getNumber(date_hour),
                      'REL_HUM':hum_values.getNumber(date_hour)})


# Map functions over the collection to retrieve:
# Saturation vapor pressure and relative humidity
# Precipitable water (from user_functions)
# Atmospheric transmissivity (from user_functions)
dataset8 = dataset7.map(vp_hum).map(prec_water).map(atm_trans)

# -----------------------------------------------------------------------------
# 2.2 DOWNWARD SHORTWAVE RADIATION

# Retrieve downward shortwave fluxes (from user_functions)
dataset9 = dataset8.map(dw_sw_rad)

# -----------------------------------------------------------------------------
# 2.3 UPWARD SHORTWAVE RADIATION

# Calculate albedo (from user_functions)
dataset10 = dataset9.map(get_albedo)

# Retrieve upward shortwave radiation (from user_functions)
dataset11 = dataset10.map(up_sw_rad)

# -----------------------------------------------------------------------------
# 2.4 SHORTWAVE RADIATION BUDGET

# Calculate shortwave radiation budget
dataset12 = dataset11.map(net_sw_rad)

# Visualization
mean_sw_rn = dataset12.select('net_sw_rad').mean().clip(basin_geom)

vis_params = {'min':350, 'max':900,
              'palette':['#1a9850', '#91cf60', '#d9ef8b',
                         '#fee08b', '#fc8d59', '#d73027']}

mean_sw_rn_map = geemap.Map()
mean_sw_rn_map.addLayer(mean_sw_rn, vis_params)
mean_sw_rn_map.centerObject(mean_sw_rn, 12)
mean_sw_rn_map.save('mean_sw_rn_map.html')

# %% PART 3: LONGWAVE RADIATION ###############################################

# -----------------------------------------------------------------------------
# 3.1 DOWNWARD LONGWAVE RADIATION

# Calculate atmospheric emissivity (from user_functions)
dataset13 = dataset12.map(atm_emiss)

# Retrieve downward longwave radiation (from user_functions)
dataset14 = dataset13.map(dw_lw_rad)

# -----------------------------------------------------------------------------
# 3.2 UPWARD LONGWAVE RADIATION

# Calculate SAVI, LAI and emissivity
def savi_lai_emiss(image):

    # Water mask (using NDWI to identify water surfaces)
    water_mask = (dataset14.mean()
                  .normalizedDifference(['SR_B3', 'SR_B5']).lt(0))

    # Select required bands
    # Remove lagoon pixels
    red = image.select('SR_B4').updateMask(water_mask)
    nir = image.select('SR_B5').updateMask(water_mask)

    # Set the value for L
    L = 0.5

    # Calculate SAVI
    savi = (nir.subtract(red).divide(nir.add(red).add(L)).multiply(1 + L)
            .rename('savi'))

    # Calculate LAI
    raw_lai = savi.multiply(-1).add(0.69).divide(0.59).log().divide(-0.91)

    # LAI <= 3 mask
    lai_lte3 = raw_lai.lte(3)

    # Apply mask to keep the pixels <= 3 and attribute 3 to masked pixels
    # Due to unmask, all masked pixels are also replaced
    # Re-apply cloud and water mask
    lai = (raw_lai.updateMask(lai_lte3).unmask(3).rename('lai')
           .updateMask(image.select('QA_PIXEL').bitwiseAnd(1<<6))
           .updateMask(water_mask))

    # Calculate emissivity
    emiss_raw = lai.multiply(0.01).add(0.95)

    # Attribute emissivity = 0.985 to water pixels (Tasumi, 2003)
    emiss = (emiss_raw.unmask(ee.Image(0.985).updateMask(water_mask.eq(0)))
             .rename('emiss'))

    # Add bands to the image
    return image.addBands(ee.Image([savi, lai, emiss]))


# Map the function over the collection
dataset15 = dataset14.map(savi_lai_emiss)

# Calculate Upward Longwave Radiation (from user_functions)
dataset16 = dataset15.map(up_lw_rad)

# -----------------------------------------------------------------------------
# 3.3 LONGWAVE RADIATION BUDGET

# Calculate longwave radiation budget (from user_functions)
dataset17 = dataset16.map(net_lw_rad)

# Visualization
mean_lw_rn = dataset17.select('net_lw_rad').mean().clip(basin_geom)

vis_params = {'min':-100, 'max':-78,
              'palette':['#1a9850', '#91cf60', '#d9ef8b',
                         '#fee08b', '#fc8d59', '#d73027']}

mean_lw_rn_map = geemap.Map()
mean_lw_rn_map.addLayer(mean_lw_rn, vis_params)
mean_lw_rn_map.centerObject(mean_lw_rn, 12)
mean_lw_rn_map.save('mean_lw_rn_map.html')

# %% PART 4: ALL-WAVE NET RADIATION ###########################################

dataset18 = dataset17.map(all_wave_rn)

# Visualization
mean_rn = dataset18.select('Rn').mean().clip(basin_geom)

vis_params = {'min':300, 'max':700,
              'palette':['#1a9850', '#91cf60', '#d9ef8b',
                         '#fee08b', '#fc8d59', '#d73027']}

mean_rn_map = geemap.Map()
mean_rn_map.addLayer(mean_rn, vis_params)
mean_rn_map.centerObject(mean_rn, 12)
mean_rn_map.save('mean_rn_map.html')

# %% PLOT OF TEMPORAL AVAILABILITY

# Retrieve collection metadata
info_list = dataset7.getInfo()['features']
info_df = list_info_df(info_list)

# Extraction of years
info_df['year'] = pd.DatetimeIndex(info_df.date).year

# Create figure
plot, ax = plt.subplots(nrows=10, figsize=(10, 5), dpi=300)
sns.set_style('white')

# Iterate on each axis (year)
for year, axis in zip(
        range(info_df.year.min(), info_df.year.max() + 1),
        range(0, 10)):

    # Total available images (with good cloud cover)
    sns.scatterplot(x='date', y=0, data=info_df[info_df.year == year],
                    color='green', ax=ax[axis])

    # Styling
    ax[axis].set(xlim=(np.datetime64(f'{year}-01-01'),
                       np.datetime64(f'{year}-12-31')),
                 ylim=(-0.1, 0.1),
                 xticklabels=[], yticklabels=[],
                 ylabel = year)
    ax[axis].grid(visible=True, which='major', axis='both')

ax[9].set(xticklabels=range(1, 13), xlabel='Mês')


# %% IMAGES VISUALIZATION

mean_rn = dataset16.select('ST_B10').mean().clip(basin_geom)

vis_params = {'min':283, 'max':303,
              'palette':['#3288bd', '#99d594', '#e6f598',
                         '#fee08b', '#fc8d59', '#d53e4f']}

mapa = geemap.Map()
mapa.addLayer(mean_rn, vis_params)
mapa.centerObject(basin_geom, 12)
mapa.save('Rn.html')
