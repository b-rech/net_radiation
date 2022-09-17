# -*- coding: utf-8 -*-
"""
FEDERAL UNIVERSITY OF SANTA CATARINA
TECHNOLOGICAL CENTER
DEPT. OF SANITARY AND ENVIRONMENTAL ENGINEERING
LABORATORY OF MARINE HYDRAULICS

Created on Sat Sep  3 15:41:01 2022
Author: Bruno Rech (b.rech@outlook.com)
"""

# %% INITIALIZATION

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


# %% SHAPEFILE UPLOADING AND CONVERSION

# Upload contours of the basin
basin = gpd.read_file('vectors\\vector_layers.gpkg', layer='basin_area')

# Extract coordinates
basin_coord = np.dstack(basin.geometry[0].geoms[0].exterior.coords.xy).tolist()

# Create ee.Geometry with external coordinates
basin_geom = ee.Geometry.Polygon(basin_coord)

# Create area for clipping images
rect = ee.Geometry.Rectangle([-48.37428696492995, -27.68296346714984,
                            -48.56174151517356, -27.42983615496652])

# Delete not useful variables
del basin, basin_coord


# %% SELECTION OF LANDSAT DATA

# Filter Landsat 8 imagery:
    # by bounds;
    # by processing level (L2SP = optical and thermal bands);
    # by sensors quality (0=worst, 9=best);
# Reproject to WGS84
landsat8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(basin_geom)
            .filter(ee.Filter.contains('.geo', basin_geom))
            .filter(ee.Filter.eq('IMAGE_QUALITY_OLI', 9))
            .filter(ee.Filter.eq('IMAGE_QUALITY_TIRS', 9))
            .map(to_31982))


# %% CLOUD COVER ASSESSMENT

# Function to be mapped over the collection
def get_cloud_percent(image):

    """
    This function calculates the cloud cover percentage within the geometry and
    adds it to the metadata as a new attribute called CLOUDINESS.
    """

    # Select cloud band
    # It is assigned 1 to the selected pixels and 0 to the others
    # Bit 6: 1 to clear sky and 0 to cloud or dilated cloud
    cloud_band = (image.select(['QA_PIXEL']).bitwiseAnd(1<<6).eq(0))

    # Generate cloud proportion
    # Since the values are just 0 e 1, the mean is equal to the proportion
    # An ee.Dictionary is generated with a key renamed to "cloudiness"
    # The proportion is multiplied by 100 to get a percentual
    cloud_percent = cloud_band.reduceRegion(**{
        'reducer':ee.Reducer.mean(),
        'geometry':basin_geom,
        'scale':30}).rename(['QA_PIXEL'], ['CLOUDINESS'], True)\
        .map(lambda key, value : ee.Number(value).multiply(100))

    # Add information to image metadata
    return image.set(cloud_percent)


# Map function over the collection
dataset1 = landsat8.map(get_cloud_percent)

# Filter images (cloudiness limit of 10%)
dataset2 = (dataset1.filter(ee.Filter.lte('CLOUDINESS', 10))
            .sort('system:time_start'))


# %% ELEVATION DATA

# Digital elevation model from NASADEM (reprocessing of SRTM data)
# Clip to a smaller area
dem = ee.Image("NASA/NASADEM_HGT/001")

# Calculate slope. Units are degrees, range is [0, 90)
# Convert slope to radians
dem = dem.addBands(ee.Terrain.slope(dem).multiply(np.pi/180))

# Calculate aspect. Units are degrees where 0=N, 90=E, 180=S, 270=W
# Transform data to 0=S, -90=E, 90=W, -180=N by subtracting 180
# Convert to radians
dem = dem.addBands(ee.Terrain.aspect(dem).subtract(180).multiply(np.pi/180))


# %% CLIP, ADD DEM BAND, ADD COORDINATES


# Function to clip the scenes
def clip_area(image):

    return image.clip(rect)


# Function to add dem as a band to each image
def dem_bands(image):

    return image.addBands(dem, names=['elevation', 'slope', 'aspect'])

# Map functions
dataset3 = (dataset2
            .map(scale_L8)          # Scale the values
            .map(dem_bands)         # Add DEM, slope and aspect
            .map(pixels_coords)    # Add bands of lat and long coords
            .map(clip_area)         # Clip to the smaller area
            .map(to_31982))

# %% CALCULATE DECLINATION AND SOLAR TIME


# Calculate declination, B, E and day of year
dataset4 = dataset3.map(declination)

# Get centroid longitude for the area of interest (basin)
basin_long = basin_geom.centroid().coordinates().getNumber(0).abs()

# Function to calculate hour angle
def hour_angle(image):

    """
    This function calculates the hour angle of the scene.
    """

    # Retrieve parameter E (from minutes to seconds)
    E = image.getNumber('E').multiply(60)

    # Retrieve local time (seconds)
    local_time = image.date().getRelative('second', 'day')

    # Local time zone standard meridian (degrees)
    # UTC -3, 15 degrees each hour
    meridian = ee.Number(15*3)

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


# %% GENERATION OF THETA BANDS AND ALBEDO

# Calculate theta_hor and theta_rel
dataset6 = dataset5.map(theta_hor).map(theta_rel)

# Calculate albedo
dataset7 = dataset6.map(albedo)


# %%

# Retrieve metadata
info_list2 = dataset2.getInfo()['features']
info_df2 = list_info_df(info_list2)

# Temporal gaps
time_gaps = month_gaps(info_df2, 'date', '2013-03-01', '2022-08-31')


# %% PLOT OF TEMPORAL AVAILABILITY

# Extraction of years
info_df2['year'] = pd.DatetimeIndex(info_df2.date).year

# info_df3['year'] = pd.DatetimeIndex(info_df3.date).year

# Creation of the figure
plot, ax = plt.subplots(nrows=10, figsize=(10, 5), dpi=300)
sns.set_style('white')

# Iterate to each axis (year)
for year, axis in zip(
        range(info_df2.year.min(), info_df2.year.max() + 1),
        range(0, 10)):

    # Total available images (with good cloud cover)
    sns.scatterplot(x='date', y=0, data=info_df2[info_df2.year == year],
                    color='green', ax=ax[axis])

    # Selected images
    #sns.scatterplot(x='date', y=0, data=info_df3[info_df3.year == year],
               #     color='#006d2c', ax=ax[axis])

    # Styling
    ax[axis].set(xlim=(np.datetime64(f'{year}-01-01'),
                       np.datetime64(f'{year}-12-31')),
                 ylim=(-0.1, 0.1),
                 xticklabels=[], yticklabels=[],
                 ylabel = year)
    ax[axis].grid(visible=True, which='major', axis='both')

ax[9].set(xticklabels=range(1, 13), xlabel='Mês')








# %% IMAGES VISUALIZATION

image = dataset7.first()

theta_band = dataset6.first().select('theta_rel')

qa_pixel = (image.select('QA_PIXEL'))

clouds = qa_pixel.bitwiseAnd(1<<6).eq(0)
clouds = clouds.updateMask(clouds)

mapa = geemap.Map()
mapa.addLayer(image.select('elevation'), {'min':0, 'max':500})
mapa.addLayer(image.select('albedo'), {'color':'reds'})
mapa.addLayer(image, {'bands':['SR_B4', 'SR_B3', 'SR_B2'],'min':0, 'max':0.3})
mapa.addLayer(theta_band, {'min':-np.pi/2, 'max':np.pi/2})
mapa.addLayer(clouds, {'palette':'red'})
mapa.centerObject(basin_geom, 10)
mapa.addLayer(rect)
mapa.save('img2.html')
