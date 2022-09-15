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

# Uploading the contours of the basin
basin = gpd.read_file('vectors\\vector_layers.gpkg', layer='basin_area')

# Extraction of coordinates
basin_coord = np.dstack(basin.geometry[0].geoms[0].exterior.coords.xy).tolist()

# Creation of an ee.Geometry with the external coordinates
basin_geom = ee.Geometry.Polygon(basin_coord)

# Rectangle for clipping the images
rectangle = ee.Geometry.Polygon([[-48.56174151517356, -27.42983615496652],
                                 [-48.3482701411702, -27.441677519047225],
                                 [-48.37428696492995, -27.68296346714984],
                                 [-48.54973375043896, -27.67292048430085]])

# Deleting not useful variables
del basin, basin_coord


# %% SELECTION OF LANDSAT DATA

# Landsat 8 imagery filtering by bounds, sensors quality (0=worst, 9=best) and
# processing level (L2SP=reflectance and thermal bands are available)
# Reproject to WGS84
landsat8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(basin_geom)
            .filter(ee.Filter.contains('.geo', basin_geom))
            .filter(ee.Filter.eq('IMAGE_QUALITY_OLI', 9))
            .filter(ee.Filter.eq('IMAGE_QUALITY_TIRS', 9))
            .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'))
            .map(to_4326))


# %% CLOUD COVER ASSESSMENT

# Function to be mapped over the collection
def get_cloud_percent(image):

    """
    This function calculates the cloud cover percentage within the geometry and
    adds it to the metadata as a new attribute called CLOUDINESS
    """

    # Selection of cloud band
    # It is assigned 1 to the selected pixels and 0 to the others
    # Bit 6: 1 to clear sky and 0 to cloud or dilated cloud
    cloud_band = (image.select(['QA_PIXEL']).bitwiseAnd(1<<6).eq(0))

    # Generation of cloud proportion
    # Since the values are just 0 e 1, the mean is equal to the proportion
    # An ee.Dictionary is generated with a key renamed to "cloudiness"
    # The proportion is multiplied by 100 to get a percentual
    cloud_percent = cloud_band.reduceRegion(**{
        'reducer':ee.Reducer.mean(),
        'geometry':basin_geom,
        'scale':30}).rename(['QA_PIXEL'], ['CLOUDINESS'], True)\
        .map(lambda key, value : ee.Number(value).multiply(100))

    # Information added to image metadata
    return image.set(cloud_percent)


# Mapping of the function over the collection
dataset1 = landsat8.map(get_cloud_percent)

# Filtering of images (cloudiness limit of 5%)
dataset2 = (dataset1.filter(ee.Filter.lte('CLOUDINESS', 5))
            .sort('system:time_start'))

# Retrieval of metadata
info_list2 = dataset2.getInfo()['features']
info_df2 = list_info_df(info_list2)

# Temporal gaps
time_gaps = month_gaps(info_df2, 'date', '2013-03-01', '2022-08-31')


# %% SELECTION OF A SINGLE SCENE PER MONTH

# Extraction of year, month and day to separated cols
info_df2['year'] = pd.DatetimeIndex(info_df2.date).year
info_df2['month'] = pd.DatetimeIndex(info_df2.date).month
info_df2['day'] = pd.DatetimeIndex(info_df2.date).day

# Number of images to be analysed
n_imgs = len(info_df2)

# Counter
count = 2

# Selected_imgs (already includes the first one)
selected_imgs = [info_df2.id[1]]

# Year and month of the anterior image (unique)
year_ant = info_df2.year[1]
month_ant = info_df2.month[1]

# Repetition to each image (starting by the second one)
for y, m in zip(info_df2.year[2:n_imgs + 1],
                    info_df2.month[2:n_imgs + 1]):

    # Filtering of month and year
    available_imgs = info_df2[(info_df2.year == y) &
                                        (info_df2.month == m)
                                        ].replace()

    # Day of the anterior available image
    previous_month_day = info_df2[
        (info_df2.year == year_ant) &
        (info_df2.month == month_ant)].day.values[0]

    # Difference between the days (the smaller the difference, the closer to
    # a month or a month multiple in case of lacks)
    available_imgs['difference'] = abs(available_imgs.day - previous_month_day)

    # Selection of image with the smallest difference
    img = available_imgs[
        available_imgs.difference == available_imgs.difference.min()
        ].id.values[0]

    if img not in selected_imgs:
        selected_imgs.append(img)

    # Use of 'try' because in the last iteraction the value 'count + 1'
    # extrapolates the length of dataframe and generates an error
    try:

        # When the next record of the list is to a month different from the
        # current, the current values become the previous
        if info_df2.month[count + 1] != m:
            year_ant = y
            month_ant = m

    except:

        break

    # Count update
    count += 1


# Final selection of images based on the ids list generated
dataset3 = ee.ImageCollection(
    ['LANDSAT/LC08/C02/T1_L2/' + x for x in selected_imgs])

# Cloud cover
dataset3 = dataset3.map(get_cloud_percent)

# Metadata extraction
info_list3 = dataset3.getInfo()['features']

# Info dataframe
info_df3 = list_info_df(info_list3)

print(f'A set of {len(info_df3)} has been selected')


# %% PLOT OF TEMPORAL AVAILABILITY

# Extraction of years
info_df2['year'] = pd.DatetimeIndex(info_df2.date).year

info_df3['year'] = pd.DatetimeIndex(info_df3.date).year

# Creation of the figure
plot, ax = plt.subplots(nrows=10, figsize=(10, 5), dpi=300)
sns.set_style('white')

# Iterate to each axis (year)
for year, axis in zip(
        range(info_df2.year.min(), info_df2.year.max() + 1),
        range(0, 10)):

    # Total available images (with good cloud cover)
    sns.scatterplot(x='date', y=0, data=info_df2[info_df2.year == year],
                    color='#b2e2e2', ax=ax[axis])

    # Selected images
    sns.scatterplot(x='date', y=0, data=info_df3[info_df3.year == year],
                    color='#006d2c', ax=ax[axis])

    # Styling
    ax[axis].set(xlim=(np.datetime64(f'{year}-01-01'),
                       np.datetime64(f'{year}-12-31')),
                 ylim=(-0.1, 0.1),
                 xticklabels=[], yticklabels=[],
                 ylabel = year)
    ax[axis].grid(visible=True, which='major', axis='both')

ax[9].set(xticklabels=range(1, 13), xlabel='Mês')


# %% ELEVATION DATA

# Digital elevation model from NASADEM (reprocessing of SRTM data)
dem = ee.Image("NASA/NASADEM_HGT/001").clip(rectangle)

# Calculate slope. Units are degrees, range is [0, 90)
# Convert slope to radians
dem = dem.addBands(ee.Terrain.slope(dem).multiply(np.pi/180))

# Calculate aspect. Units are degrees where 0=N, 90=E, 180=S, 270=W
# Transform data to 0=S, -90=E, 90=W, -180=N by subtracting 180
# Convert to radians
dem = dem.addBands(ee.Terrain.aspect(dem).subtract(180).multiply(np.pi/180))


# %% CLIP AND ADD DEM BAND TO EACH IMAGE

# Function to clip the scenes
def clip_rect(image):

    return image.clip(rectangle)


# Function to add dem as a band to each image
def dem_bands(image):

    return image.addBands(dem, names=['elevation', 'slope', 'aspect'])


dataset4 = (dataset2
            .map(clip_rect)         # Clip to the rectangle
            .map(scale_L8)          # Scale the values
            .map(dem_bands)         # Add DEM, slope and aspect
            .map(pixels_coords))    # Add bands of lat and long coords



# %% IMAGES VISUALIZATION

image = (ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_220079_20130417')
         .select('SR_B.').multiply(0.0000275).add(-0.2))

qa_pixel = (ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_219079_20140208')
            .select('QA_PIXEL'))

clouds = qa_pixel.bitwiseAnd(1<<6).eq(0)
clouds = clouds.updateMask(clouds)

mapa = geemap.Map()
mapa.addLayer(image, {'bands':['SR_B4', 'SR_B3', 'SR_B2'],'min':0, 'max':0.3})
mapa.addLayer(clouds, {'palette':'red'})
mapa.centerObject(basin_geom, 10)
mapa.addLayer(rectangle)
mapa.save('img2.html')
