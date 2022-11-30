# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MODIS vs MAR runoff limits
#
# Data preparation: see `ncks_mar.sh`, which was run on climato.be.

# %% trusted=true
import xarray as xr
import os
import numpy as np
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt

# %% trusted=true
import marutils

# %% [markdown]
# ## Dask cluster support

# %% trusted=true
from dask_jobqueue import SLURMCluster as MyCluster
from dask.distributed import Client
cluster = MyCluster()
cluster.scale(jobs=4)
client = Client(cluster)

# %% trusted=true
cluster.scale(jobs=4)

# %% trusted=true
client

# %% [markdown]
# ## Paths and data pre-requisites

# %% trusted=true
pth_mar = '/flash/tedstona/MAR-v3.12.1'
pth_project = '/flash/tedstona/modis_vs_rcms'

# %% trusted=true
mar_eg = marutils.open_dataset(os.path.join(pth_mar, 'ICE.2020.01-12.h70.nc'), crs='epsg:3413')

# %% trusted=true
dem = rioxarray.open_rasterio(os.path.join(pth_project, 'arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled_MAR6km.tif')).squeeze()
# The coordinates are already 'the same' but not to sufficient precision...
dem['x'] = mar_eg['x']
# Remember that we have to flip y :facepalm:
dem['y'] = mar_eg['y'][::-1]

# %% trusted=true
plt.figure()
dem.plot()

# %% trusted=true
polys = gpd.read_file(os.path.join(pth_project, 'Ys_polygons_v3.4b.shp'))
polys.index = polys['index']

# %% trusted=true
polys.plot(column='index')

# %% [markdown]
# ## Reproject DEM to match MAR
#
# Requires opening 'dummy' file to check coordinates

# %% trusted=true
mar_eg

# %% trusted=true
# Get bounding coordinates to create matching clipped DEM
res = mar_eg.x[1] - mar_eg.x[0]
half = res / 2
bounds = (
    (min(mar_eg.x)-half).item(), 
    (min(mar_eg.y)-half).item(), 
    (max(mar_eg.x)+half).item(),
    (max(mar_eg.y)+half).item()
)

# %% trusted=true
bounds

# %% trusted=true
# gdalwarp -tr 6000 6000 -te -639000.0 -3354927.734375 855000.0 -648927.734375 arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled.tif arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled_MAR6km.tif

# %% [markdown]
# ## Create raster AOIs

# %% trusted=true
polys.head()

# %% trusted=true
from rasterio import features
from collections.abc import Iterable   # import directly from collections for Python < 3.3

#if isinstance(the_element, Iterable):

def polygons_to_mar_mask(
    mar_ds, 
    conform_with=None, 
    polygon_file=None, 
    polygons=None,
    invert=False, 
    as_xr=True, 
    all_touched=False
    ):
    """
    Originally from paper_rlim_detection_repo/load_env.py.
    
    From the input polygon(s), produces a single rasterised mask with dimensions of mar_ds.
    """

    try:
        out_shape = (mar_ds.dims['y'], mar_ds.dims['x'])
    except TypeError:
        out_shape = mar_ds.shape
    
    if polygon_file is not None:
        if not os.path.exists(polygon_file):
            raise IOError
        polygons = gpd.read_file(polygon_file)
    
    # Adjust the geometry to include the no-detection areas
    if conform_with is not None:
        polygons = polygons.append(conform_with)
        polygons['FID'] = 0
        polygons['geometry'] = polys.dissolve(by='FID')
        
    # Following the dissolve, can now convert to MAR projection
    #import shapely
    #if isinstance(polygons, shapely.geometry.polygon.Polygon):
        
    #polygons = polygons.to_crs(mar_ds.rio.crs)
    #polygons['geometry'] = polygons.buffer(1000)
    #poly = polys.iloc[0]

    # Rasterize the box polygon
    if isinstance(polygons.geometry, Iterable):
        p = polygons.geometry
    else:
        p = [polygons.geometry]
        
    mask = features.rasterize(
        ((poly, 1) for poly in p),
        out_shape=out_shape,
        transform=mar_ds.rio.transform(),
        all_touched=all_touched
    )
    
    mask = mask.astype(bool)
    if invert:
        mask = ~mask

    if as_xr:
        mask = xr.DataArray(mask, dims=('y', 'x'), 
            coords={'y':mar_ds.y, 'x':mar_ds.x})
    
    return mask


# %% trusted=true
store = []
for ix, aoi in polys.iterrows():
    m = polygons_to_mar_mask(mar_eg, polygons=aoi)
    m['aoi'] = ix
    store.append(m)
    

# %% trusted=true
aois = xr.concat(store, dim='aoi')

# %% trusted=true
aois

# %% trusted=true
# %matplotlib widget
aois.sum(dim='aoi').plot()

# %% [markdown]
# ## Prepare full MAR time series

# %% trusted=true
runoff = marutils.open_dataset(os.path.join(pth_mar, '*.runoff.nc'), crs='epsg:3413', projection=None, base_proj4=None)

# %% trusted=true
runoff

# %% trusted=true
# Liege criteria = all cells where runoff exceeds 1mm day-1 on one or more days during a year.
# This means we need to exclude cells on days where they produce less than 1 mm day,
# before we resample to annual resolution.
ru_annual = runoff.RU.sel(SECTOR=1).where(runoff.RU.sel(SECTOR=1) > 1).resample(time='1AS').sum()

# %% trusted=true
ru_annual

# %% trusted=true
# Single AOI test case
aoi1 = dem.where(ru_annual > 1).where(aois.sel(aoi=10))
plt.figure()
aoi1.plot(col='time', col_wrap=5, vmin=1000, vmax=2500)


# %% trusted=true
# Map the operation across all AOIs

def _reduce_runoff(aoi, ru, el):
    return el.where(ru > 1).where(aoi).max(dim=('x','y'))

def reduce_runoff_annual(aoi):
    return _reduce_runoff(aoi, ru_annual, dem)

#sel(aoi=slice(1,5))
aoi_ru = aois.map_blocks(reduce_runoff_annual)
aoi_ru

# %% trusted=true
# Prepare data for export
aoi_ru.attrs['units'] = 'm'
aoi_ru.attrs['elevation_info'] = 'Metres above sea level from geoid-corrected ArcticDEM regridded to 6 km horizontal resolution'
aoi_ru_save = xr.Dataset({'runoff_limit':aoi_ru})
aoi_ru_save

# %% trusted=true
aoi_ru_save.to_netcdf(os.path.join(pth_project, 'MAR-v3.12.1-rlim-RUa1mm.nc'))

# %% [markdown]
# ---
# Reload the data from disk

# %% trusted=true
mar_rlim = xr.open_dataset(os.path.join(pth_project, 'MAR-v3.12.1-rlim-RUa1mm.nc'))

# %% trusted=true
mar_rlim

# %% trusted=true
plt.figure()
mar_rlim.runoff_limit.T.plot(vmin=1000, vmax=2000)

# %% trusted=true
plt.figure()
(mar_rlim.runoff_limit - mar_rlim.runoff_limit.mean(dim='time')).T.plot(vmin=-200, vmax=200, cmap='RdBu_r')

# %% trusted=true
rlim_pd =  mar_rlim['runoff_limit'].squeeze().to_dataframe().unstack()['runoff_limit']
rlim_pd.index = rlim_pd.index.year
rlim_pd.columns = rlim_pd.columns.astype(int)
rlim_pd

# %% trusted=true
rlim_pd.to_csv(os.path.join(pth_project, 'MAR-v3.12.1-rlim-RUa1mm.csv'))

# %% trusted=true
mar_rlim.close()

# %% [markdown]
# ## Slush area maximum elevation
#
# Using equations from Clerx et al. (2022).
#
# Because this approach for detecting surface runoff doesn't yield a useful result (see below), there's no analysis on AOIs here - just a 'quick' bulk look at the whole dataset.

# %% [markdown]
# ### 1. Test retrieval approach

# %% trusted=true
RO1_1m = runoff.RO1.sel(time=slice('2012-07-01', '2012-09-01')).sel(OUTLAY=slice(0, 0.1)).mean(dim='OUTLAY')

# %% trusted=true
WA1_1m = runoff.WA1.sel(time=slice('2012-07-01', '2012-09-01')).sel(OUTLAY=slice(0, 0.1)).mean(dim='OUTLAY')

# %% trusted=true
por = 1. - (RO1_1m / 920.)

# %% trusted=true
plt.figure(),por.plot()

# %% trusted=true
# %matplotlib widget
plt.figure(), (WA1_1m / por).max(dim='time').plot(vmin=0, vmax=1)

# %% [markdown]
# ## 2. Deploy
#
# Get uppermost 0.2 m of snow surface. Calculate daily water saturation. Take the maximum annual value.

# %% trusted=true
WA1_upper = runoff.WA1.sel(OUTLAY=slice(0, 0.2)).mean(dim='OUTLAY')
RO1_upper = runoff.RO1.sel(OUTLAY=slice(0, 0.2)).mean(dim='OUTLAY')
# Reduce to slush apperance based on maximum water saturation each year
por = (1. - (RO1_upper / 920.))
S_w = (WA1_upper / por).resample(time='1AS').max(dim='time')

# %% trusted=true
S_w_save = xr.Dataset({'water_saturation':S_w})
S_w_save.water_saturation.attrs['units'] = 'unitless'
S_w_save.to_netcdf(os.path.join(pth_project, 'MAR-v3.12.1-rlim-slush.nc'))

# %% trusted=true
S_w = xr.open_dataset(os.path.join(pth_project, 'MAR-v3.12.1-rlim-slush.nc'))

# %% trusted=true
plt.figure(), S_w.water_saturation.plot(col='time', col_wrap=5, vmin=0, vmax=0.5)

# %% trusted=true
# We can use this code to verify that no pixels meet the saturated criterion.
RO1_upper_max = RO1_upper.resample(time='1AS').max()
(S_w.water_saturation.where(S_w.water_saturation > 0.8)).where(mar_eg.MSK > 90).where(RO1_upper_max < 920).count(dim=('x', 'y')).compute()

# %% trusted=true
cluster.close()

# %% trusted=true
