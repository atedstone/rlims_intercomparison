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
# # MODIS vs RACMO runoff limits

# %% trusted=true
import xarray as xr
import os
import numpy as np
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
import pandas as pd

# %% [markdown]
# ## Dask cluster support

# %% trusted=true
from dask_jobqueue import SLURMCluster as MyCluster
from dask.distributed import Client
cluster = MyCluster()
cluster.scale(jobs=4)
client = Client(cluster)

# %% trusted=true
cluster.scale(jobs=4) #jobs=4

# %% trusted=true
client

# %% trusted=true
print(cluster.job_script())

# %% [markdown]
# ## Paths and data pre-requisites

# %% trusted=true
pth_racmo = '/flash/tedstona/RACMO/1km/runoff'
pth_project = '/flash/tedstona/modis_vs_rcms'

# %% trusted=true
racmo_eg = xr.open_dataset(os.path.join(pth_racmo, 'runoff.2020_OND.BN_RACMO2.3p2_ERA5_3h_FGRN055.1km.DD.nc'))
racmo_eg.rio.write_crs('epsg:3413', inplace=True)

# %% [markdown]
# ## Reproject DEM to match RACMO
#
# Requires opening 'dummy' file to check coordinates

# %% trusted=true
racmo_eg

# %% trusted=true
# Get bounding coordinates to create matching clipped DEM
res = racmo_eg.x[1] - racmo_eg.x[0]
half = res / 2
bounds = (
    (min(racmo_eg.x)-half).item(), 
    (min(racmo_eg.y)-half).item(), 
    (max(racmo_eg.x)+half).item(),
    (max(racmo_eg.y)+half).item()
)

# %% trusted=true
bounds

# %% trusted=true
'gdalwarp -tr {r} {r} -te {xmin} {ymin} {xmax} {ymax}'.format(r=res, xmin=bounds[0], ymin=bounds[1], xmax=bounds[2], ymax=bounds[3])

# %% trusted=true
# gdalwarp -tr 6000 6000 -te -639000.0 -3354927.734375 855000.0 -648927.734375 arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled.tif arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled_MAR6km.tif

# %% [markdown]
# ## Load DEM

# %% trusted=true
dem = rioxarray.open_rasterio(os.path.join(pth_project, 'arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled_RACMO1km.tif')).squeeze()
# The coordinates are already 'the same' but not to sufficient precision...
dem['x'] = racmo_eg['x']
# Remember that we have to flip y :facepalm:
dem['y'] = racmo_eg['y'][::-1]

# %% trusted=true
plt.figure()
dem.plot()

# %% trusted=true
dem.load()

# %% [markdown]
# ## Create raster AOIs

# %% trusted=true
from rasterio import features
from collections.abc import Iterable   # import directly from collections for Python < 3.3

#if isinstance(the_element, Iterable):

def polygons_to_mask(
    ds, 
    polygon_file=None, 
    polygons=None,
    invert=False, 
    as_xr=True, 
    all_touched=False,
    value=1,
    to_bool=True
    ):
    """
    Originally from paper_rlim_detection_repo/load_env.py.
    
    From the input polygon(s), produces a single rasterised mask with dimensions of racmo_ds.
    """

    try:
        out_shape = (ds.dims['y'], ds.dims['x'])
    except TypeError:
        out_shape = ds.shape
    
    if polygon_file is not None:
        if not os.path.exists(polygon_file):
            raise IOError
        polygons = gpd.read_file(polygon_file)

    # Rasterize the box polygon
    if isinstance(polygons.geometry, Iterable):
        p = polygons.geometry
    else:
        p = [polygons.geometry]
        
    mask = features.rasterize(
        ((poly, value) for poly in p),
        out_shape=out_shape,
        transform=ds.rio.transform(),
        all_touched=all_touched
    )
    
    if to_bool:
        mask = mask.astype(bool)
        
    if invert:
        mask = ~mask

    if as_xr:
        mask = xr.DataArray(mask, dims=('y', 'x'), 
            coords={'y':ds.y, 'x':ds.x})
    
    return mask


# %% trusted=true
aois_f = os.path.join(pth_project, 'racmo_polys.nc')

if not os.path.exists(aois_f):
    print('Rasterizing AOIs')
    
    polys = gpd.read_file(os.path.join(pth_project, 'Ys_polygons_v3.4b.shp'))
    polys.index = polys['index']
    polys.plot(column='index')
    
    store = []
    for ix, aoi in polys.iterrows():
        m = polygons_to_mask(racmo_eg, polygons=aoi)
        m['aoi'] = ix
        store.append(m)
    aois = xr.concat(store, dim='aoi')
    aois.to_netcdf(aois_f)

# %% trusted=true
aois = xr.open_dataarray(os.path.join(pth_project, 'racmo_polys.nc'))

# %% trusted=true
aois.sum(dim='aoi').plot()

# %% [markdown]
# ## Prepare full RACMO time series

# %% trusted=true
THRESHOLD = 1

# %% [markdown]
# RACMO data is so large that we need to render down intermediate steps to disk then load them back in. Sequence:
#
# 1. Resample daily data to annual summed runoff. Save.
# 2. Broadcast DEMs over annual time axis, yielding DEMs with only the active runoff area retained. Save.
# 3. Apply AOIs over the annual DEMs and reduce with max().

# %% [markdown]
# ### Resample RACMO to annual resolution

# %% trusted=true
ru_annual_f = os.path.join(pth_project, '_tmp_RACMO_ru_annual.nc')

# %% trusted=true
if not os.path.exists(ru_annual_f):
    print('Resampling RACMO to annual...')
    runoff = xr.open_mfdataset(os.path.join(pth_racmo, '*.nc'))
    runoff.rio.write_crs('epsg:3413', inplace=True)

    runoff = runoff.sel(time=slice('2000-01-01','2022-01-01'))

    t_chunks = tuple([len(pd.date_range('%s-01-01'%year, '%s-12-31'%year, freq='D')) for year in range(2000, 2021)])

    runoff = runoff.chunk({'time':t_chunks, 'y':2700, 'x':1496})

    ru_annual = runoff.runoffcorr.resample(time='1AS').sum()

    ru_annual.to_netcdf()
else:
    print('Loading annual data from disk.')
    

# %% trusted=true
ru_annual_nc = xr.open_dataarray(ru_annual_f, chunks={'time':21})

# %% [markdown]
# ### Create runoff-masked DEMS
#
# This applies the threshold specified above!

# %% trusted=true
elev_masked_by_ru_f = os.path.join(pth_project, '_tmp_RACMO_ru_annual_masked_%smm.nc' %THRESHOLD)

if not os.path.exists(elev_masked_by_ru_f):
    elev_masked_by_ru = dem.where(ru_annual_nc > THRESHOLD)
    elev_masked_by_ru = elev_masked_by_ru.to_netcdf(elev_masked_by_ru_f)
    print('Created %s.' %elev_masked_by_ru_f)

# %% trusted=true
elev_masked_by_ru = xr.open_dataarray(elev_masked_by_ru_f, chunks={'time':21})

# %% [markdown]
# ### Reduce by AOIs

# %% trusted=true
# Prepare masked elevations and AOIs chunks
elev_masked_by_ru = elev_masked_by_ru.chunk({'time':1})
aois = aois.chunk({'aoi':100})

# %% trusted=true
aoi_ru = elev_masked_by_ru.where(aois).max(dim=('x','y'))
aoi_ru

# %% trusted=true
# Prepare data for export
aoi_ru.attrs['units'] = 'm'
aoi_ru.attrs['elevation_info'] = 'Metres above sea level from geoid-corrected ArcticDEM regridded to RCM horizontal resolution'
aoi_ru_save = xr.Dataset({'runoff_limit':aoi_ru})
aoi_ru_save

# %% trusted=true
aoi_ru_f = os.path.join(pth_project, 'RACMO2.3p2_ERA5_3h_FGRN055.1km-rlim-RUa%smm.nc' %THRESHOLD)
aoi_ru_save.to_netcdf(aoi_ru_f)

# %% [markdown]
# ---
#
# ### Export to CSV; Initial analysis

# %% trusted=true
racmo_rlim = xr.open_dataset(aoi_ru_f)

# %% trusted=true
plt.figure()
racmo_rlim.runoff_limit.T.plot(vmin=1000, vmax=2000)

# %% trusted=true
plt.figure()
(racmo_rlim.runoff_limit - racmo_rlim.runoff_limit.mean(dim='time')).T.plot(vmin=-200, vmax=200, cmap='RdBu_r')

# %% trusted=true
rlim_pd =  racmo_rlim['runoff_limit'].squeeze().to_dataframe().unstack()['runoff_limit']
rlim_pd.index = rlim_pd.index.year
rlim_pd.columns = rlim_pd.columns.astype(int)
rlim_pd

# %% trusted=true
aoi_ru_csv_f = aoi_ru_f[:-2] + 'csv'
rlim_pd.to_csv(aoi_ru_csv_f)

# %% trusted=true
# %matplotlib widget
plt.figure()
ax = plt.subplot(111)
racmo_rlim.runoff_limit.mean(dim='time').plot(ax=ax)

# %% trusted=true
racmo_rlim.close()

# %% trusted=true
racmo_rlim_10mm = xr.open_dataset(os.path.join(pth_project, 'RACMO2.3p2_ERA5_3h_FGRN055.1km-rlim-RUa10mm.nc'))

# %% trusted=true
racmo_rlim_10mm.runoff_limit.mean(dim='time').plot(ax=ax)

# %% trusted=true
racmo_rlim_1mm

# %%
