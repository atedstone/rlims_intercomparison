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
import pandas as pd
import seaborn as sns

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

# %% trusted=true
cluster.close()

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
# ## Load MAR data

# %% trusted=true
runoff = marutils.open_dataset(os.path.join(pth_mar, '*.runoff.nc'), crs='epsg:3413', projection=None, base_proj4=None)

# %% tags=[] trusted=true
runoff

# %% [markdown] tags=[]
# <hr />
#
# ## Annual maximum runoff limit

# %% trusted=true
# Liege criteria = all cells where runoff exceeds 1mm day-1 on one or more days during a year.
# This means we need to exclude cells on days where they produce less than 1 mm day,
# before we resample to annual resolution.
ru_annual = runoff.RU.sel(SECTOR=1).where(runoff.RU.sel(SECTOR=1) > 1).resample(time='1AS').sum()

# %% trusted=true
ru_annual

# %% [markdown]
# #### DEBUG
#
# This is for checking that the extraction is working correctly.

# %% trusted=true
plt.figure()
aois.sel(aoi=156).plot()

# %% trusted=true
plt.figure()
dem.where(aois.sel(aoi=156)).plot(vmin=0, vmax=2500)

# %% trusted=true
plt.figure()
dem.where(aois.sel(aoi=156)).plot.hist(range=(0, 3000))

# %% trusted=true
plt.figure()
(ru_annual.where(aois.sel(aoi=156)).sum(dim=('x','y')) / 10 / 100 * (6000*6000) / 1e9).plot()

# %% trusted=true
fig = plt.figure()
#ru_annual.plot(col='time', col_wrap=5, cmap='Greys_r', vmin=0, vmax=100)
ru_annual.where(aois.sel(aoi=156)).plot(col='time', col_wrap=5, cmap='viridis', vmin=0, vmax=100)

# %% trusted=true
plt.figure()
dem.where(ru_annual > 1).where(aois.sel(aoi=156)).plot(col='time', col_wrap=5, cmap='viridis', vmin=0, vmax=2500)

# %% trusted=true
ru_gt1_count = ru_annual.where(ru_annual > 1).count(dim='time').where(mar_eg.MSK > 50)


# %% trusted=true
plt.figure()
ru_gt1_count.plot()

# %% trusted=true
ru_gt1_count.rio.to_raster('/flash/tedstona/ru_gt1_count.tif')

# %% [markdown]
# #### END DEBUG

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
# ## Annual maximum elevation of 'slush limit'
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
# ### 2. Deploy
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

# %% [markdown]
# ## Seasonal runoff limits extraction

# %% [markdown]
# ### Threshold analysis

# %% trusted=true
year = '2020'
store = {}
for t in [1, 5, 10, 20, 100]:
    rlim12 = dem \
        .where(aois.sel(aoi=156)) \
        .where(runoff.sel(time=year).RU.sel(SECTOR=1) >= 1) \
        .where(mar_eg.MSK > 50) \
        .where(runoff.sel(time=year).RU.sel(SECTOR=1).sum(dim='time') >= t) \
        .max(dim=('x','y'))
    store[t] = rlim12.squeeze().to_pandas()

# %% trusted=true
rlims = pd.DataFrame(store)

# %% trusted=true
with sns.color_palette('PuRd', 7):
    rlims[rlims > 0].plot(marker='.')


# %% trusted=true
rlims[rlims >0].to_csv('/flash/tedstona/fl156_%s_thresholds_mar.csv' %year)

# %% [markdown]
# ## Extract seasonal runoff limits for multiple flowlines at specific threshold

# %% trusted=true
# Load polygons list provided by Horst
fl = pd.read_excel('/flash/tedstona/_list_PolyIDs.xlsx')

# %% trusted=true
fl

# %% trusted=true
runoff

# %% trusted=true
store = {}
t_daily = 1
t_annual = 10

"""
We do this year-by-year:
    1. Corresponds with RACMO approach, which requires year-by-year to be computationally feasible (graphing doesn't work otherwise)
    2. The ffill() operation still misses the final year of data, because there isn't a time point for the following year.
"""
for year in range(2000, 2022):
    runoff_annual_mask = runoff.RU.sel(SECTOR=1).resample(time='1AS').sum() #.resample(time='1D', loffset='12H').ffill()
    m = runoff_annual_mask.sel(time=str(year)).squeeze()
    rm = m.expand_dims(dim={'time': pd.date_range('%s-01-01 12:00' %year, '%s-12-31 12:00' %year, freq='D')})
    runoff_prepared = runoff.RU.sel(SECTOR=1, time=str(year)).where(rm >= t_annual)
    for ix, aoi in fl.iterrows():
        print(aoi)
        rlim = dem \
            .where(aois.sel(aoi=aoi.PolyID)) \
            .where(runoff_prepared > t_daily) \
            .where(mar_eg.MSK > 50) \
            .max(dim=('x','y'))
        store[aoi.PolyID] = rlim.squeeze().to_pandas()
    rlims_here = pd.DataFrame(store)
    rlims_here[rlims_here > 0].to_csv('/flash/tedstona/flowlines_daily_rlims_MAR_%smmEvents_%smmAnnual_%s.csv' %(t_daily, t_annual, year))

# %% trusted=true
fl_rlims = pd.DataFrame(store)

# %% trusted=true
fl_rlims[fl_rlims > 0].plot()

# %% trusted=true
## Combine yearly files
s = []
#s.append(pd.read_csv('/flash/tedstona/flowlines_daily_rlims_MAR_1mmEvents_10mmAnnual.csv'))
for year in range(2000, 2022):
    p = '/flash/tedstona/flowlines_daily_rlims_MAR_1mmEvents_10mmAnnual_%s.csv' %year
    d = pd.read_csv(p)
    s.append(d)
all_mar = pd.concat(s, axis=0)

# %% trusted=true
all_mar

# %% trusted=true
all_mar.to_csv('/flash/tedstona/flowlines_daily_rlims_MAR_1mmEvents_10mmAnnual_2000_2021.csv')

# %% [markdown]
# ## (Defunct): Testing seasonal extraction with Flowline 156

# %% trusted=true
rlim156 = dem.where(aois.sel(aoi=156)).where(runoff.RU.sel(SECTOR=1) > t).where(mar_eg.MSK > 50).max(dim=('x','y')).to_pandas()

# %% trusted=true
rlim156

# %% trusted=true
rlim156.name = 'rlim'
rlim156_df = rlim156.to_frame()
rlim156_df.loc[:,'doy'] = rlim156_df.index.dayofyear
rlim156_df.loc[:,'year'] = rlim156_df.index.year
rlim156_piv = pd.pivot_table(rlim156_df, columns='year', index='doy', values='rlim')

# %% trusted=true
rlim156_piv

# %% trusted=true
rlim156_piv[rlim156_piv > 0].cummax().plot()

# %% trusted=true
m = rlim156_piv[rlim156_piv > 0].cummax().mean(axis=1).loc[150:250]
s = rlim156_piv[rlim156_piv > 0].cummax().std(axis=1).loc[150:250]

# %% trusted=true
sns.set_context('paper', rc={'font.family':'arial'})
plt.figure()
plt.fill_between(m.index, m-s, m+s, color='tab:blue', alpha=0.2, label='+/- 1 std 2000-2020')
m.loc[150:250].plot(linewidth=2, label='Mean 2000-2020')
rlim156_piv[rlim156_piv > 0].cummax().loc[150:250, 2012].plot(label='2012')
rlim156_piv[rlim156_piv > 0].cummax().loc[150:250, 2020].plot(label='2020')
plt.legend()
plt.ylabel('Runoff limit (m asl)')
plt.xlabel('Day of year')
plt.title('Flowline 156 | MAR | > 10 mm w.e. d')
plt.ylim(750, 2500)
plt.grid()
sns.despine()

# %% trusted=true
rlim156_piv[rlim156_piv > 0].cummax().to_csv('/flash/tedstona/fl156_2012_cummax_mar.csv')

# %% trusted=true
plt.figure()
rlim12.sel(SECTOR=1).plot(marker='o', linestyle='none')
plt.ylim(0, 2500)

# %% trusted=true
runoff.rio.clip(polys[polys['index'] == 156].geometry)

# %% trusted=true
runoff.drop_vars(['OUTLAY_bnds', 'TIME_bnds']).rio.clip(polys[polys['index'] == 156].geometry)

# %%
