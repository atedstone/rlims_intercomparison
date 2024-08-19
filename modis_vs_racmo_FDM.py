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
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MODIS vs RACMO-offline FDM runoff limits

# %% trusted=true
import xarray as xr
import os
import numpy as np
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% [markdown]
# ## Dask cluster support

# %% trusted=true
from dask_jobqueue import SLURMCluster as MyCluster
from dask.distributed import Client
cluster = MyCluster()
cluster.scale(jobs=4)
client = Client(cluster)

# %% trusted=true
cluster.scale(jobs=8) #jobs=4

# %% trusted=true
client

# %% trusted=true
print(cluster.job_script())

# %% [markdown]
# ## Paths and data pre-requisites

# %% trusted=true
pth_racmo = '/flash/tedstona/machguth/FDM_Runoff_FGRN055_1957-2020_GrIS_GIC.nc'
pth_project = '/flash/tedstona/modis_vs_rcms'

# %% trusted=true
racmo_ds = xr.open_dataset(pth_racmo)
racmo_ds.rio.write_crs(racmo_ds.rotated_pole.proj4_params, inplace=True)

# %% [markdown]
# ## Project RACMO on to Polar Stereographic

# %% trusted=true
racmo_ds = racmo_ds.rio.reproject('epsg:3413')

# %% [markdown]
# ## Reproject DEM to match RACMO
#
# Requires opening 'dummy' file to check coordinates

# %% trusted=true
racmo_ds

# %% trusted=true
# Get bounding coordinates to create matching clipped DEM
res = racmo_ds.x[1] - racmo_ds.x[0]
half = res / 2
bounds = (
    (min(racmo_ds.x)-half).item(), 
    (min(racmo_ds.y)-half).item(), 
    (max(racmo_ds.x)+half).item(),
    (max(racmo_ds.y)+half).item()
)

# %% trusted=true
bounds

# %% trusted=true
'gdalwarp -tr {r} {r} -te {xmin} {ymin} {xmax} {ymax}'.format(r=int(res), xmin=int(bounds[0]), ymin=int(bounds[1]), xmax=int(bounds[2]), ymax=int(bounds[3]))

# %% trusted=true
# gdalwarp -tr 6000 6000 -te -639000.0 -3354927.734375 855000.0 -648927.734375 arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled.tif arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled_RACMO_FDM.tif

# %% [markdown]
# ## Load DEM

# %% trusted=true
dem = rioxarray.open_rasterio(os.path.join(pth_project, 'arcticdem_mosaic_500m_v30_greenland_icesheet_GeoidCorr_GapFilled_RACMO_FDM.tif')).squeeze()
# The coordinates are already 'the same' but not to sufficient precision...
dem['x'] = racmo_ds['x']
dem['y'] = racmo_ds['y']

# %% trusted=true
plt.figure()
dem.plot()

# %% trusted=true
dem.load()

# %% trusted=true
# Sanity check DEM orientation against the internal RACMO ice mask - 
# check the direction of y coordinates with respect to previous plot!
racmo_ds.lsm.plot()

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
aois_f = os.path.join(pth_project, 'racmo_fdm_polys.nc')

if not os.path.exists(aois_f):
    print('Rasterizing AOIs')
    
    polys = gpd.read_file(os.path.join(pth_project, 'Ys_polygons_v3.4b.shp'))
    polys.index = polys['index']
    polys.plot(column='index')
    
    store = []
    for ix, aoi in polys.iterrows():
        m = polygons_to_mask(racmo_ds, polygons=aoi)
        m['aoi'] = ix
        store.append(m)
    aois = xr.concat(store, dim='aoi')
    aois.to_netcdf(aois_f)

# %% trusted=true
aois = xr.open_dataarray(os.path.join(pth_project, aois_f))

# %% trusted=true
aois.sum(dim='aoi').plot()

# %% [markdown]
# <hr />

# %% [markdown]
# ## Annual maximum runoff limit

# %% trusted=true
THRESHOLD = 0

# %% [markdown]
# RACMO data is so large that we need to render down intermediate steps to disk then load them back in. Sequence:
#
# 1. Resample daily data to annual summed runoff. Save.
# 2. Broadcast DEMs over annual time axis, yielding DEMs with only the active runoff area retained. Save.
# 3. Apply AOIs over the annual DEMs and reduce with max().

# %% [markdown]
# ### Resample RACMO to annual resolution

# %% trusted=true
ru_annual_f = os.path.join(pth_project, '_tmp_RACMO_FDM_ru_annual.nc')

# %% trusted=true
racmo_ds.Runoff

# %% trusted=true
if not os.path.exists(ru_annual_f):
    print('Resampling RACMO to annual...')
    runoff = racmo_ds.Runoff.sel(time=slice('2000-01-01','2022-01-01'))

    t_chunks = tuple([len(pd.date_range('%s-01-01'%year, '%s-12-31'%year, freq='D')) for year in range(2000, 2021)])

    #runoff = runoff.chunk({'time':t_chunks})

    ru_annual = runoff.resample(time='1AS').sum()

    ru_annual.to_netcdf(ru_annual_f)
else:
    print('Loading annual data from disk.')


# %% trusted=true
ru_annual_nc = xr.open_dataarray(ru_annual_f, chunks={'time':21})

# %% [markdown] tags=[]
# #### DEBUG

# %% trusted=true
# %matplotlib widget
plt.figure()
ru_count = ru_annual_nc.where(ru_annual_nc > THRESHOLD).count(dim='time') 
ru_count.plot(vmin=0, vmax=20)

# %% trusted=true
ru_annual_nc.plot(col='time', col_wrap=5)

# %% trusted=true
ru_count.rio.to_raster('/flash/tedstona/racmo_FDM_ru_gt%s_count.tif' %THRESHOLD)

# %% [markdown]
# #### END DEBUG

# %% [markdown]
# ### Create runoff-masked DEMS
#
# This applies the threshold specified above!

# %% trusted=true
elev_masked_by_ru_f = os.path.join(pth_project, '_tmp_RACMO_FDM_ru_annual_masked_%smm.nc' %THRESHOLD)

if not os.path.exists(elev_masked_by_ru_f):
    elev_masked_by_ru = dem.where(ru_annual_nc > THRESHOLD)
    elev_masked_by_ru = elev_masked_by_ru.to_netcdf(elev_masked_by_ru_f)
    print('Created %s.' %elev_masked_by_ru_f)

# %% trusted=true
elev_masked_by_ru = xr.open_dataarray(elev_masked_by_ru_f, chunks={'time':21})

# %% trusted=true
elev_masked_by_ru

# %% [markdown]
# ### Reduce by AOIs

# %% trusted=true
aois

# %% trusted=true
# Prepare masked elevations and AOIs chunks
elev_masked_by_ru = elev_masked_by_ru.chunk({'time':1})
aois = aois.chunk({'aoi':100})

# %% trusted=true
elev_masked_by_ru

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
aoi_ru_f = os.path.join(pth_project, 'FDM_Runoff_FGRN055_1957-2020_GrIS_GIC-rlim-RUa%smm.nc' %THRESHOLD)
aoi_ru_save.to_netcdf(aoi_ru_f)

# %% [markdown]
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
racmo_rlim_reload = xr.open_dataset(os.path.join(pth_project, 'FDM_Runoff_FGRN055_1957-2020_GrIS_GIC-rlim-RUa%smm.nc' %THRESHOLD))

# %% trusted=true
racmo_rlim_reload.runoff_limit.mean(dim='time').plot(ax=ax)

# %% [markdown]
# ## Processing for comparisons with Zhang et al. (2023)
#
# From their text, "Daily RACMO2.3p2 simulations of surface water runoff are available from No¨el et al. (2019). RACMO2.3p2 exhibits a high spatial resolution of 1 km through statistical downscaling of its native 5.5 km resolution (No¨el et al., 2019). For the 2018 and 2019 mapping periods, we intersect the 1 km × 1 km RACMO grid cells with each basin and divide the sum of runoff by basin area to obtain mean runoff for each basin, following Li et al. (2022). Additionally, RACMO runoff values for each grid cell in the
# 2018 and 2019 mapping periods are temporally averaged to obtain spatial representations of mean daily runoff. The line where runoff values change from zero to positive values is used as the runoff elevation limit."
#
# The mapping periods are defined as follows:
# 2018: July 25-30, plus July 17-24,, July 31 to August 5
# 2019: July 29 to August 5, plus additional July 15-28, August 6-15, July 6, July 13, August 19
#
# The authors do not provide the code for their RACMO processing and the text lacks full clarity about whether the "mapping periods" are just the core date ranges, or whether they also include the other dates listed.

# %% scrolled=true tags=[] trusted=true
runoff = xr.open_mfdataset(os.path.join(pth_racmo, '*.nc'))
runoff.rio.write_crs('epsg:3413', inplace=True)
runoff = runoff.runoffcorr

# %% trusted=true
ru2018 = runoff.sel(time=slice('2018-07-25', '2018-07-30')).mean(dim='time') > 1
ru2019 = runoff.sel(time=slice('2019-07-29', '2019-08-05')).mean(dim='time') > 1

# %% trusted=true
fig, ax = plt.subplots()
ru2018.plot(ax=ax)
ru2019.plot(ax=ax, alpha=0.5)

# %% trusted=true
ru2018 = runoff.sel(time=slice('2018-06-01', '2018-09-30')).sum(dim='time') > 10
ru2019 = runoff.sel(time=slice('2019-06-01', '2019-09-30')).sum(dim='time') > 10
fig, ax = plt.subplots()
ru2018.plot(ax=ax)
ru2019.plot(ax=ax, alpha=0.5, cmap='rocket')

# %% [markdown]
# ## Seasonal runoff limits extraction

# %% [markdown]
# ### Load data

# %% trusted=true
polys = gpd.read_file(os.path.join(pth_project, 'Ys_polygons_v3.4b.shp'))
polys.index = polys['index']

# %% scrolled=true tags=[] trusted=true
runoff = xr.open_mfdataset(os.path.join(pth_racmo, 'runoff.20*.nc'))
runoff.rio.write_crs('epsg:3413', inplace=True)
runoff = runoff.sel(time=slice('2000-01-01','2022-01-01'))
t_chunks = tuple([len(pd.date_range('%s-01-01'%year, '%s-12-31'%year, freq='D')) for year in range(2000, 2022)])
#runoff = runoff.chunk({'time':t_chunks, 'y':2700, 'x':1496})
#runoff = runoff.rio.clip(polys[polys['index'] == 156].geometry)

#dem = dem.rio.clip(polys[polys['index'] == 156].geometry)

# %% trusted=true
aois.rio.write_crs('epsg:3413',inplace=True)

# %% [markdown]
# ### Threshold analysis

# %% trusted=true
store = {}
daily_t = 1
year = '2020'
for annual_t in [1, 5, 10, 20, 100]:
    rlim = dem \
        .where(aois.sel(aoi=156)) \
        .where(runoff.sel(time=year).runoffcorr > daily_t) \
        .where(runoff.sel(time=year).runoffcorr.sum(dim='time') > annual_t) \
        .max(dim=('x','y'))
    store[annual_t] = rlim.squeeze().to_pandas()

# %% trusted=true
rlims = pd.DataFrame(store)

# %% trusted=true
rlims

# %% trusted=true
# %matplotlib inline
import seaborn as sns
with sns.color_palette('PuRd', 6):
    rlims[rlims > 0].plot(marker='.')
    plt.ylim(1000, 1800)

# %% trusted=true
rlims[rlims >0].to_csv('/flash/tedstona/fl156_%s_thresholds_racmo.csv' %year)

# %% [markdown]
# ### Extract seasonal runoff limits for multiple flowlines at specific threshold

# %% trusted=true
fl = pd.read_excel('/flash/tedstona/_list_PolyIDs.xlsx')

# %% trusted=true
fl

# %% trusted=true
runoff

# %% trusted=true
t_daily = 1
t_annual = 10
for year in range(2000, 2022):
    print(year)
    store = {}
    rhere = runoff.sel(time=str(year))
    runoff_annual_mask = rhere.runoffcorr.resample(time='1AS').sum().resample(time='1D').ffill()
    rm = runoff_annual_mask.squeeze()
    rm = rm.expand_dims(dim={'time': pd.date_range('%s-01-01' %year, '%s-12-31' %year, freq='D')}) #.assign_coords({'time':runoff.sel(time='2012').time})
    runoff_prepared = rhere.runoffcorr.where(rm >= t_annual)
    for ix, aoi in fl.iterrows():
        print(ix)
        rlim = dem \
            .where(aois.sel(aoi=aoi.PolyID)) \
            .where(runoff_prepared > t_daily) \
            .max(dim=('x','y'))
        res = rlim.squeeze().to_pandas()
        store[aoi.PolyID] = res
    rlims_here = pd.DataFrame(store)
    rlims_here[rlims_here > 0].to_csv('/flash/tedstona/flowlines_daily_rlims_RACMO_%smmEvents_%smmAnnual_%s.csv' %(t_daily, t_annual, year))


# %% trusted=true
pd.DataFrame(store).plot()
plt.ylim(0, 2000)

# %% trusted=true
## Combine yearly files
s = []
for year in range(2000, 2022):
    p = '/flash/tedstona/flowlines_daily_rlims_RACMO_1mmEvents_10mmAnnual_%s.csv' %year
    d = pd.read_csv(p)
    s.append(d)
all_racmo = pd.concat(s, axis=0)

# %% trusted=true
all_racmo

# %% trusted=true
all_racmo.to_csv('/flash/tedstona/flowlines_daily_rlims_RACMO_1mmEvents_10mmAnnual_2000_2021.csv')

# %% [markdown]
# ### (Defunct): Pilot analysis specifically for FL156

# %% trusted=true
rlim156 = dem.where(aois.sel(aoi=156)).where(runoff.runoffcorr > 10).max(dim=('x','y')) #.to_pandas()

# %% trusted=true
rlim156.chunks

# %% trusted=true
rlim156pd = rlim156.to_pandas()

# %% trusted=true
rlim156pd.name = 'rlim'
rlim156_df = rlim156pd.to_frame()
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
