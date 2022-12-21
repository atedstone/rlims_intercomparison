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
# <hr />

# %% [markdown]
# ## Annual maximum runoff limit

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

# %% [markdown] tags=[]
# #### DEBUG

# %% trusted=true
# %matplotlib widget
plt.figure()
ru_count = ru_annual_nc.where(ru_annual_nc > 1).count(dim='time') #.plot(vmin=0, vmax=20)

# %% trusted=true
ru_count.rio.to_raster('/flash/tedstona/racmo_ru_gt1_count.tif')

# %% [markdown]
# #### END DEBUG

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

# %%
