# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analysis of Landsat visible runoff limits vs MODIS slush limits

# %%
import geopandas as gpd
import pandas as pd
#For pandas < 3, https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pd.options.mode.copy_on_write = True
import matplotlib.pyplot as plt
import os
import geoutils as gu
import statsmodels.api as sm
import seaborn as sns
import numpy as np

from angles import *

# %% [markdown]
# ## Data paths

# %%
# Path to ArcticDEM with geoid correction, i.e. same source as Ys, needed to lookup equivalent heights for Landsat data
pth_landsat = '/Users/atedston/Dropbox/work/tmp_shares/visible_runoff_limits_submission2/rlim_annual_maxm/xytpd.csv'
pth_Ys = '/Users/atedston/Dropbox/work/papers/machguth_intercomparison/submission2/__slush-limit_output_table_OnlyValidEntries.xlsx'
pth_adem = '/scratch/arcticdem/arcticdem_mosaic_100m_v30_greenland_geoidCorr.tif'
pth_polys = '/Users/atedston/Dropbox/work/gis/Ys_polygons/Ys_polygons_v3.4b.shp'

pth_saveto = '/Users/atedston/Dropbox/work/papers/machguth_intercomparison'

# %% [markdown]
# ## Specify ranges of Ys polygons to consider

# %%
# Inclusive ranges
exclude = [(0, 114), (256, 278), (456, 568)]

# %% [markdown]
# ## Load Landsat data

# %%
# Load Landsat annual maximum runoff limits
xytpd = pd.read_csv(pth_landsat, index_col='index')
xytpd['date'] = pd.to_datetime(xytpd.date)

# Restrict to only Landsat retrievals temporally coincident with MODIS
xytpd = xytpd[xytpd.year >= 2000]

# Add geoferencing
xytpd = gpd.GeoDataFrame(xytpd, geometry=gpd.points_from_xy(xytpd.x, xytpd.y, crs=3413))

# Drop unnecessary columns
xytpd = xytpd.drop(labels=['Unnamed: 0', 'label', 'box_id', 'slice_id', 'loc_id'], axis='columns')

# Sanity check
xytpd.head()

# %% [markdown]
# ## Allocate a MODIS polyline to each Landsat retrieval and reduce

# %%
# Load MODIS polygons (these are already in EPSG3413)
ys_polys = gpd.read_file(pth_polys)
# read_file() cannot cope with setting index directly, so we do it now.
ys_polys.index = ys_polys['index']
ys_polys = ys_polys.drop(labels='index', axis='columns')

# %%
# Verify that indexes set correctly.
ys_polys.head()

# %%
# Apply Ys polygon exclusions
for excl in exclude:
    ys_polys = ys_polys[~ys_polys.index.isin(np.arange(excl[0], excl[1]+1))]

# %%
ys_polys

# %%
## OUTDATED/INCORRECT? - this doesn't work properly because the FlowPolys overlap in space.

# # Join
# xytpd_with_Ys = xytpd.sjoin(ys_polys, how='left', predicate='intersects')
# # Drop any rows which are not allocated to a Ys polygon
# xytpd_with_Ys = xytpd_with_Ys[xytpd_with_Ys.index_right.notna()]
# # Convert index to integer
# xytpd_with_Ys['index_right'] = xytpd_with_Ys['index_right'].astype('int')
# # Rename columns as needed
# xytpd_with_Ys = xytpd_with_Ys.rename(columns={
#     'index_right':'landsat_in_Ys_poly_id'
# })

# %%
# 'Correct' approach: iterate through all flowline polygons, grabbing the block of Landsat data which is within that polygon.
# This results in some duplication of Landsat data, which is addressed by data reduction later.

store = []
for ix, p in ys_polys.iterrows():
    data = xytpd[xytpd.within(p.geometry)]
    if len(data) == 0:
        continue
    data.loc[:,'PolyID'] = ix
    store.append(data)
xytpd_with_Ys = pd.concat(store)

# Drop the geometry column, otherwise the groupby command doesn't work.
xytpd_with_Ys = xytpd_with_Ys.drop(columns=['geometry'])

# For each unique PolyID-date combination, take the median of all Landsat data columns.
xytpd_with_Ys = xytpd_with_Ys.groupby(['PolyID','date']).median()

# %% [markdown]
# ## Join Landsat and MODIS datasets based on polygon IDs
#
# From HM email, 20.03.2025:
#
# `__slush-limit_output_table_OnlyValidEntries.xlsx` contains all dates and all runoff limit (slush limit SL) retrievals. 
#
# The coordinates of the point where flowline and runoff limit intersect are defined by SL_x_coord and SL_y_coord (EPSG:3413) 
#
# SL is the elevation of the runoff limit in m a.s.l. 
#
# The column that links to the flowline polygons is "ID".

# %%
Ys = pd.read_excel(pth_Ys)
Ys = Ys.filter(items=['date', 'year', 'ID', 'SL', 'SL_x_coord', 'SL_y_coord'], axis=1)
Ys = Ys.rename(columns={'ID':'PolyID'})

# Remove flowline polygons in areas which we excluded from Landsat analysis
# Approximate choices, based on manual visualisation of polygons in QGIS
Ys = Ys[(Ys.PolyID >= 114) & (Ys.PolyID < 450)]

# %%
merged = pd.merge(xytpd_with_Ys, Ys, left_on=['PolyID', 'date'], right_on=['PolyID', 'date'], how='inner', suffixes=['landsat','modis'])

# %% [markdown]
# ## Get ArcticDEM EGM2008 elevations for all Landsat points

# %%
adem = gu.Raster(pth_adem)
elevs = adem.value_at_coords(merged.x, merged.y)
merged['landsat_elev_adem_masl'] = elevs

# %% [markdown]
# ## Compute distances and angles between the points

# %%
## For the mean MODIS and Landsat points:
# Direction of MODIS detection with respect to Landsat
geometry_landsat = gpd.GeoSeries(gpd.points_from_xy(merged.x, merged.y, crs=3413), index=merged.index)
geometry_modis = gpd.GeoSeries(gpd.points_from_xy(merged.SL_x_coord, merged.SL_y_coord, crs=3413))
merged['angle_degrees'] = calc_angles(geometry_landsat, geometry_modis).astype(int)
# Distance between x,y points
merged['distance_metres'] = geometry_landsat.distance(geometry_modis).astype(int)

# %% [markdown]
# ## 'Complete' merged dataset

# %%
merged

# %% [markdown]
# ## Basic checks

# %%
# Histogram of elevation differences
plt.figure()
plt.hist((merged.SL - merged.landsat_elev_adem_masl), bins=np.arange(-1000,1000, 50))
plt.xlabel('Elevation difference [m] (+ve: Ys > RL)')
plt.ylabel('Freq.')
plt.savefig(os.path.join(pth_saveto, 'histogram_elev_difference.png'))

# %%
# Boxplot of distances between Landsat and MODIS points
plt.figure()
sns.boxplot(merged['distance_metres'] / 1000)
plt.ylabel('Distance Ys-RL [km EPSG:3413]')
plt.savefig(os.path.join(pth_saveto, 'boxplot_distances.png'))

# %%
# Elevation-based scatter plot
plt.figure()
ax = plt.subplot(111, aspect='equal')
plt.plot((0,2500), (0,2500), '--', color='grey', linewidth=0.5)

plt.plot(merged.landsat_elev_adem_masl, merged.SL, '.', alpha=0.1, color='tab:blue')
plt.xlim(0,2500)
plt.ylim(0, 2500)
plt.xlabel('$RL$ (Landsat, m a.s.l.)')
plt.ylabel('$Y_s$ (MODIS, m a.s.l.)')
plt.grid()
plt.savefig(os.path.join(pth_saveto, 'scatterplot_elevation.png'))
plt.savefig(os.path.join(pth_saveto, 'scatterplot_elevation.pdf'))

# %%
X = sm.add_constant(merged.landsat_elev_adem_masl)
y = merged.SL
m = sm.OLS(y, X).fit()
print(m.summary())

# %%
merged.to_csv(os.path.join(pth_saveto, 'modis_landsat_xytpd.csv'))

# %% [markdown]
# ## Debugging codes

# %%
# %matplotlib widget
plt.figure()
plt.plot(merged.x, merged.y, '.', color='tab:green', label='Landsat', alpha=0.2)
plt.plot(merged.SL_x_coord, merged.SL_y_coord, '.', color='tab:blue', label='MODIS', alpha=0.2)


# %%
def reduce_by_dist(x):
    min_dist = x.dist.min()
    return x[x.dist == min_dist]
merged_polybydist = merged.groupby(['date','PolyID']).apply(reduce_by_dist)

# %%
merged_polybydist

# %%
plt.figure()
plt.hist((merged_polybydist.SL - merged_polybydist.landsat_elev_adem), bins=np.arange(-1000,1000, 50))

# %%
