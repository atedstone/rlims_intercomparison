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
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Comparing Landsat and MODIS runoff limits

# %% [markdown]
# ## Background
#
# This script requires the outputs of `compare_coinc.ipynb`, which produces the joined DataFrame of coincident Landsat and MODIS runoff/slush limits.

# %% trusted=true
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import dates
import datetime as dt
import os
import numpy as np

# %% [markdown]
# ## Load dataset

# %% trusted=true
compare = pd.read_csv('/Users/atedston/Dropbox/work/papers/machguth_intercomparison/modis_landsat_xytpd.csv')
#compare['year'] = pd.DatetimeIndex(compare.date_ls).year

# %% [markdown]
# ## Before cleaning, investigate dataset properties

# %% trusted=true
compare.describe()

# %% trusted=true
# Entire (unculled) comparison dataset
# %matplotlib inline
sns.histplot(compare.dist_c, binwidth=1000)
plt.axvline(compare.dist_c.quantile(.75), linestyle=':', color='tab:red')
plt.axvline(compare.dist_c.quantile(.9), linestyle=':', color='tab:pink')
plt.grid(False)
plt.ylabel('Count')
plt.xlabel('Distance difference (m)')

# %% trusted=true
# Look at median absolute deviation of Landsat elevations
# %matplotlib inline
sns.histplot(compare.zmad)
plt.axvline(compare.zmad.quantile(.5), linestyle=':', color='tab:pink')
plt.axvline(compare.zmad.quantile(.9), linestyle=':', color='tab:pink')
plt.axvline(compare.zmad.quantile(.95), linestyle=':', color='tab:pink')

# %% trusted=true
ts = compare.groupby(pd.DatetimeIndex(compare.date_ls).year).dist.median()
ts.index = pd.date_range(dt.datetime(min(ts.index),1,1), dt.datetime(max(ts.index),1,1), freq='1AS')
# %matplotlib inline
ts.plot(marker='o')
plt.grid()
plt.title('Median distance between retrievals')

# %% trusted=true
# Check regression of uncleaned dataset
m = sm.OLS(compare.SL_gimp, sm.add_constant(compare.elev))
r = m.fit()
print(r.summary())

# %% [markdown]
# ### Identify whether MODIS retrievals fall within area of Landsat scene

# %% trusted=true
wrs = gpd.read_file('/home/geoscience/nobackup_cassandra/L0data/WRS2_descending')

# %% trusted=true
from shapely.wkt import loads
# Convert str 'POINT(xx,yy)' from underlying CSV file to proper geometry with the `loads` function.
compare_ll = gpd.GeoDataFrame(compare, geometry=compare['geometry_ls'].apply(loads), crs=3413).to_crs(4326)
insides = {}
for ix, row in compare_ll.iterrows():
    geom = wrs[(wrs.PATH == row.wrs_path) & (wrs.ROW == row.wrs_row)]
    if len(geom) > 0:
        geom = geom.iloc[0].geometry
        inside = row.geometry.within(geom)
    else:
        inside = False
    insides[ix] = inside
    
    

# %% trusted=true
compare.loc[:,'modis_inside_wrs'] = pd.Series(insides)

# %% [markdown]
# ### Compute the elevation difference between Landsat and MODIS

# %% trusted=true
compare.loc[:,'ediff'] = np.abs(compare.elev - compare.SL_gimp)

# %% trusted=true
# %matplotlib inline
sns.ecdfplot(compare.ediff)
plt.axvline(compare.ediff.quantile(.95))

# %% [markdown]
# ### Check the GIMP vs ArcticDEM elevations of MODIS retrievals

# %% trusted=true
# %matplotlib widget
sns.histplot(compare.SL_gimp - compare.SL)

# %% trusted=true
len(compare[np.abs(compare.SL_gimp-compare.SL) > 100])

# %% [markdown]
# ## Apply cleaning/filters and analyse
# These filters are based on inspection of the dataset above.
#

# %% [markdown]
# Rationale for each filter criterion:
#
# * MAD of Landsat elevation retrievals: recall that 'many' Landsat retrievals are averaged together for comparison with the MODIS stripe value. If the MAD is high, this indicates that the Landsat retrievals were at best too heterogeneous for good MODIS comparison, at worst plain wrong. A value of 100 m means that about 90-95% of the data are retained.
# * Only look at comparisons where the MODIS slush limit falls inside the WRS path/row of the Landsat image. If it does not then this is probably a false comparison, likely comparing a Landsat scene which only covers lower in the ablation zone with a MODIS slush limit from much higher up. N.b. this will not work 100% here - if there happens to be a day with multiple Landsat scenes covering the MODIS stripe then only the modal Path/Row will be checked.
# * The previous condition is therefore not 100% sufficient to capture problem scenes like this. Fairly approximate WRS bounds in the shapefile also don't help. So we only retain comparisons where the Landsat detection is at least 500 pixels away from the Landsat scene edge.
# * Finally, we only keep comparisons with less than 200 m elevation difference between MODIS and Landsat. This removes a handful of very clear outliers that I think should have been caught by `modis_inside_wrs`. Remember that 200 m elevation can be a very long distance - at higher elevations, approximately between KAN_U to Dye-2!
#

# %% trusted=true
all_pts =len(compare)
print('All points:', all_pts)
only_mad = len(compare[compare.zmad > 100])
print('Only large MAD:', only_mad)
print(100 / all_pts * only_mad)

# %% trusted=true
# Cleaning begins...
# Step 1: Constrain to Landsat scene bounds and edge proximity.
cleaned = compare[(compare.modis_inside_wrs == True) &
                  (compare.edge_proximity > 500) ]

nvalid1 = len(cleaned)

# %% trusted=true
# Step 2: Threshold on Landsat ZMAD.
cleaned = cleaned[(cleaned.zmad <= 100)]
nvalid2 = len(cleaned)

# %% trusted=true
# Step 3: GIMP vs ADEM for MODIS.
# Look at the difference between GIMP and ADEM for "valid" retrievals

# %% trusted=true
# %matplotlib widget
sns.histplot(cleaned.SL_gimp-cleaned.SL, binwidth=20)

# %% trusted=true
# Step 3 cont'd : apply threshold identified visually above.
cleaned = cleaned[np.abs(cleaned.SL_gimp-cleaned.SL) < 100]
nvalid3 = len(cleaned)

# %% trusted=true
100 - (100 / nvalid1 * nvalid2)

# %% trusted=true
nvalid3

# %% [markdown]
# #### Check out the cleaning-by-elevation-difference

# %% trusted=true
# %matplotlib inline
sns.ecdfplot(cleaned.ediff)
plt.axvline(cleaned.ediff.quantile(.95), linestyle=':')
plt.axvline(cleaned.ediff.quantile(.99), linestyle=':')

# %% trusted=true
# %matplotlib inline
sns.histplot(cleaned.ediff)
plt.axvline(cleaned.ediff.quantile(.99))

# %% trusted=true
len(cleaned[(cleaned.ediff <= cleaned.ediff.quantile(.99)) & (cleaned.elev < cleaned.SL_gimp)])

# %% trusted=true
len(compare)

# %% trusted=true
len(cleaned)

# %% trusted=true
cleaned.ediff.quantile(.99)

# %% trusted=true
len(cleaned[cleaned.ediff > 200])

# %% trusted=true
cleaned[cleaned.ediff > 200].filter(items=['elev', 'SL_gimp', 'date_ls', 'wrs_path', 'wrs_row'])

# %% trusted=true
d = cleaned[cleaned.ediff > 200]
d.geometry_ls = d.geometry_ls.apply(loads)
d.geometry_mo = d.geometry_mo.apply(loads)
# %matplotlib inline
plt.plot([xy.x for xy in d.geometry_ls], [xy.y for xy in d.geometry_ls], '*')
plt.plot([xy.x for xy in d.geometry_mo], [xy.y for xy in d.geometry_mo], '^')


# %% [markdown]
# #### Apply the cleaning-by-elevation-difference

# %% trusted=true
cleaned2 = cleaned[~((cleaned.ediff >= cleaned.ediff.quantile(.99)) & (cleaned.elev < cleaned.SL_gimp))]

# %% [markdown]
# #### Examine cleaning results

# %% trusted=true
# Check regression of cleaned dataset
m = sm.OLS(cleaned.SL_gimp, sm.add_constant(cleaned.elev))
r = m.fit()
print(r.summary())

# %% trusted=true
# Check regression of cleaned dataset
m = sm.OLS(cleaned2.SL_gimp, sm.add_constant(cleaned2.elev))
r = m.fit()
print(r.summary())

# %% trusted=true
# %matplotlib inline
plt.figure(figsize=(3.5,3.5))
plt.plot(cleaned.elev, cleaned.SL_gimp, marker='o', linestyle='none', alpha=0.2, markersize=3)
plt.plot((500,2500), (500,2500), ':', color='grey', alpha=0.6, label='1:1')
xx = np.arange(900, 2300, 100)
yy = r.params['elev'] * xx + r.params['const']
plt.plot(xx, yy, '-', color='k', label='Linear Regression')
plt.legend(edgecolor='none')
sns.despine()
plt.xlim(500, 2500)
plt.ylim(500, 2500)
plt.xticks(np.arange(500,3000,500))
plt.yticks(np.arange(500,3000,500))
plt.xlabel('$\`u$ (Landsat, m)')
plt.ylabel(r'$\Upsilon_S$ (MODIS, m)')
plt.tight_layout()
plt.savefig('/home/tedstona/fig_landsat_modis_scatter.pdf')

# %% trusted=true
# %matplotlib inline
sns.histplot(cleaned.dist_c, binwidth=1000)
plt.axvline(cleaned.dist_c.quantile(.75), linestyle=':', color='tab:red')
plt.axvline(cleaned.dist_c.quantile(.9), linestyle=':', color='tab:pink')


# %% trusted=true
cleaned.describe()

# %% trusted=true
cleaned.columns

# %% trusted=true
cleaned_export = cleaned.filter(items=['date_ls', 'elev', 'geometry_ls', 'zmad', 'edge_proximity', 'wrs_path', 'wrs_row', 'stripe', 'SL_x_coord', 'SL_y_coord', 'geometry_mo', 'SL_gimp', 'SL'], axis=1)
cleaned_export = cleaned_export.rename({'elev':'elev_ls', 'SL_gimp':'elev_mo_gimp', 'SL':'elev_mo_adem',
                                       'edge_proximity':'ls_edge_proximity_px', 'zmad':'ls_zmad'}, axis=1)

# %% trusted=true
cleaned_export

# %% trusted=true
cleaned_export.to_csv('/home/tedstona/modis_landsat_coincident_daily_v20211115_cleaned.csv', index=False)

# %% [markdown]
# ## Conduct a south-west only analysis 

# %% trusted=true
sw_only = compare[(compare.stripe >= -2714750) & (compare.stripe <= -2414750)]

# %% trusted=true
sw_only.describe()

# %% trusted=true
# %matplotlib inline
sns.histplot(data=sw_only, x='elev', y='SL_gimp')

# %% trusted=true
sw_only_cleaned = cleaned[(cleaned.stripe >= -2714750) & (cleaned.stripe <= -2414750)]

# sw_only[(sw_only.zmad <= 100) & #sw_only.zmad.quantile(.5)) &
#                          (sw_only.edge_proximity > 500)]
# #                          (sw_only.elev > sw_only.elev.quantile(.1)) &
m = sm.OLS(sw_only_cleaned.SL_gimp, sm.add_constant(sw_only_cleaned.elev))
r = m.fit()
print(r.summary())

# %% trusted=true
# %matplotlib inline
sns.jointplot(data=sw_only_cleaned, x='elev', y='SL_gimp', alpha=0.2, xlim=(600,2100), ylim=(600,2100))

# %% trusted=true
# %matplotlib inline
plt.plot(sw_only_cleaned.elev, sw_only_cleaned.SL_gimp, 'o', alpha=0.2)
plt.plot(sw_only_cleaned.elev, r.fittedvalues)
plt.plot((1500,2000), (1500,2000), '--')

# %% trusted=true
# %matplotlib inline
ax = plt.subplot(111)
sns.ecdfplot(sw_only.elev, ax=ax, label='Landsat uncleaned')
sns.ecdfplot(sw_only.SL_gimp, ax=ax, label='MODIS uncleaned')
sns.ecdfplot(sw_only_cleaned.elev, ax=ax, label='Landsat cleaned')
sns.ecdfplot(sw_only_cleaned.SL_gimp, ax=ax, label='MODIS cleaned')
plt.axvline(sw_only.elev.quantile(0.1))
plt.legend()

# %% trusted=true
sw_only_cleaned.describe()

# %% trusted=true
# %matplotlib inline
sns.ecdfplot(sw_only_cleaned.ediff)
plt.axvline(sw_only_cleaned.ediff.quantile(.95))

# %% [markdown]
# ### Manual examination of runoff limits with larger differences

# %% trusted=true
to_examine = sw_only_cleaned[sw_only_cleaned['ediff'] > 50]
to_examine = gpd.GeoDataFrame(to_examine, geometry=to_examine['geometry_ls'].apply(loads), crs=3413)

# %% trusted=true
do_plotting = False

if do_plotting:
    # %matplotlib inline

    import geoutils as gu
    from glob import glob

    nn = len(to_examine)
    n = 1
    for ix, examine in to_examine.iterrows():
        print('{n}/{nn}'.format(n=n,nn=nn))
        cand_glob = '*{p}{r}_{date}*'.format(p=str(examine.wrs_path).zfill(3), r=str(examine.wrs_row).zfill(3), date=dt.datetime.strptime(examine.date_ls,'%Y-%m-%d').strftime('%Y%m%d'))
        search_path = os.path.join(os.environ['L0lib'], '*', cand_glob)
        results = glob(search_path)

        for result in results:
            product = result.split('/')[-1]
            if 'LC08' in product:
                band = 'B5'
            else:
                band = 'B4'
            im_fn = os.path.join(result, product + '_' + band + '.TIF')
            if not os.path.exists(im_fn):
                continue
            im = gu.Raster(im_fn, downsample=2)
            plt.figure()
            b = im.bounds
            plt.imshow(im.data[0,:,:], extent=[b.left, b.right, b.bottom, b.top], cmap='Greys_r')
            row = to_examine.loc[[ix]]
            coords = row.to_crs(im.crs)
            x = coords.iloc[0].geometry.x
            y = coords.iloc[0].geometry.y
            plt.plot(x, y, '*', label='Landsat')

            plt.xlim(x-20e3, x+50e3)
            plt.ylim(y-20e3, y+20e3)

            row.geometry = row.geometry_mo.apply(loads)
            coords = row.to_crs(im.crs)
            x = coords.iloc[0].geometry.x
            y = coords.iloc[0].geometry.y
            plt.plot(x, y, '*', Label='MODIS')

            plt.title(product)

            fn_out = product + '.jpg'
            plt.savefig(os.path.join('/', 'flash', 'tedstona', 'landsat_modis', 'mismatches_sw', fn_out), dpi=300)
            plt.close()

        n += 1
