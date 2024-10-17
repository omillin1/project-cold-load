###### Import modules ######
import sys
sys.path.insert(2, '/share/data1/Students/ollie/CAOs/project-cold-load/')
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from utils import funcs
import pandas as pd
import xarray as xr

from argparse import ArgumentParser

parser = ArgumentParser(description='Calculate N American Winter Weather Regimes using method by Lee et al. (2019)',
                        epilog='Example: calc_regimes.py -y 1950 2023 -c 1979 2020')
parser.add_argument('-y','--years',nargs=2,type=int,help='Years to download',default=[1950,2023])
parser.add_argument('-c','--clim',nargs=2,type=int,help='Climo bounds',default=[1979,2020])
args = parser.parse_args()

###### Read in the data ######
# Set bounds for lat and lon.
lat1, lat2 = 80, 20
lon1, lon2 = 180, 330

# Read in hgt data.
hgt_data, time, latitude, longitude = funcs.format_daily_ERA5(var = 'hgt', level = 500, lat_bounds = [lat1, lat2], lon_bounds = [lon1, lon2], year_bounds = [args.years[0], args.years[1]], months = ['01','02','03','11','12'], ndays = 151, isentropic = False)
# Read in T2M data.
t2m_data = funcs.format_daily_ERA5(var = 't2m', level = None, lat_bounds = [lat1, lat2], lon_bounds = [lon1, lon2], year_bounds = [args.years[0], args.years[1]], months = ['01','02','03','11','12'], ndays = 151, isentropic = False)[0]
# Read in SLP data.
slp_data = funcs.format_daily_ERA5(var = 'slp', level = None, lat_bounds = [lat1, lat2], lon_bounds = [lon1, lon2], year_bounds = [args.years[0], args.years[1]], months = ['01','02','03','11','12'], ndays = 151, isentropic = False)[0]

###### Anomalies ######
# Set year array for data.
years_arr = np.arange(args.years[0], args.years[1]+1, 1)

# Find indices of the select climo years.
year_ind1, year_ind2 = np.where(years_arr == args.clim[0])[0][0], np.where(years_arr == args.clim[1])[0][0]

# Find climo for hgt.
climo = np.nanmean(hgt_data[year_ind1:year_ind2+1], axis = 0)
# Find climo for T2M.
climo_t2m = np.nanmean(t2m_data[year_ind1:year_ind2+1], axis = 0)

# Find hgt and T2M anomalies.
anom = hgt_data - climo
anom_t2m = t2m_data - climo_t2m

###### Reshaping ######
# Reshape hgt to get (days, lat, lon)
flat_anom = anom.reshape(anom.shape[0]*anom.shape[1], anom.shape[2], anom.shape[3])
# Reshape T2M to get (days, lat, lon)
flat_anom_t2m = anom_t2m.reshape(anom_t2m.shape[0]*anom_t2m.shape[1], anom_t2m.shape[2], anom_t2m.shape[3])
# Reshape SLP to get (days, lat, lon)
flat_slp = slp_data.reshape(slp_data.shape[0]*slp_data.shape[1], slp_data.shape[2], slp_data.shape[3])
# Reshape time to get (days,)
flat_time = time.reshape(time.shape[0]*time.shape[1])

###### Linearly Detrend GPH and T2M ######
# Set month and year tracker for data.
months_all = np.array([d.month for d in flat_time])
days_all = np.array([d.day for d in flat_time])
ltm_dates = time[0] # Calendar dates for one year.

# Set arrays for linearly detrended GPH and T2M data to be stored.
detrended_gph = np.zeros((flat_anom.shape))
detrended_t2m = np.zeros((flat_anom_t2m.shape))

# Go through each calendar day and detrend the calendar day and place back into detrended array.
J, I = detrended_gph.shape[1], detrended_gph.shape[2] # Lat and lon sizes.
for i in tqdm(range(ltm_dates.shape[0])):
    time_ind = np.where((months_all == ltm_dates[i].month)&(days_all == ltm_dates[i].day))[0] # Get the time indices where time array is a given calendar day.
    T = time_ind.shape[0] # Get size of the number of calendar days.
    detrended_gph[time_ind, :, :] = (funcs.LinearDetrend(flat_anom[time_ind, :, :].reshape(T, J*I))[0]).reshape(T, J, I) # Detrend hgt by calendar day, reshape and store back in the zeros array.
    detrended_t2m[time_ind, :, :] = (funcs.LinearDetrend(flat_anom_t2m[time_ind, :, :].reshape(T, J*I))[0]).reshape(T, J, I) # Detrend T2M by calendar day, reshape and store back in the zeros array.

###### PCA and Clustering ######
# Multiply hgt data by square-root cosine weights
weights = np.sqrt(np.cos(np.radians(latitude)))
z500_anom_sc = detrended_gph*weights[:,np.newaxis]

# Flatten lat-lon grid so that data array is shaped (time, space)
z500_anom_flat = np.reshape(z500_anom_sc, (z500_anom_sc.shape[0],z500_anom_sc.shape[1]*z500_anom_sc.shape[2]))

# Just some code to check how many EOFs lead to 80% explained variance, in our case 12 EOFs.
'''var_to_explain = 0.8
pca = PCA().fit(z500_anom_flat)
var_explained = pca.explained_variance_ratio_
cum_var = var_explained.cumsum()
#
n_eof = np.where(cum_var > var_to_explain)[0].min()'''

# Project data onto leading EOFs
n_eof=12
print("n_eofs",n_eof)
pca = PCA(n_components=n_eof).fit(z500_anom_flat)

# get the principal component timeseries, non-standardised
pc_ts = pca.transform(z500_anom_flat)

# Clustering! This is performed on the PC timeseries
ncluster = 5
print ("Begin K-means clustering...")
print (ncluster, "clusters")
## Do kmeans clustering.
kmeans = KMeans(n_clusters=ncluster, n_init=500, max_iter=500,random_state=42).fit(pc_ts)

# fit a regime to each day
model_clust = kmeans.predict(pc_ts)

# This isn't sorted - the indices are in the order that kmeans does the assignment, and that can vary each time the clustering runs
# Specify the regime ID by its occupation frequency, low-to-high (ArH to PT)
# Calculate what percentage of days fall into each type
ratios = [] # Empty list for ratios of each regime.
for i in range(ncluster): # Loop through number of clusters.
	ratio = 100*(len(np.where(model_clust==i)[0])/float(len(model_clust))) # This is a percentage of all days that are in that regime.
	ratios.append(ratio) # Append the ratio.
# Sort the ratios.
ratios_sorted = sorted(ratios)
# Keep original ratios and ratios sorted.
ratios = np.array(ratios); ratios_sorted=np.array(ratios_sorted)

# make new array to input the new identifiers
new_clust = np.zeros(np.shape(model_clust))
# Loop through each cluster and give each cluster 0-4 based on their occupancy frequency by reassigning the id.
for i in range(ncluster):
	new_id = np.where(ratios_sorted==ratios[i])[0][0]
	new_clust[np.where(model_clust==i)[0]]=new_id

model_clust=new_clust

###### Save the regimes in a txt file here ######
# Change the numbers of regimes into the string name (done via inspection of the maps to confirm the names). Define list to append to.
regime_list = []
for i in range(len(model_clust)):
    if model_clust[i] == 0.0:
        regime_list.append('AkR')
    elif model_clust[i] == 1.0:
        regime_list.append('ArH')
    elif model_clust[i] == 2.0:
        regime_list.append('PT')
    elif model_clust[i] == 3.0:
        regime_list.append('WCR')
    elif model_clust[i] == 4.0:
        regime_list.append('ArL')

# Turn regime indicators into an array.
regime_array = np.array(regime_list)

## Save the timeseries using pandas
weather_types = pd.Series(regime_array, index=flat_time) # Make the regime array a pandas series.
cluster_df = pd.Series.to_frame(weather_types) # Pandas series to dataframe.
cluster_df.to_csv(f'/share/data1/Students/ollie/CAOs/Data/Energy/Regimes/detrended_regimes_{args.years[0]}_{args.years[1]}_NDJFM.txt', sep=' ', index=True) # Save as a txt file.


###### Get composite Z500 and T2M anomalies in each regime ######
# Empty arrays to store the 5 composite patterns.
regime_composite = np.zeros((5,detrended_gph.shape[1],detrended_gph.shape[2])) # hgt
regime_composite_t2m = np.zeros((5,detrended_t2m.shape[1],detrended_t2m.shape[2])) # T2M
regime_composite_slp = np.zeros((5,flat_slp.shape[1],flat_slp.shape[2])) # slp
# Loop through each regime and make a composite of each day in that regime.
for r in range(5):
    subset = np.where(model_clust == r)[0] # Find where the data is in a given regime.
    regime_composite[r] = np.nanmean(detrended_gph[subset],axis=0)/10 # Composite all those days, convert to dam.
    regime_composite_t2m[r] = np.nanmean(detrended_t2m[subset], axis=0) # Composite all those regime days for T2M.
    regime_composite_slp[r] = np.nanmean(flat_slp[subset], axis=0)/100 # Composite all those regime days for slp, convert to hPa.


###### Save the composite maps/ratios for plotting in separate environment with geopandas ######
# Make an xarray dataset.
ds = xr.Dataset(
    data_vars=dict(
        t2m=(["regime","lat","lon"], regime_composite_t2m),
        slp=(["regime","lat","lon"], regime_composite_slp),
        hgt =(["regime","lat","lon"], regime_composite),
        ratio=(["regime"], ratios_sorted)
    ),
    coords=dict(
        regime=np.array(['AkR', 'ArH', 'PT', 'WCR', 'ArL']),
        lat = latitude,
        lon = longitude
    ),
    attrs=dict(description=f"Regime composites"),
)

# Write to netcdf.
ds.to_netcdf(f'/share/data1/Students/ollie/CAOs/Data/Energy/Regime_Comps/regime_comps.nc')