###### Import modules ######
from calendar import month
import sys
sys.path.insert(2, '/share/data1/Students/ollie/CAOs/project-cold-load')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import AxesGrid
import scipy.stats
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape, Polygon, Point
from shapely import vectorized
import geopandas as gpd
from utils import funcs

###### Set up regions to retrieve and years to use ######
divs = ['KACY', 'WR', 'SPS', 'OKGE', 'CSWS', 'SECI', 'WFEC', 'EDE', 'NPPD', 'OPPD', 'KCPL', 'MPS']
year1, year2 = 1999, 2022

###### Get dates for time period ######
dates_arr = funcs.read_early_load(div = 'OKGE', year1 = year1, month_bnd = [3, 11])[0] # 1999-2010 dates.
dates_arr2 = funcs.read_late_load(2011, year2, months=['01', '02', '03', '04', '10', '11', '12'], month_bnd = [3, 11], div = 'OKGE')[0] # 2011-2022 dates.
# Join the dates.
all_dates = np.concatenate((dates_arr, dates_arr2))


###### Loop through the regions and get the data ######
data_region = [] # List to append the data for each region.
for i in tqdm(range(len(divs))): # Loop through divisions
    load_early = funcs.read_early_load(div = divs[i], year1 = 1999, month_bnd = [3, 11])[1] # Get the 1999-2010 hourly load.
    load_late = funcs.read_late_load(2011, 2022, months=['01', '02', '03', '04', '10', '11', '12'], month_bnd = [3, 11], div = divs[i])[1] # Get the 2011-2022 hourly load.
    load_comb = np.concatenate((load_early, load_late)) # Join the loads for the two periods.
    data_region.append(load_comb) # Append the joined load to the empty list.

# Now vstack, to get an array of shape (region, time).
all_regions = np.stack(data_region)

###### Now for peak load ######
# Reshape the load and dates into shape (div, days, 24 hours) and (days, 24 hours)
load_reshape = all_regions.reshape(len(divs), int(all_regions.shape[1]/24), 24)
date_reshape = all_dates.reshape(int(all_dates.shape[0]/24), 24)

# Now sum the data for all regions (i.e., total of SPP regions we have).
sum_load = np.nansum(load_reshape, axis = 0)

# Now find the max for each day for peak load (i.e., take maximum for each 24 hours).
peak_load = np.nanmax(sum_load, axis = 1)
date_load = date_reshape[:, 0] # This is to get a date for each day on the record (midnight selected).

###### Scale by number of customers ######
# Get a years, months, and days tracker.
years_all = np.array([d.year for d in date_load])
months_all = np.array([d.month for d in date_load])
days_all = np.array([d.day for d in date_load])
# Get an array of dates for one complete year for reference later. 2012 in this case.
dates_select = date_load[np.where(years_all == 2012)[0]]

# Now read number of customers by division.
cust_region = [] # Empty list to store customer numbers.
for i in tqdm(range(len(divs))): # Loop through divisions.
    year_cust, data_cust = funcs.read_cust_data(div = divs[i], year1 = year1, year2 = year2, param = 'Customers') # Get the customer number data

    cust_region.append(data_cust) # Append customer number by year to the empty list.

# Now vstack to get shape (regions, year).
all_cust = np.stack(cust_region)/1000 # For thousands of customers.

# Sum for all customers by region. Will give shape (year,)
sum_cust = np.nansum(all_cust, axis = 0)

# Now go through and scale the peak load by number of customers.
norm_peak_load = np.zeros(peak_load.size) # Empty array to store the peak load when scaled.
for i in range(len(year_cust)): # Loop through the years of customers.
    time_ind = np.where(years_all == year_cust[i])[0] # For each year, find where the load dates are that year.
    norm_peak_load[time_ind] = peak_load[time_ind]/sum_cust[i] # Then take the peak load for those dates and divide by that years customers.

###### Calculate anomalies by calendar day ######
anom_peak_load = np.zeros((norm_peak_load.size)) # Empty array to store anomalies.
for i in tqdm(range(dates_select.shape[0])): # Loop through each calendar day.
    time_ind = np.where((months_all == dates_select[i].month)&(days_all == dates_select[i].day))[0] # Find the indices where it is a given calendar day in the peak load dates.
    mean = np.nanmean(norm_peak_load[time_ind]) # Take the mean of those data points across the whole time series.
    anom_peak_load[time_ind] = (norm_peak_load[time_ind]-mean) # Now take the scaled peak load for a given calendar day, subtract the mean, and store it in the right indices.

###### Read in the ERA5 T2M data ######
# Lat and lons for the T2M area.
lat1, lat2 = 80, 20
lon1, lon2 = 180, 330
# Daily average 2 m temperature.
era5_t2m, era5_time, era5_lat, era5_lon = funcs.format_daily_ERA5(var = 't2m', level = None, lat_bounds = [lat1, lat2], lon_bounds = [lon1, lon2], year_bounds = [1991, year2], months = ['01', '02', '03', '11', '12'], ndays = 151, isentropic = False)

###### Now get anomalies ######
years_arr = np.arange(1991, year2+1, 1)

ltm = np.nanmean(era5_t2m[np.where(years_arr == 1991)[0][0]:np.where(years_arr == 2020)[0][0]+1], axis = 0)

anom = era5_t2m - ltm

# Reshape T2M ddaily average data to (days,).
t2m_reshape = anom.reshape(anom.shape[0]*anom.shape[1], anom.shape[2], anom.shape[3]) # For degrees C.
t2m_time_reshape = era5_time.reshape(era5_time.shape[0]*era5_time.shape[1])

###### Now get corresponding T2M data for each peak load day ######
t2m_peak_load = np.zeros((anom_peak_load.shape[0], era5_lat.shape[0], era5_lon.shape[0])) # Array to store daily average T2M for peak load dates.
for i in range(len(anom_peak_load)): # Loop through the shape of peak load dates.
    ind = np.where(t2m_time_reshape == date_load[i])[0][0] # Find where the ERA5 time is the same as your dates for peak load.
    t2m_peak_load[i] = t2m_reshape[ind] # Store the ERA5 T2M daily average for that date.


###### Read in regimes ######
# Go to directory.
directory = '/share/data1/Students/ollie/CAOs/Data/Energy/Regimes/'
path = os.chdir(directory)
# Get filename of regimes file.
filename = 'detrended_regimes_1950_2023_NDJFM.txt'
# Set headers for txt file.
headers = ['Day', 'Regime']

# Read in data for txt file of regimes.
data = pd.read_csv(filename, delim_whitespace=True, skiprows = [0], names=headers, index_col=False)

# Get regimes and corresponding days.
days = data['Day'].values
regimes = data['Regime'].values

# Turn dates into datetimes.
dates_regime_list = [] # Empty list to store datetime objects.
for i in range(len(days)): # Loop through days (string format).
    dates_regime_list.append(datetime.strptime(days[i], '%Y-%m-%d')) # Append associated datetime.
# Convert list of dates to a python array.
dates_regime = np.array(dates_regime_list)

###### Now get the regime for each load day ######
regime_peak_load_list = [] # Empty list to append the regime for each peak load day.
for i in range(len(date_load)): # Loop through each date in the peak loads.
    ind = np.where(dates_regime == date_load[i])[0][0] # Find the index where the regime dates equals the chosen peak load date.
    regime_peak_load_list.append(regimes[ind]) # Append the regime to the regime list.
# Make the list of regimes for peak load days into an array.
regimes_peak_load = np.asarray(regime_peak_load_list)

###### Split the anomalous peak load by weather regime ######
akr_load = anom_peak_load[np.where(regimes_peak_load == 'AkR')[0]] # AkR.
arh_load = anom_peak_load[np.where(regimes_peak_load == 'ArH')[0]] # ArH.
pt_load = anom_peak_load[np.where(regimes_peak_load == 'PT')[0]] # PT.
wcr_load = anom_peak_load[np.where(regimes_peak_load == 'WCR')[0]] # WCR.
arl_load = anom_peak_load[np.where(regimes_peak_load == 'ArL')[0]] # ArL.


###### Now get percentile thresholds of the load categories ######
perc_akr = np.percentile(akr_load, q = 90)
perc_arh = np.percentile(arh_load, q = 90)
perc_wcr = np.percentile(wcr_load, q = 90)
perc_pt = np.percentile(pt_load, q = 90)
perc_arl = np.percentile(arl_load, q = 90)

###### Now find where the threshold is met and is a given regime ######
ind_akr = np.where((regimes_peak_load == 'AkR')&(anom_peak_load >= perc_akr))[0]
ind_arh = np.where((regimes_peak_load == 'ArH')&(anom_peak_load >= perc_arh))[0]
ind_wcr = np.where((regimes_peak_load == 'WCR')&(anom_peak_load >= perc_wcr))[0]
ind_pt = np.where((regimes_peak_load == 'PT')&(anom_peak_load >= perc_pt))[0]
ind_arl = np.where((regimes_peak_load == 'ArL')&(anom_peak_load >= perc_arl))[0]

###### Now get the T2M patterns associated with each one ######

t2m_akr = np.nanmean(t2m_peak_load[ind_akr], axis = 0)
t2m_arh = np.nanmean(t2m_peak_load[ind_arh], axis = 0)
t2m_wcr = np.nanmean(t2m_peak_load[ind_wcr], axis = 0)
t2m_pt = np.nanmean(t2m_peak_load[ind_pt], axis = 0)
t2m_arl = np.nanmean(t2m_peak_load[ind_arl], axis = 0)

###### Now get the Southwest Power Pool shapefile ######
dir = '/share/data1/Students/ollie/CAOs/Data/Energy/Shapefiles/'
path = os.chdir(dir) # Go to directory.

# Open the shapefile
shpfile = gpd.read_file('Control__Areas.shp')
shpfile = shpfile.to_crs('EPSG:4326') # Convert to standard, i.e., PlateCarree.

# Select the multipolygon.
geometry = shape(shpfile['geometry'].iloc[-1]) # SPP is in the last index.


poly = geometry.geoms[-11]

# Convert lons.
new_lon = np.zeros(era5_lon.size)
for i in range(len(era5_lon)):
    if era5_lon[i] <= 180:
        new_lon[i] = era5_lon[i]
    else:
        new_lon[i] = era5_lon[i]-360

bin = np.zeros((era5_lat.size, new_lon.size))
for i in range(len(new_lon)):
    for j in range(len(era5_lat)):
        t = Point([new_lon[i], era5_lat[j]]).within(poly)
        if t == True:
            bin[j, i] = 1
        else:
            bin[j, i] = 0

# Set updated lats and lons.

# Mesh the grid.
lons, lats = np.meshgrid(era5_lon, era5_lat)

upd_lon1, upd_lon2 = 250, 270
upd_lat1, upd_lat2 = 50, 27.5

# Set colormap.
clevs_t2m = np.arange(-10, 10.5, 0.5)
my_cmap_t2m, norm_t2m = funcs.NormColorMap('RdBu_r', clevs_t2m)

# Plot T2M next
fig = plt.figure(figsize=(12, 9.6))
ax1 = plt.subplot2grid(shape = (4,6), loc = (0,0), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255))
#cs = ax1.contourf(lons, lats, np.ma.array(t2m_akr, mask= (bin == 0)), clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m)
cs = ax1.contourf(lons, lats, t2m_akr, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m)
for poly in geometry.geoms:
    if poly.area > 5:
        x,y = poly.exterior.xy
        ax1.plot(x,y,transform=ccrs.PlateCarree(), color = 'k', lw = 2.5)
    else:
        pass
ax1.coastlines()
ax1.add_feature(cfeature.BORDERS)
ax1.add_feature(cfeature.STATES,linewidth =1.5, edgecolor = 'darkslategray')
ax1.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree())
plt.title(f"(a) AkR (n = {ind_akr.size})",weight="bold", fontsize = 15)
plt.tight_layout()

ax2 = plt.subplot2grid(shape = (4,6), loc = (0,2), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255))
cs = ax2.contourf(lons, lats, t2m_arh, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m)
for poly in geometry.geoms:
    if poly.area > 5:
        x,y = poly.exterior.xy
        ax2.plot(x,y,transform=ccrs.PlateCarree(), color = 'k', lw = 2.5)
    else:
        pass
ax2.coastlines()
ax2.add_feature(cfeature.BORDERS)
ax2.add_feature(cfeature.STATES,linewidth =1.5, edgecolor = 'darkslategray')
ax2.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree())
plt.title(f"(b) ArH (n = {ind_arh.size})",weight="bold", fontsize = 15)
plt.tight_layout()

ax3 = plt.subplot2grid(shape = (4,6), loc = (0,4), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255))
cs = ax3.contourf(lons, lats, t2m_pt, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m)
for poly in geometry.geoms:
    if poly.area > 5:
        x,y = poly.exterior.xy
        ax3.plot(x,y,transform=ccrs.PlateCarree(), color = 'k', lw = 2.5)
    else:
        pass
ax3.coastlines()
ax3.add_feature(cfeature.BORDERS)
ax3.add_feature(cfeature.STATES,linewidth =1.5, edgecolor = 'darkslategray')
ax3.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree())
plt.title(f"(c) PT (n = {ind_pt.size})",weight="bold", fontsize = 15)
plt.tight_layout()

ax4 = plt.subplot2grid(shape = (4,6), loc = (2,1), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255))
cs = ax4.contourf(lons, lats, t2m_wcr, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m)
for poly in geometry.geoms:
    if poly.area > 5:
        x,y = poly.exterior.xy
        ax4.plot(x,y,transform=ccrs.PlateCarree(), color = 'k', lw = 2.5)
    else:
        pass
ax4.coastlines()
ax4.add_feature(cfeature.BORDERS)
ax4.add_feature(cfeature.STATES, linewidth =1.5, edgecolor = 'darkslategray')
ax4.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree())
plt.title(f"(d) WCR (n = {ind_wcr.size})",weight="bold", fontsize = 15)
plt.tight_layout()

ax5 = plt.subplot2grid(shape = (4,6), loc = (2,3), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255))
cs = ax5.contourf(lons, lats, t2m_arl, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m)
for poly in geometry.geoms:
    if poly.area > 5:
        x,y = poly.exterior.xy
        ax5.plot(x,y,transform=ccrs.PlateCarree(), color = 'k', lw = 2.5)
    else:
        pass
ax5.coastlines()
ax5.add_feature(cfeature.BORDERS)
ax5.add_feature(cfeature.STATES, linewidth =1.5, edgecolor = 'darkslategray')
ax5.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree())
plt.title(f"(e) ArL (n = {ind_arl.size})",weight="bold", fontsize = 15)
plt.tight_layout()

cb_ax = fig.add_axes([0.05, -0.02, 0.91, 0.04])
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks=np.arange(-10, 12, 2),extend="both",spacing='proportional')
cbar.set_label("2m Temperature Anomaly ($^\circ$C)", fontsize = 14)
cbar.ax.tick_params(labelsize=12)
plt.savefig("/share/data1/Students/ollie/CAOs/project-cold-load/Figures/Regimes/regimes_extrload_t2m.png", bbox_inches = 'tight', dpi = 500)