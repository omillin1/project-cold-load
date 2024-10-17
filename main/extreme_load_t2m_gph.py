###### Import modules ######
import sys
sys.path.insert(2, '/share/data1/Students/ollie/CAOs/project-cold-load')
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import shape, Point
from utils import funcs

###### Set up balancing authorities to retrieve and years to use ######
divs = ['KACY', 'WR', 'SPS', 'OKGE', 'CSWS', 'SECI', 'WFEC', 'EDE', 'NPPD', 'OPPD', 'KCPL', 'MPS'] # Our 13 balancing authorities (12 here but KCPL is merged with INDN in the functions).
year1, year2 = 1999, 2022 # Years to choose.

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

# Now stack, to get an array of shape (region, time).
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
    year_cust, data_cust = funcs.read_cust_data(div = divs[i], year1 = year1, year2 = year2, param = 'Customers') # Get the customer number data.
    cust_region.append(data_cust) # Append customer number by year to the empty list.

# Now stack to get shape (regions, year).
all_cust = np.stack(cust_region)/1000 # For thousands of customers.

# Sum customers across region. Will give shape (year,).
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

###### Read in the ERA5 T2M and 500 hPa GPH data ######
# Lat and lons for the T2M area.
lat1, lat2 = 80, 20
lon1, lon2 = 180, 330
# Daily average 2 m temperature.
era5_t2m, era5_time, era5_lat, era5_lon = funcs.format_daily_ERA5(var = 't2m', level = None, lat_bounds = [lat1, lat2], lon_bounds = [lon1, lon2], year_bounds = [1991, year2], months = ['01', '02', '03', '11', '12'], ndays = 151, isentropic = False)
# Daily average GPH.
era5_hgt = funcs.format_daily_ERA5(var = 'hgt', level = 500, lat_bounds = [lat1, lat2], lon_bounds = [lon1, lon2], year_bounds = [1991, year2], months = ['01', '02', '03', '11', '12'], ndays = 151, isentropic = False)[0]


###### Now get anomalies ######
# Array of years to read in.
years_arr = np.arange(1991, year2+1, 1)
# Select climo years.
climo_year1, climo_year2 = 1991, 2020
# Take the mean of the T2M array between selected climo years.
ltm_t2m = np.nanmean(era5_t2m[np.where(years_arr == climo_year1)[0][0]:np.where(years_arr == climo_year2)[0][0]+1], axis = 0)
# Take the mean of the GPH array between selected climo years.
ltm_hgt = np.nanmean(era5_hgt[np.where(years_arr == climo_year1)[0][0]:np.where(years_arr == climo_year2)[0][0]+1], axis = 0)
# Get anomalies.
anom_t2m = era5_t2m - ltm_t2m
anom_hgt = (era5_hgt - ltm_hgt)/10 # For dam.

# Reshape T2M and hgt daily average data and dates to (days,) in the first dimension.
t2m_reshape = anom_t2m.reshape(anom_t2m.shape[0]*anom_t2m.shape[1], anom_t2m.shape[2], anom_t2m.shape[3])
hgt_reshape = anom_hgt.reshape(anom_hgt.shape[0]*anom_hgt.shape[1], anom_hgt.shape[2], anom_hgt.shape[3])
t2m_time_reshape = era5_time.reshape(era5_time.shape[0]*era5_time.shape[1])

###### Now get corresponding T2M and 500 hPa GPH data for each peak load day ######
t2m_peak_load = np.zeros((anom_peak_load.shape[0], era5_lat.shape[0], era5_lon.shape[0])) # Array to store daily average T2M for peak load dates.
hgt_peak_load = np.zeros((anom_peak_load.shape[0], era5_lat.shape[0], era5_lon.shape[0])) # Array to store daily average hgt for peak load dates.
for i in range(len(anom_peak_load)): # Loop through the shape of peak load dates.
    ind = np.where(t2m_time_reshape == date_load[i])[0][0] # Find where the ERA5 time is the same as your dates for peak load.
    t2m_peak_load[i] = t2m_reshape[ind] # Store the ERA5 T2M daily average for that date.
    hgt_peak_load[i] = hgt_reshape[ind] # Store the ERA5 hgt daily average for that date.

###### Read in regimes ######
# Go to directory.
directory = '/share/data1/Students/ollie/CAOs/Data/Energy/Regimes/'
path = os.chdir(directory)
# Get filename of regimes file.
filename = 'detrended_regimes_1950_2023_NDJFM.txt'
# Read the regimes in with dates.
regimes, dates_regime = funcs.read_regimes(filename)

###### Now get the regime for each load day ######
regime_peak_load_list = [] # Empty list to append the regime for each peak load day.
for i in range(len(date_load)): # Loop through each date in the peak loads.
    ind = np.where(dates_regime == date_load[i])[0][0] # Find the index where the regime dates equals the chosen peak load date.
    regime_peak_load_list.append(regimes[ind]) # Append the regime to the regime list.
# Make the list of regimes for peak load days into an array.
regimes_peak_load = np.array(regime_peak_load_list)

###### Now get composites of extreme demand days ######
selected_perc = [90, 95, 97.5]
# Set updated lat/lon for projection.
upd_lon1, upd_lon2 = 180, 330
upd_lat1, upd_lat2 = 80, 20

# Set colormap.
clevs_t2m = np.arange(-12, 12.5, 0.5)
my_cmap_t2m, norm_t2m = funcs.NormColorMap('RdBu_r', clevs_t2m)

# Mesh the grid.
lons, lats = np.meshgrid(era5_lon, era5_lat)

###### Set up labels and titles ######
perc1_titles = ['a) All 90th', 'd) AkR 90th', 'g) ArH 90th', 'j) WCR 90th', 'm) PT 90th', 'p) ArL 90th']
perc2_titles = ['b) All 95th', 'e) AkR 95th', 'h) ArH 95th', 'k) WCR 95th', 'n) PT 95th', 'q) ArL 95th']
perc3_titles = ['c) All 97.5th', 'f) AkR 97.5th', 'i) ArH 97.5th', 'l) WCR 97.5th', 'o) PT 97.5th', 'r) ArL 97.5th']

perc_labels = [perc1_titles, perc2_titles, perc3_titles]

###### Plot ######
fig = plt.figure(figsize=(10, 10.5))
# Loop through the percentiles.
for i in tqdm(range(len(selected_perc))):
    percval = np.nanpercentile(anom_peak_load, q = selected_perc[i]) # Find the extreme threshold of all data.
    pt_inds = np.where((anom_peak_load >= percval)&(regimes_peak_load == 'PT'))[0] # Now find where it is PT and exceeds the threshold.
    arl_inds = np.where((anom_peak_load >= percval)&(regimes_peak_load == 'ArL'))[0] # Now find where it is ArL and exceeds the threshold.
    akr_inds = np.where((anom_peak_load >= percval)&(regimes_peak_load == 'AkR'))[0] # Now find where it is AkR and exceeds the threshold.
    wcr_inds = np.where((anom_peak_load >= percval)&(regimes_peak_load == 'WCR'))[0] # Now find where it is WCR and exceeds the threshold.
    arh_inds = np.where((anom_peak_load >= percval)&(regimes_peak_load == 'ArH'))[0] # Now find where it is ArH and exceeds the threshold.
    all_inds = np.where(anom_peak_load >= percval)[0] # Now find where it totally exceeds the threshold.

    # Now composites.
    pt_hgt, pt_t2m = np.nanmean(hgt_peak_load[pt_inds], axis = 0), np.nanmean(t2m_peak_load[pt_inds], axis = 0) # Average hgt and T2M for the PT extreme days.
    arl_hgt, arl_t2m = np.nanmean(hgt_peak_load[arl_inds], axis = 0), np.nanmean(t2m_peak_load[arl_inds], axis = 0) # Average hgt and T2M for the ArL extreme days.
    akr_hgt, akr_t2m = np.nanmean(hgt_peak_load[akr_inds], axis = 0), np.nanmean(t2m_peak_load[akr_inds], axis = 0) # Average hgt and T2M for the AkR extreme days.
    wcr_hgt, wcr_t2m = np.nanmean(hgt_peak_load[wcr_inds], axis = 0), np.nanmean(t2m_peak_load[wcr_inds], axis = 0) # Average hgt and T2M for the WCR extreme days.
    arh_hgt, arh_t2m = np.nanmean(hgt_peak_load[arh_inds], axis = 0), np.nanmean(t2m_peak_load[arh_inds], axis = 0) # Average hgt and T2M for the ArH extreme days.
    all_hgt, all_t2m = np.nanmean(hgt_peak_load[all_inds], axis = 0), np.nanmean(t2m_peak_load[all_inds], axis = 0) # Average hgt and T2M for all extreme days.


    ###### Now plot the patterns ######
    # Subplot.
    ax1 = fig.add_subplot(6, 3, i+1, projection=ccrs.PlateCarree(central_longitude = 255))
    cs = ax1.contourf(lons, lats, all_t2m, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf all extreme days T2M.
    lines = ax1.contour(lons, lats, all_hgt, [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25], colors = 'black', transform=ccrs.PlateCarree()) # Contour all extreme days hgt.
    # Contour labels.
    ax1.clabel(lines, colors = 'black')
    ax1.coastlines() # Plot coastlines.
    ax1.add_feature(cfeature.BORDERS) # Plot borders.
    ax1.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
    ax1.set_title(f"{perc_labels[i][0]} (n={all_inds.size})",weight="bold", fontsize = 15) # Set title.
    plt.tight_layout() # Tight layout.

    ax2 = fig.add_subplot(6, 3, i+4, projection=ccrs.PlateCarree(central_longitude = 255))
    cs = ax2.contourf(lons, lats, akr_t2m, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf AkR extreme days T2M.
    lines = ax2.contour(lons, lats, akr_hgt, [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25], colors = 'black', transform=ccrs.PlateCarree()) # Contour AkR extreme days hgt.
    ax2.clabel(lines, colors = 'black')
    ax2.coastlines() # Plot coastlines.
    ax2.add_feature(cfeature.BORDERS) # Plot borders.
    ax2.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
    ax2.set_title(f"{perc_labels[i][1]} (n={akr_inds.size})",weight="bold", fontsize = 15) # Set title.
    plt.tight_layout() # Tight layout.

    ax3 = fig.add_subplot(6, 3, i+7, projection=ccrs.PlateCarree(central_longitude = 255))
    cs = ax3.contourf(lons, lats, arh_t2m, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf ArH extreme days T2M.
    lines = ax3.contour(lons, lats, arh_hgt, [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25], colors = 'black', transform=ccrs.PlateCarree()) # Contour ArH extreme days hgt.
    ax3.clabel(lines, colors = 'black')
    ax3.coastlines() # Plot coastlines.
    ax3.add_feature(cfeature.BORDERS) # Plot borders.
    ax3.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
    ax3.set_title(f"{perc_labels[i][2]} (n={arh_inds.size})",weight="bold", fontsize = 15) # Set title.
    plt.tight_layout() # Tight layout.

    ax4 = fig.add_subplot(6, 3, i+10, projection=ccrs.PlateCarree(central_longitude = 255))
    cs = ax4.contourf(lons, lats, wcr_t2m, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf WCR extreme days T2M.
    lines = ax4.contour(lons, lats, wcr_hgt, [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25], colors = 'black', transform=ccrs.PlateCarree()) # Contour WCR extreme days hgt.
    ax4.clabel(lines, colors = 'black')
    ax4.coastlines() # Plot coastlines.
    ax4.add_feature(cfeature.BORDERS) # Plot borders.
    ax4.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
    ax4.set_title(f"{perc_labels[i][3]} (n={wcr_inds.size})",weight="bold", fontsize = 15) # Set title.
    plt.tight_layout() # Tight layout.

    ax5 = fig.add_subplot(6, 3, i+13, projection=ccrs.PlateCarree(central_longitude = 255))
    cs = ax5.contourf(lons, lats, pt_t2m, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf PT extreme days T2M.
    lines = ax5.contour(lons, lats, pt_hgt, [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25], colors = 'black', transform=ccrs.PlateCarree()) # Contour PT extreme days hgt.
    ax5.clabel(lines, colors = 'black')
    ax5.coastlines() # Plot coastlines.
    ax5.add_feature(cfeature.BORDERS) # Plot borders.
    ax5.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
    ax5.set_title(f"{perc_labels[i][4]} (n={pt_inds.size})",weight="bold", fontsize = 15) # Set title.
    plt.tight_layout() # Tight layout.

    ax6 = fig.add_subplot(6, 3, i+16, projection=ccrs.PlateCarree(central_longitude = 255))
    cs = ax6.contourf(lons, lats, arl_t2m, clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf ArL extreme days T2M.
    lines = ax6.contour(lons, lats, arl_hgt, [-25, -20, -15, -10, -5, 5, 10, 15, 20, 25], colors = 'black', transform=ccrs.PlateCarree()) # Contour ArL extreme days hgt.
    ax6.clabel(lines, colors = 'black')
    ax6.coastlines() # Plot coastlines.
    ax6.add_feature(cfeature.BORDERS) # Plot borders.
    ax6.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
    ax6.set_title(f"{perc_labels[i][5]} (n={arl_inds.size})",weight="bold", fontsize = 15) # Set title.
    plt.tight_layout() # Tight layout.

fig.tight_layout()

# Set up colorbar.
cb_ax = fig.add_axes([0.05, -0.02, 0.91, 0.04]) # Axes for colorbar.
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks=np.arange(-12, 15, 3),extend="both",spacing='proportional') # Plot colorbar.
cbar.set_label("2m Temperature Anomaly ($^\circ$C)", fontsize = 14) # Colorbar label.
cbar.ax.tick_params(labelsize=12) # Colorbar ticks.
plt.savefig("/share/data1/Students/ollie/CAOs/project-cold-load/Figures/Regimes/extreme_load_circ_panel.png", bbox_inches = 'tight', dpi = 500)







