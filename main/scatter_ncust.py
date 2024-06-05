###### Import modules ######
import sys
sys.path.insert(2, '/share/data1/Students/ollie/CAOs/project-cold-load')
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import funcs

###### Set up balancing authorities to retrieve and years to use ######

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

# Now stack to get shape (regions, year).
all_cust = np.stack(cust_region)/1000 # For thousands of customers.

# Sum customers across region. Will give shape (year,)
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
lat1, lat2 = 43, 31
lon1, lon2 = 256, 268
# Min 2m temperature.
era5_tmin, era5_time, era5_lat, era5_lon = funcs.minmax_2d_ERA5(var = 'Tmin', lat_bounds = [lat1, lat2], lon_bounds = [lon1, lon2], year_bounds = [year1, year2], months = ['01', '02', '03', '11', '12'], ndays = 151)
# Daily average 2 m temperature.
era5_t2m = funcs.format_daily_ERA5(var = 't2m', level = None, lat_bounds = [lat1, lat2], lon_bounds = [lon1, lon2], year_bounds = [year1, year2], months = ['01', '02', '03', '11', '12'], ndays = 151, isentropic = False)[0]

# Latitude average the temperature data.
weights = np.cos(np.radians(era5_lat)) # Cosine latitude weights.
lat_ave_t2m = np.average(era5_t2m, weights = weights, axis = 2) # Latitude weighted average for daily ave T2M.
lat_ave_tmin = np.average(era5_tmin, weights = weights, axis = 2) # Latitude weighted average for daily min T2M.

# Average across longitude.
area_ave_t2m = np.nanmean(lat_ave_t2m, axis = -1) # Daily average T2M.
area_ave_tmin = np.nanmean(lat_ave_tmin, axis = -1) # Daily min T2M.

# Reshape T2M daily average data to (days,).
t2m_reshape = area_ave_t2m.reshape(area_ave_t2m.shape[0]*area_ave_t2m.shape[1])-273.15 # For degrees C.
t2m_time_reshape = era5_time.reshape(era5_time.shape[0]*era5_time.shape[1])
# Reshape T2M min data to (days,).
tmin_reshape = area_ave_tmin.reshape(area_ave_tmin.shape[0]*area_ave_tmin.shape[1])-273.15 # For degrees C.

###### Now get corresponding T2M data for each peak load day ######
t2m_peak_load = np.zeros((anom_peak_load.shape)) # Array to store daily average T2M for peak load dates.
tmin_peak_load = np.zeros((anom_peak_load.shape)) # Array to store daily min T2M for peak load dates.
for i in range(len(anom_peak_load)): # Loop through the shape of peak load dates.
    ind = np.where(t2m_time_reshape == date_load[i])[0][0] # Find where the ERA5 time is the same as your dates for peak load.
    t2m_peak_load[i] = t2m_reshape[ind] # Store the ERA5 T2M daily average for that date.
    tmin_peak_load[i] = tmin_reshape[ind] # Store the ERA5 T2M daily min for that date.


###### Polynomial Function ######
X_fit, Y_fit, a, b, c, eqn_t2m = funcs.plot_fit(t2m_peak_load, anom_peak_load) # Fit a polynomial function for peak load-daily average T2M.
X_fit_tmin, Y_fit_tmin, amin, bmin, cmin, eqn_tmin = funcs.plot_fit(tmin_peak_load, anom_peak_load) # Fit a polynomial function for peak load-daily min T2M.

###### Calculate coefficient of determination for each polynomial ######
R2_t2m = funcs.rsquared(anom_peak_load, eqn_t2m(t2m_peak_load))
R2_tmin = funcs.rsquared(anom_peak_load, eqn_tmin(tmin_peak_load))

###### Plot scatter plots for each model ######
# First, anomalous peak load against daily average T2M ######
fig = plt.figure(figsize = (10,5)) # Generate figure.
ax = fig.add_subplot(1, 2, 1) # Add subplot.
plt.scatter(t2m_peak_load, anom_peak_load, color = 'black', s = 0.5) # Scatter the data.
plt.plot(X_fit, Y_fit, color = 'red') # Plot the polynomial model.
plt.axhline(y=0, lw = 2, color = 'blue', ls = '--')
plt.xlabel("Temperature ($^\circ$C)", weight = 'bold', fontsize = 13) # Add xlabel.
plt.ylabel("Peak Load Anomaly (MWh/1000 Customers)", weight = 'bold', fontsize = 12) # Add y label.
plt.xticks(np.arange(-24, 30, 6)) # Add x ticks.
plt.yticks(np.arange(-1.5, 3.5, 0.5)) # Add yticks.
plt.xlim([-24, 24]) # Add x lim.
plt.ylim([-1.5, 3.0]) # Add y lim.
plt.text(-18, -1, f"R$^{2}$ = {np.round(R2_t2m, 2)}") # Plot text with the R squared value.
plt.text(-6, 2.5, f"Peak Load = {np.round(a, 3)}T$^{2}$ {np.round(b, 3)}T + {np.round(c, 3)}", fontsize = 9) # Plot the equation for the polynomial.
plt.title("a) Daily Mean T2M", weight = 'bold', fontsize = 14) # Add title.
plt.tight_layout() # Tight layout.

# Second, anomalous peak load against daily min T2M ######
ax = fig.add_subplot(1, 2, 2) # Add subplot.
plt.scatter(tmin_peak_load, anom_peak_load, color = 'black', s = 0.5) # Scatter the data.
plt.plot(X_fit_tmin, Y_fit_tmin, color = 'red') # Plot the polynomial model.
plt.axhline(y=0, lw = 2, color = 'blue', ls = '--')
plt.xlabel("Temperature ($^\circ$C)", weight = 'bold', fontsize = 13) # Add xlabel.
plt.xticks(np.arange(-24, 30, 6)) # Add x ticks.
plt.yticks(np.arange(-1.5, 3.5, 0.5)) # Add y ticks.
plt.xlim([-24, 24]) # Add xlim.
plt.ylim([-1.5, 3.0]) # Add ylim.
plt.text(-18, -1, f"R$^{2}$ = {np.round(R2_tmin, 2)}") # Plot text with the R squared value.
plt.text(-6, 2.5, f"Peak Load = {np.round(amin, 3)}T$^{2}$ {np.round(bmin, 3)}T  {np.round(cmin, 3)}", fontsize = 9) # Plot the equation for the polynomial.
plt.title("b) Daily Min T2M", weight = 'bold', fontsize = 14) # Add title.
plt.tight_layout() # Tight layout.
plt.savefig("/share/data1/Students/ollie/CAOs/project-cold-load/Figures/Scatter/scatter_temp.png", bbox_inches = 'tight', dpi = 500) # Save figure.
