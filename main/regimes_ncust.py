###### Import modules ######
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import sys
sys.path.insert(2, '/share/data1/Students/ollie/CAOs/project-cold-load')
import numpy as np
import pandas as pd
from datetime import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
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
regimes_peak_load = np.asarray(regime_peak_load_list)

###### Split the anomalous peak load by weather regime ######
akr_load = anom_peak_load[np.where(regimes_peak_load == 'AkR')[0]] # AkR.
arh_load = anom_peak_load[np.where(regimes_peak_load == 'ArH')[0]] # ArH.
pt_load = anom_peak_load[np.where(regimes_peak_load == 'PT')[0]] # PT.
wcr_load = anom_peak_load[np.where(regimes_peak_load == 'WCR')[0]] # WCR.
arl_load = anom_peak_load[np.where(regimes_peak_load == 'ArL')[0]] # ArL.

###### Now calculate risk ratios ######
# Set up the percentiles array to do risk ratios for.
perc_arr = np.array([90, 95, 97.5])
# Set up empty arrays to store each regimes risk ratio in.
akr_rr = np.zeros((perc_arr.size))
arh_rr = np.zeros((perc_arr.size))
wcr_rr = np.zeros((perc_arr.size))
pt_rr = np.zeros((perc_arr.size))
arl_rr = np.zeros((perc_arr.size))

# Loop through each percentile and calculate the risk ratio.
for i in range(len(perc_arr)):
    threshold = np.nanpercentile(anom_peak_load, q = perc_arr[i]) # First, get threshold of the percentile from all days.
    p_all = (np.where(anom_peak_load >= threshold)[0]).size/anom_peak_load.size # Then find the probability of a day being >= that threshold in the whole distribution.
    # Now calculate the probabilities of exceeding that same threshold in each separate regime distribution.
    p_akr = (np.where(akr_load >= threshold)[0]).size/akr_load.size # AkR Prob.
    p_arh = (np.where(arh_load >= threshold)[0]).size/arh_load.size # ArH Prob.
    p_wcr = (np.where(wcr_load >= threshold)[0]).size/wcr_load.size # WCR Prob.
    p_arl = (np.where(arl_load >= threshold)[0]).size/arl_load.size # ArL Prob.
    p_pt = (np.where(pt_load >= threshold)[0]).size/pt_load.size # PT Prob.
    # Now calculate each regime's risk ratio.
    akr_rr[i] = p_akr/p_all # AkR RR.
    arh_rr[i] = p_arh/p_all # ArH RR.
    wcr_rr[i] = p_wcr/p_all # WCR RR.
    arl_rr[i] = p_arl/p_all # ArL RR.
    pt_rr[i] = p_pt/p_all # PT RR.

###### Bootstrapping for the risk ratios ######
# Number of samples.
n_samples=5000

# Set up arrays to store the bootstrapped data of shape (n_samples, percentile).
rr_boot_akr = np.zeros((n_samples, perc_arr.size))
rr_boot_arh = np.zeros((n_samples, perc_arr.size))
rr_boot_wcr = np.zeros((n_samples, perc_arr.size))
rr_boot_arl = np.zeros((n_samples, perc_arr.size))
rr_boot_pt = np.zeros((n_samples, perc_arr.size))

# Now loop through number of samples.
for i in tqdm(range(n_samples)): # For each sample:
    rand_load_akr = np.random.choice(akr_load, size=akr_load.size, replace=True) # Pull random AkR loads from the AkR distribution with replacement.
    rand_load_arh = np.random.choice(arh_load, size=arh_load.size, replace=True) # Pull random ArH loads from the ArH distribution with replacement.
    rand_load_wcr = np.random.choice(wcr_load, size=wcr_load.size, replace=True) # Pull random WCR loads from the WCR distribution with replacement.
    rand_load_arl = np.random.choice(arl_load, size=arl_load.size, replace=True) # Pull random ArL loads from the ArL distribution with replacement.
    rand_load_pt = np.random.choice(pt_load, size=pt_load.size, replace=True) # Pull random PT loads from the PT distribution with replacement.

    # Now for these sets of dates in each sample, loop through the percentiles and find the risk ratio.
    for j in range(len(perc_arr)):
        # Threshold and probability for original distribution.
        threshold_boot = np.nanpercentile(anom_peak_load, q = perc_arr[j])
        p_all_boot = (np.where(anom_peak_load >= threshold_boot)[0]).size/anom_peak_load.size
        # Probability and risk ratio for AkR.
        p_boot_akr = (np.where(rand_load_akr >= threshold_boot)[0]).size/rand_load_akr.size
        rr_boot_akr[i, j] = p_boot_akr/p_all_boot
        # Probability and risk ratio for ArH.
        p_boot_arh = (np.where(rand_load_arh >= threshold_boot)[0]).size/rand_load_arh.size
        rr_boot_arh[i, j] = p_boot_arh/p_all_boot
        # Probability and risk ratio for WCR.
        p_boot_wcr = (np.where(rand_load_wcr >= threshold_boot)[0]).size/rand_load_wcr.size
        rr_boot_wcr[i, j] = p_boot_wcr/p_all_boot
        # Probability and risk ratio for ArL.
        p_boot_arl = (np.where(rand_load_arl >= threshold_boot)[0]).size/rand_load_arl.size
        rr_boot_arl[i, j] = p_boot_arl/p_all_boot
        # Probability and risk ratio for PT.
        p_boot_pt = (np.where(rand_load_pt >= threshold_boot)[0]).size/rand_load_pt.size
        rr_boot_pt[i, j] = p_boot_pt/p_all_boot

# Now get the signficance percentiles of the random distribution for confidence intervals.
# Set up percentiles to use, in this case 95th percentile confidence intervals.
perc1, perc2 = 2.5, 97.5
# Empty arrays of shape (conf bounds, percentile) for each regime.
akr_bars = np.zeros((2, perc_arr.size))
arh_bars = np.zeros((2, perc_arr.size))
wcr_bars = np.zeros((2, perc_arr.size))
arl_bars = np.zeros((2, perc_arr.size))
pt_bars = np.zeros((2, perc_arr.size))

# Now loop through each percentile and calculate the upper and lower bounds of confidence.
for i in range(perc_arr.size):
    akr_bars[0, i], akr_bars[1, i] = np.nanpercentile(rr_boot_akr[:,i], q = perc1) - akr_rr[i], np.nanpercentile(rr_boot_akr[:,i], q = perc2) - akr_rr[i]
    arh_bars[0, i], arh_bars[1, i] = np.nanpercentile(rr_boot_arh[:,i], q = perc1) - arh_rr[i], np.nanpercentile(rr_boot_arh[:,i], q = perc2) - arh_rr[i]
    wcr_bars[0, i], wcr_bars[1, i] = np.nanpercentile(rr_boot_wcr[:,i], q = perc1) - wcr_rr[i], np.nanpercentile(rr_boot_wcr[:,i], q = perc2) - wcr_rr[i]
    arl_bars[0, i], arl_bars[1, i] = np.nanpercentile(rr_boot_arl[:,i], q = perc1) - arl_rr[i], np.nanpercentile(rr_boot_arl[:,i], q = perc2) - arl_rr[i]
    pt_bars[0, i], pt_bars[1, i] = np.nanpercentile(rr_boot_pt[:,i], q = perc1) - pt_rr[i], np.nanpercentile(rr_boot_pt[:,i], q = perc2) - pt_rr[i]

###### Get the composite mean anomalies for each regime ######
mean_akr = np.nanmean(akr_load)
mean_arh = np.nanmean(arh_load)
mean_wcr = np.nanmean(wcr_load)
mean_arl = np.nanmean(arl_load)
mean_pt = np.nanmean(pt_load)

###### Bootstrap for the mean values ######
# Set up empty arrays to store bootstrapped data.
akr_mean_boot = np.zeros((n_samples)) # AkR.
arh_mean_boot = np.zeros((n_samples)) # ArH.
wcr_mean_boot = np.zeros((n_samples)) # WCR.
pt_mean_boot = np.zeros((n_samples)) # PT.
arl_mean_boot = np.zeros((n_samples)) # ArL.

# Now loop through number of samples, get random peak load values and then composite and store.
for i in tqdm(range(n_samples)):
    # Get the random loads for each regime type.
    rand_load_akr = np.random.choice(anom_peak_load, size = akr_load.size, replace = True)
    rand_load_arh = np.random.choice(anom_peak_load, size = arh_load.size, replace = True)
    rand_load_wcr = np.random.choice(anom_peak_load, size = wcr_load.size, replace = True)
    rand_load_pt = np.random.choice(anom_peak_load, size = pt_load.size, replace = True)
    rand_load_arl = np.random.choice(anom_peak_load, size = arl_load.size, replace = True)
    # Now do the composite of the random loads for that iteration.
    akr_mean_boot[i] = np.nanmean(rand_load_akr)
    arh_mean_boot[i] = np.nanmean(rand_load_arh)
    wcr_mean_boot[i] = np.nanmean(rand_load_wcr)
    pt_mean_boot[i] = np.nanmean(rand_load_pt)
    arl_mean_boot[i] = np.nanmean(rand_load_arl)

# Now get the percentile of scores for each regime.
akr_percentile = percentileofscore(akr_mean_boot, mean_akr)
arh_percentile = percentileofscore(arh_mean_boot, mean_arh)
wcr_percentile = percentileofscore(wcr_mean_boot, mean_wcr)
pt_percentile = percentileofscore(pt_mean_boot, mean_pt)
arl_percentile = percentileofscore(arl_mean_boot, mean_arl)

# Now put percentile and means in a list.
lst_perc = [akr_percentile, arh_percentile, wcr_percentile, pt_percentile, arl_percentile]
lst_mean = [mean_akr, mean_arh, mean_wcr, mean_pt, mean_arl]
# Loop through the percentiles and append the mean value if sig, otherwise nan.
plot_sig = np.zeros((len(lst_perc)))
for i in range(len(lst_perc)):
    if (lst_perc[i] >= perc2)|(lst_perc[i] <= perc1): # If significant, store mean.
        plot_sig[i] = lst_mean[i]
    else:
        plot_sig[i] = np.nan # Non-sig, store nan.

###### Now the PDFs for the regime load ######

# Set out points to fit the PDF.
points = np.arange(-3, 3.01, 0.01)

# AkR first.
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.arange(0.1, 30.1, 0.1)},
                    cv=5) # 5-fold cross-validation using KDE.
grid.fit(akr_load[:, None]) # Use the KDE on the AkR data.
print(grid.best_params_) # Print bandwidth.
akr_kde = grid.best_estimator_ # Get the kde estimator for AkR.
# AkR PDF is calculated.
akr_pdf = np.exp(akr_kde.score_samples(points[:, None]))

# ArH next.
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.arange(0.1, 30.1, 0.1)},
                    cv=5) # 5-fold cross-validation using KDE.
grid.fit(arh_load[:, None]) # Use the KDE on the ArH data.
print(grid.best_params_) # Print bandwidth.
arh_kde = grid.best_estimator_ # Get the kde estimator for ArH.
# ArH PDF is calculated.
arh_pdf = np.exp(arh_kde.score_samples(points[:, None]))

# PT next.
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.arange(0.1, 30.1, 0.1)},
                    cv=5) # 5-fold cross-validation using KDE.
grid.fit(pt_load[:, None]) # Use the KDE on the PT data.
print(grid.best_params_) # Print bandwidth.
pt_kde = grid.best_estimator_ # Get the kde estimator for PT.
# PT PDF is calculated.
pt_pdf = np.exp(pt_kde.score_samples(points[:, None]))

# WCR next.
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.arange(0.1, 30.1, 0.1)},
                    cv=5) # 5-fold cross-validation using KDE.
grid.fit(wcr_load[:, None]) # Use the KDE on the WCR data.
print(grid.best_params_) # Print bandwidth.
wcr_kde = grid.best_estimator_ # Get the kde estimator for WCR.

wcr_pdf = np.exp(wcr_kde.score_samples(points[:, None]))

# ArL next.
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.arange(0.1, 30.1, 0.1)},
                    cv=5) # 5-fold cross-validation using KDE.
grid.fit(arl_load[:, None]) # Use the KDE on the ArL data.
print(grid.best_params_) # Print bandwidth.
arl_kde = grid.best_estimator_ # Get the kde estimator for ArL.

arl_pdf = np.exp(arl_kde.score_samples(points[:, None]))

# All next.
grid = GridSearchCV(KernelDensity(),
                    {'bandwidth': np.arange(0.1, 30.1, 0.1)},
                    cv=5) # 5-fold cross-validation using KDE.
grid.fit(anom_peak_load[:, None]) # Use the KDE on the total data.
print(grid.best_params_) # Print bandwidth.
all_kde = grid.best_estimator_ # Get the kde estimator for ALL.

all_pdf = np.exp(all_kde.score_samples(points[:, None]))


###### PLOT THE FIGURES ######
fig = plt.figure(figsize=(12, 10)) # Set up figure.

# Plot the PDFs for each regime.
ax1 = plt.subplot2grid(shape = (4,4), loc = (0,0), colspan = 2, rowspan = 2) # Subplot.
ax1.plot(points, akr_pdf, color = 'darkorange', lw = 2, label = 'AkR') # Plot AkR.
ax1.plot(points, arh_pdf, color = 'darkred', lw = 2, label = 'ArH') # Plot ArH.
ax1.plot(points, pt_pdf, color = 'darkgreen', lw = 2, label = 'PT') # Plot PT.
ax1.plot(points, wcr_pdf, color = 'darkblue', lw = 2, label = 'WCR') # Plot WCR.
ax1.plot(points, arl_pdf, color = 'purple', lw = 2, label = 'ArL') # Plot ArL.
ax1.plot(points, all_pdf, color = 'black', lw = 2, label = 'All') # Plot ALL.
plt.axvline(x = 0, color = 'black', lw = 2, ls = '--') # Vertical dashed line for 0 line.
ax1.set_xticks(np.arange(-5, 6, 1)) # Set xticks.
ax1.set_yticks(np.arange(0, 1.2, 0.2)) # Set yticks.
ax1.set_xlim([-2, 2]) # Set x-lim.
ax1.set_ylim([0, 1]) # Set y-lim.
ax1.set_xlabel('Peak Load Anomaly (MWh/1000 Cust)', fontsize = 13) # Set x label.
ax1.set_ylabel('Probability Density', fontsize = 13) # Set y label.
ax1.set_title('a) Probability Density Function', weight = 'bold', fontsize = 14) # Set title.
ax1.legend() # Set legend.

# Plot the mean loads by regime.
ax2 = plt.subplot2grid(shape = (4,4), loc = (0,2), colspan = 2, rowspan = 2) # Subplot.
reg_labs = ['AkR', 'ArH', 'WCR', 'PT', 'ArL'] # Labels for the regimes.
mean_anoms = [mean_akr, mean_arh, mean_wcr, mean_pt, mean_arl] # Mean anomalies for the bar plot for each regime.
ax2.bar(reg_labs, mean_anoms, color = ['darkorange', 'darkred', 'darkblue', 'darkgreen', 'purple'], edgecolor = 'black', label = reg_labs) # Barplot.
ax2.plot(reg_labs, plot_sig, marker="D", linestyle="", alpha=1, color="white", markeredgecolor = 'black') # Plot the dots for significance.
ax2.set_xlabel("Regime", fontsize = 13) # Set x label.
ax2.set_ylabel('Peak Load Anomaly (MWh/1000 Cust)', fontsize = 13) # Set y label.
ax2.set_title("b) Composite Peak Load Anomaly", weight = 'bold', fontsize = 14) # Set the title.
ax2.set_yticks(np.arange(-0.2, 0.25, 0.05)) # Set y ticks.
ax2.set_ylim([-0.2, 0.2]) # Set y lim.
ax2.set_xticklabels(reg_labs, rotation=45, fontsize = 10) # Plot the xtick labels.
plt.axhline(y=0, lw = 1, ls = '-', color = 'black') # Horizontal line for 0 line.
ax2.legend() # Set legend.

# Plot the risk ratios.
ax3 = plt.subplot2grid(shape = (4,4), loc = (2,1), colspan = 2, rowspan = 2) # Subplot.
X = np.arange(1, 4, 1) # Get x tick numbers.
labels = [f'{int(perc_arr[0])}th',f'{int(perc_arr[1])}th', f'{perc_arr[2]}th'] # Set out labels of percentiles.
bottom = 1 # Set bottom value for risk ratio plot.
reg_bar = funcs.subcategorybar(X, [akr_rr,arh_rr,wcr_rr,pt_rr,arl_rr], [np.absolute(akr_bars), np.absolute(arh_bars), np.absolute(wcr_bars), np.absolute(pt_bars), np.absolute(arl_bars)], error_col = 'gray',  bt = bottom, colors = ['darkorange', 'darkred', 'darkblue', 'darkgreen', 'purple'], capsize = 10) # Plot the bars.
ax3.set_xticklabels(labels, rotation=45, fontsize = 10) # Plot the xtick labels.
ax3.set_yticks(np.arange(0, 3, 0.5)) # Plot the y ticks.
ax3.set_ylim([0, 2.5]) # Set the y limit.
ax3.set_ylabel("Risk Ratio", fontsize = 13) # Plot the y label.
ax3.set_xlabel("Percentile", fontsize = 13) # Plot the x label.
ax3.set_title("c) Extreme Peak Load Risk Ratios", weight = 'bold', fontsize = 14) # Set the title.
ax3.legend(reg_labs) # Put a legend.
plt.axhline(y=1, lw = 1, ls = '-', color = 'black') # Horizontal line for the 1 risk ratio value.
fig.tight_layout() # Tight layout.
plt.savefig("/share/data1/Students/ollie/CAOs/project-cold-load/Figures/Regimes/regimes_load.png", bbox_inches = 'tight', dpi = 500) # Save figure.