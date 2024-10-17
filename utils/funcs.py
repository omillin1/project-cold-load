###### Import modules ######
import numpy as np
from netCDF4 import Dataset, num2date
import os
from datetime import datetime
from tqdm import tqdm
from matplotlib import cm, colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cftime
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

# Define a normal colormap function.
def NormColorMap(cmap_select, my_bounds):
    """Creates a normalized color map.

    Parameters
    ---------
    cmap_select: string.
        The Python cmap to normalize.
    my_bounds: numpy.ndarray, shape(my_bounds,).
        1-D NumPy array containing the array of conotur levels in the colormap. Has to be diverging and centered on zero.

    Returns
    ---------
    my_cmap: matplotlib.colors.ListedColorMap object.
        The normalized colormap object used for plotting the cmap in contourf.
    norm: matplotlib.colors.BoundaryNorm object.
        The normalization object for plotting the cmap in contourf.
    """
    # Get selected cmap.
    cmap = cm.get_cmap(cmap_select)
    # Get the selected bounds.
    bounds = my_bounds
    # Number of colors based on bounds size.
    numColors = bounds.size - 1
    # Set list to append colors.
    my_colors = []
    # Loop through colors and append colors for custom cmap.
    for i in range(1,numColors+1):
        if (i != (numColors / 2.)): # If i is not half the size of bounds, append cmap colors.
            my_colors.append(cmap(i/float(numColors)))
        else: # Otherwise, white for around the "zero mark" of the bounds specified.
            my_colors.extend(['white']*2)
    # Make the cmap and norm from custom features.
    my_cmap = colors.ListedColormap(my_colors)
    my_cmap.set_under(cmap(0.0))
    my_cmap.set_over(cmap(1.0))
    norm = colors.BoundaryNorm(bounds, my_cmap.N)

    return my_cmap, norm

def format_daily_ERA5(var = 'hgt', level = 500, lat_bounds = [90, 0], lon_bounds = [0, 359.5], year_bounds = [1996, 2014], months = ['01','02','03','11','12'], ndays = 151, isentropic = False):
    '''
    Function to format ERA5 daily data into (year, day in season, lat, lon) at a specific pressure level.
    This just makes it easier to subtract climatology.

    Parameters
    -------
    var: string. 
        The variable to use.
    level: int. 
        The pressure level to use.
    lat_bounds: list, length 2. 
        Floats/ints with the two latitude bounds in degrees north, highest latitude first and lowest latitude second.
    lon_bounds: list, length 2. 
        Floats/ints with the longitude bounds in degrees east, furthest west longitude first and furthest east longitude second.
    year_bounds: list, length 2. 
        Ints representing the year bounds to use, earliest year first and latest year second.
    months: list, maximum length of 12.
        Strings of the month labels to use, e.g., '01' for January.
    ndays: int. 
        Describes how many days there are total for your months cumulatively excluding leap days (i.e., 90 for DJF).
    isentropic: boolean.
        Describes whether it is an isentropic level or not, True for isentropic and False for isobaric.

    Outputs
    -------
    era_var: numpy.ndarray, shape (years, days of season, lat, lon).
        4-D NumPy array of ERA5 daily data for the variable of choice.
    era_time: numpy.ndarray, shape (years, days of season).
        4-D NumPy array of datetimes for the ERA5 data.
    latitude: numpy.ndarray, shape (lat,).
        1-D NumPy array of floats representing latitude.
    longitude: numpy.ndarray, shape (lon,).
        1-D NumPy array of floats representing longitude.
    '''
    if level == None: # If it is a 2D variable.
        # Go to the data directory.
        dir = f'/data/deluge/reanalysis/REANALYSIS/ERA5/2D/4xdaily/{var}'
        path = os.chdir(dir)
        # Read in one file's lat and lon to get the dimensions:
        track_file = f'{var}.200001.nc' # Test filename.
        nc_track = Dataset(track_file, 'r') # Open file.
        lat_track = nc_track.variables['latitude'][:] # Latitude array.
        lon_track = nc_track.variables['longitude'][:] # Longitude array.
        nc_track.close() # Close the file.

        # Get lat bound indices.
        lat_index_1 = np.where(lat_track == lat_bounds[0])[0][0]
        lat_index_2 = np.where(lat_track == lat_bounds[1])[0][0]
        # Get lon bound indices.
        lon_index_1 = np.where(lon_track == lon_bounds[0])[0][0]
        lon_index_2 = np.where(lon_track == lon_bounds[1])[0][0]
        # Restrict lat and lon to the bounds.
        latitude = lat_track[lat_index_1:lat_index_2+1]
        longitude = lon_track[lon_index_1:lon_index_2+1]

        # Set array of years.
        years_ra = np.arange(year_bounds[0], year_bounds[1]+1, 1)
        # Set arrays to store the era5 data and times.
        era_var = np.zeros((len(years_ra),ndays,len(latitude),len(longitude))) # Shape (years, days in season, lat, lon)
        era_time = np.zeros((len(years_ra),ndays),dtype=object) # Shape (years, days in season).
        # Now loop through the years and months and format the ERA5 data.
        for y, year in enumerate(tqdm(years_ra)):
            season_list = [] # List to append season data to.
            dates_list = [] # List to append season datetimes to.
            for m, month in enumerate(months): # For each year go through each month.
                fname = f"{var}.{year}{month}.nc" # Set filename.
                print(fname, year, month)
                era = Dataset(fname,'r') # Open file.
                data = era.variables[var][:,lat_index_1:lat_index_2+1,lon_index_1:lon_index_2+1] # Read in the spatially restricted data.
                time = num2date(era.variables['time'][:],era.variables['time'].units,era.variables['time'].calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True) # Read in datetimes.
                era.close() # Close the file.
                reshape = data.reshape(int(len(time)/4), 4, data.shape[1], data.shape[2]) # Reshape to (days, hourly steps, lat, lon) for the month.
                daily_ave = np.nanmean(reshape, axis = 1) # Daily average.
                time_select = time[::4] # Get time stamp for every day but only once at midnight.
                if daily_ave.shape[0] == 29: # If the length of days is 29, remove leap day data.
                    daily_ave = daily_ave[:-1] # Exclude Feb 29th from data.
                    time_select = time_select[:-1] # Exclude Feb 29th from datetimes.
                else: # Else keep the variables the same.
                    daily_ave = daily_ave
                    time_select = time_select
                season_list.append(daily_ave) # Append monthly daily averaged data to the data list.
                # Append the times to the time list.
                for i in range(len(time_select)): # For each datetime, append to the list separately.
                    dates_list.append(time_select[i])

            era_var[y] = np.vstack(season_list) # Now for each year, vertically stack (i.e., concatenate along first axis) the ERA5 data and store it in the array.
            era_time[y] = np.asarray(dates_list) # Then store an array of datetimes in the time zeros array.

    else: # If the variable is a 3D variable with a level.
        # Go to the data.
        if isentropic == True: # Isentropic data in a different place.
            dir = f'/data/deluge/reanalysis/REANALYSIS/ERA5/isentropic/4xdaily/{var}'
            path = os.chdir(dir)
        else: # Otherwise go to 3D data.
            dir = f'/data/deluge/reanalysis/REANALYSIS/ERA5/3D/4xdaily/{var}'
            path = os.chdir(dir)
        # Read in one file's lat, lon, and level to get the dimensions:
        track_file = f'{var}.200001.nc' # Test filename.
        nc_track = Dataset(track_file, 'r') # Open file.
        lat_track = nc_track.variables['latitude'][:] # Latitude array.
        lon_track = nc_track.variables['longitude'][:] # Longitude array.
        level_track = nc_track.variables['level'][:] # Level array.
        nc_track.close() # Close the file.

        # Get lat bound indices.
        lat_index_1 = np.where(lat_track == lat_bounds[0])[0][0]
        lat_index_2 = np.where(lat_track == lat_bounds[1])[0][0]
        # Get lon bound indices.
        lon_index_1 = np.where(lon_track == lon_bounds[0])[0][0]
        lon_index_2 = np.where(lon_track == lon_bounds[1])[0][0]
         # Get level index.
        level_index = np.where(level_track == level)[0][0]

        # Restrict lat and lon to the bounds.
        latitude = lat_track[lat_index_1:lat_index_2+1]
        longitude = lon_track[lon_index_1:lon_index_2+1]

        # Set array of years.
        years_ra = np.arange(year_bounds[0], year_bounds[1]+1, 1)
        # Set arrays to store the era5 data and times.
        era_var = np.zeros((len(years_ra),ndays,len(latitude),len(longitude))) # Shape (years, days in season, lat, lon)
        era_time = np.zeros((len(years_ra),ndays),dtype=object) # Shape (years, days in season).
        # Now loop through the years and months and format the ERA5 data.
        for y, year in enumerate(tqdm(years_ra)):
            season_list = [] # List to append seasons data to.
            dates_list = [] # List to append seasons datetimes to.
            for m, month in enumerate(months): # For each year go through each month.
                fname = f"{var}.{year}{month}.nc" # Set filename.
                print(fname, year, month)
                era = Dataset(fname,'r') # Open file.
                data = era.variables[var][:,level_index,lat_index_1:lat_index_2+1,lon_index_1:lon_index_2+1] # Read in the spatially restricted data.
                time = num2date(era.variables['time'][:],era.variables['time'].units,era.variables['time'].calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True) # Read in datetimes.
                era.close() # Close the file.
                reshape = data.reshape(int(len(time)/4), 4, data.shape[1], data.shape[2]) # Reshape to (days, hourly steps, lat, lon) for the month.
                daily_ave = np.nanmean(reshape, axis = 1) # Daily average.
                time_select = time[::4] # Get time stamp for every day but only once at midnight.
                if daily_ave.shape[0] == 29: # If the length of days is 29, remove leap day data.
                    daily_ave = daily_ave[:-1] # Exclude Feb 29th from data.
                    time_select = time_select[:-1] # Exclude Feb 29th from datetimes.
                else: # Else keep the variables the same.
                    daily_ave = daily_ave
                    time_select = time_select
                season_list.append(daily_ave) # Append monthly daily averaged data to the data list.
                # Append the times to the time list.
                for i in range(len(time_select)): # For each datetime, append to the list separately.
                    dates_list.append(time_select[i])

            era_var[y] = np.vstack(season_list) # Now for each year, vertically stack (i.e., concatenate along first axis) the ERA5 data and store it in the array.
            era_time[y] = np.asarray(dates_list) # Then store an array of datetimes in the time zeros array.


    return era_var, era_time, latitude, longitude

def minmax_2d_ERA5(var = 'Tmin', lat_bounds = [90, 0], lon_bounds = [0, 359.5], year_bounds = [1996, 2014], months = ['01','02','03','11','12'], ndays = 151):
    '''
    Function to format ERA5 daily min or max data into (year, day in season, lat, lon).
    This just makes it easier to subtract climatology.

    Parameters
    -------
    var: string. 
        The variable to use.
    lat_bounds: list, length 2. 
        Floats/ints with the two latitude bounds in degrees north, highest latitude first and lowest latitude second.
    lon_bounds: list, length 2. 
        Floats/ints with the longitude bounds in degrees east, furthest west longitude first and furthest east longitude second.
    year_bounds: list, length 2. 
        Ints representing the year bounds to use, earliest year first and latest year second.
    months: list, maximum length of 12.
        Strings of the month labels to use, e.g., '01' for January.
    ndays: int. 
        Describes how many days there are total for your months cumulatively excluding leap days (i.e., 90 for DJF).

    Outputs
    -------
    era_var: numpy.ndarray, shape (years, days of season, lat, lon).
        4-D NumPy array of ERA5 daily data for the variable of choice.
    era_time: numpy.ndarray, shape (years, days of season).
        4-D NumPy array of datetimes for the ERA5 data.
    latitude: numpy.ndarray, shape (lat,).
        1-D NumPy array of floats representing latitude.
    longitude: numpy.ndarray, shape (lon,).
        1-D NumPy array of floats representing longitude.
    '''
    
    # Go to the data.
    dir = f'/data/deluge/reanalysis/REANALYSIS/ERA5/2D/daily/{var}'
    path = os.chdir(dir)
    # Read in one file's lat and lon to get the dimensions:
    track_file = f'{var}.200001.nc' # Test filename.
    nc_track = Dataset(track_file, 'r') # Open file.
    lat_track = nc_track.variables['latitude'][:] # Latitude array.
    lon_track = nc_track.variables['longitude'][:] # Longitude array.
    nc_track.close() # Close the file.

    # Get lat bound indices.
    lat_index_1 = np.where(lat_track == lat_bounds[0])[0][0]
    lat_index_2 = np.where(lat_track == lat_bounds[1])[0][0]
    # Get lon bound indices.
    lon_index_1 = np.where(lon_track == lon_bounds[0])[0][0]
    lon_index_2 = np.where(lon_track == lon_bounds[1])[0][0]
    # Restrict lat and lon to the bounds.
    latitude = lat_track[lat_index_1:lat_index_2+1]
    longitude = lon_track[lon_index_1:lon_index_2+1]

    # Set array of years.
    years_ra = np.arange(year_bounds[0], year_bounds[1]+1, 1)
    # Set arrays to store the era5 data and times.
    era_var = np.zeros((len(years_ra),ndays,len(latitude),len(longitude))) # Shape (years, days in season, lat, lon)
    era_time = np.zeros((len(years_ra),ndays),dtype=object) # Shape (years, days in season).
    # Now loop through the years and months and format the ERA5 data.
    for y, year in enumerate(tqdm(years_ra)):
        season_list = [] # List to append seasons data to.
        dates_list = [] # List to append seasons datetimes to.
        for m, month in enumerate(months):  # For each year go through each month.
            if year >= 2022: # Local 2022+ data is in a different directory.
                fname = f"/data/deluge/scratch/tmp/Ollie_ERA5/Tmin/{var}.{year}{month}.nc" # Set filename.
            else: # Otherwise it is in the original directory.
                fname = f"{var}.{year}{month}.nc" # Set filename.
            print(fname, year, month)
            era = Dataset(fname,'r') # Open file.
            if var == 'Tmin': # Specific indicator for Tmin.
                var_ind = 't2m'
            else:
                var_ind = var # Otherwise var=var.
            data = era.variables[var_ind][:,lat_index_1:lat_index_2+1,lon_index_1:lon_index_2+1] # Read in the spatially restricted data.
            time = num2date(era.variables['time'][:],era.variables['time'].units,era.variables['time'].calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True) # Read in datetimes.
            era.close() # Close the file.
            if data.shape[0] == 29: # If the length of days is 29, remove leap day data.
                data = data[:-1] # Exclude Feb 29th from data.
                time = time[:-1] # Exclude Feb 29th from datetimes.
            else: # Else keep the variables the same.
                data = data
                time = time
            season_list.append(data) # Append monthly minmax data to the data list.
            # Append the times to the time list.
            for i in range(len(time)): # For each datetime, append to the list separately.
                dates_list.append(time[i])

        era_var[y] = np.vstack(season_list) # Now for each year, vertically stack (i.e., concatenate along first axis) the ERA5 data and store it in the array.
        era_time[y] = np.asarray(dates_list) # Then store an array of datetimes in the time zeros array.

    return era_var, era_time, latitude, longitude

# Define a draw polygon function
def DrawPolygon(ax, lat_bounds = [-8, -40], lon_bounds = [295, 325], res = 0.5, color = 'black', lw = 2):
    '''
    Function to plot a box polygon on a cartopy map.

    Parameters
    -------
    ax: matplotlib.pyplot.figure.add_subplot object.
        ax = fig.add_subplot() object for subplot.
    lat_bounds: list, length 2. 
        Floats/ints containing the box latitude bounds in degrees north, highest latitude bound first and lowest latitude bound second.
    lon_bounds: list, length 2. 
        Floats/ints containing the box longitude bounds in degrees east, furthest west bound first and furthest east bound second.
    res: float or int. 
        Grid resolution of the plotting data.
    color: string. 
        Python color of the box outline.
    lw: float or int. 
        Width of box outline.

    Outputs
    -------
    plot1: numpy.ndarray.
        1-D NumPy array of spatial points for the box side 1.
    plot2: numpy.ndarray.
        1-D NumPy array of spatial points for the box side 2.
    plot3: numpy.ndarray.
        1-D NumPy array of spatial points for the box side 3.
    plot4: numpy.ndarray.
        1-D NumPy array of spatial points for the box side 4.
    '''
    # Define lats for box.
    lat_restr = np.arange(lat_bounds[0], lat_bounds[1]-res, -res)
    # Define lons for box.
    lon_restr = np.arange(lon_bounds[0], lon_bounds[1]+res, res)

    # Now get the left side of the box.
    lon_rep_left = np.repeat(lon_bounds[0], lat_restr.shape[0]) # Repeated lons for left side.

    # Now get the right side of the box.
    lon_rep_right = np.repeat(lon_bounds[1], lat_restr.shape[0]) # Repeated lons for right side.
    
    # Now get the upper bound of the box.
    lat_rep_top = np.repeat(lat_bounds[0], lon_restr.shape[0]) # Repeated lats for top side.

    # Now get the lower bound of the box.
    lat_rep_bottom = np.repeat(lat_bounds[1], lon_restr.shape[0]) # Repeated lats for top side.

    # Plot the box.
    # Left side.
    plot1 = ax.plot(lon_rep_left, lat_restr, color=color, lw = lw, transform=ccrs.PlateCarree())
    # Right side.
    plot2 = ax.plot(lon_rep_right, lat_restr, color=color, lw = lw, transform=ccrs.PlateCarree())
    # Top side.
    plot3 = ax.plot(lon_restr, lat_rep_top, color=color, lw = lw, transform=ccrs.PlateCarree())
    # Bottom side.
    plot4 = ax.plot(lon_restr, lat_rep_bottom, color=color, lw = lw, transform=ccrs.PlateCarree())
    
    return plot1, plot2, plot3, plot4


# Define a Plate Carree Map.
def PlateCarreeMap(fig, subplot = [1, 1, 1], lon_bounds = [260, 330], lat_bounds = [20, -60], cen_lon = 295):
    '''
    Function to plot a cartopy PlateCarree map.

    Parameters
    -------
    fig: matplotlib.pyplo.figure object.
        The figure which to plot the map subplot on.
    subplot: list, length 3. 
        Integers representing the rows, columns, and position of the figure subplot.
    lon_bounds: list, length 2.
        Floats/ints representing the longitude bounds for the plot in degrees east, with the furthest west longitude in the first position and furthest east in the second position.
    lat_bounds: list, length 2. 
        Floats/ints representing the latitude bounds for the plot in degrees north, with the furthest north latitude in the first position and the furthest south latitude in the second position.
    cen_lon: float or integer. 
        The central longitude coordinate of the map projection.

    Outputs
    -------
    ax1: matplotlib.pyplot.figure.add_subplot feature.
        The object containing the map to subplot.
    '''
    # Add subplot.
    ax1 = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection=ccrs.PlateCarree(central_longitude = cen_lon)) # Generate the figure at the given subplot position.
    # Set extent of map.
    ax1.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree()) # Set the extent of the map.
    # Add coastlines and borders.
    ax1.coastlines() # Draw coastlines.
    ax1.add_feature(cfeature.BORDERS) # Add borders.
    return ax1

# Define the linear detrend function.
def LinearDetrend(y):
    """Linearly detrends an array of data.

    Parameters
    ---------
    y: numpy.ndarray, shape.
        1-D NumPy array with the data to be detrended.

    Returns
    ---------
    detrendedData: numpy.nadarray.
        1-D NumPy array containing the detrended data.
    trend: float. 
        The slope of the trend line.
    """
    time = np.arange(y.shape[0]) # Get an array of points of shape time.

    E = np.ones((y.shape[0],2)) # Create E matrix.
    E[:,0] = time # Fill the first column with the time indices.

    invETE = np.linalg.inv(np.dot(E.T,E)) # Do the matrix multiplication E.T by E.
    xhat = invETE.dot(E.T).dot(y) # Get the xhat.
    trend = np.dot(E,xhat) # Matrix multiplication for the trend (E x xhat).
    detrendedData = y - trend # Remove trend for the detrended data.

    return detrendedData,trend

def is_months(month, month_bnd = [3, 11]):
    """Returns boolean indices of months selected
        
    Parameters
    ---------
    month: numpy.ndarray.
        1-D NumPy array, representing the months in a time array.
    month_bnd: list, length 2. 
        Ints representing the early and late month bounds.
        
    Returns
    ---------
    (month <= month_bnd[0]) | (month >= month_bnd[1]): numpy.ndarray.
        1-D NumPy array, representing the booleans for whether the month condition is met.
    """
    return (month <= month_bnd[0]) | (month >= month_bnd[1])


def read_early_load(div = 'OKGE', year1 = 1999, month_bnd = [3, 11]):
    """Retrieves hourly SPP load for a given division for the dataset from 1999-2010.
        
    Parameters
    ---------
    div: string. 
        The division name.
    year1: int. 
        The first year bound of the data to retrieve.
    month_bnd: list, length 2.
        Ints contains the month bounds for retrieval. Will be less than or equal to the first position, and greater than or equal to the last position.
        
    Returns
    ---------
    dates_final: numpy.ndarray.
        1D NumPy array of datetimes, the datetimestimes associated with the load values.
    load_final: numpy.ndarray.
        1D NumPy array, the load values for the given SPP division.
    """
    # Go to directory.
    dir = '/share/data1/Students/ollie/CAOs/Data/Energy/Load/SPP/Divisional/1999_2010'
    path = os.chdir(dir)

    # Open the data.
    filename = 'spp_load_1999_2010_ba.csv'
    energy_frame = pd.read_csv(filename, skiprows = [0], usecols=[0, 1, 2, 4, 5, 6], names = ['Region', 'Year', 'Month', 'Day', 'Hour', 'Load'])

    # Get arrays for datetimes.
    year_arr = energy_frame['Year'].values # Years.
    month_arr = energy_frame['Month'].values # Months.
    day_arr = energy_frame['Day'].values # Days.
    hour_arr = energy_frame['Hour'].values # Hours.

    # Now get load and region.
    region_arr = energy_frame['Region'].values # The regions.
    load_arr = energy_frame['Load'].values # The load.

    # Select the region.
    ind_region = np.where(region_arr == div)[0] # Get indices where the region is your chosen one.
    # Restrict the time data to your chosen division.
    year_region, month_region, day_region, hour_region = year_arr[ind_region], month_arr[ind_region], day_arr[ind_region], hour_arr[ind_region]
    load_region = load_arr[ind_region] # Restrict the load data to your specific region.
    # Get the indices for the selected months.
    ind_month = np.where((month_region <= month_bnd[0])|(month_region >= month_bnd[1]))[0]
    # Restrict time data to the chosen month range.
    year_period, month_period, day_period, hour_period = year_region[ind_month], month_region[ind_month], day_region[ind_month], hour_region[ind_month]
    load_period = load_region[ind_month] # Restrict load data to the chosen month range. This is now load for the selected period and months.

    # Now get datetimes for this period.
    dates_list = [] # Empty list for the dates.
    for i in range(len(load_period)): # Loop through each day and append the datetime to the empty list.
        dates_list.append(datetime(year_period[i], month_period[i], day_period[i], hour_period[i]))
    dates_arr = np.asarray(dates_list) # Make to an array.

    # Now restrict to the start year you want.
    year_track = np.array([d.year for d in dates_arr]) # Get year tracker.
    year_ind = np.where(year_track >= year1)[0] # Find the indices for where the dates are newer than the given year.

    # Now restrict the dates and load to the years you want.
    dates_restr, load_restr = dates_arr[year_ind], load_period[year_ind]

    # Turn into xarray dataset.
    ds = xr.Dataset(
        data_vars=dict(
            load=(["time"], load_restr),
        ),
        coords=dict(
            time=dates_restr
        ),
        attrs=dict(description=f"Load (MW) for {div}"),
    )

    # Sort by time.
    sorted_ds = ds.sortby('time')

    # Remove leap days.
    leap_remove = sorted_ds.convert_calendar('noleap')

    # Get the data and dates out of the dataset.
    load_final = leap_remove['load'].values
    dates_cftime = leap_remove['time'].values

    # Convert to python datetimes.
    dt_date_lst = [] # List to append dates.
    for i in range(len(dates_cftime)):
        dt_date_lst.append(datetime(dates_cftime[i].year, dates_cftime[i].month, dates_cftime[i].day, dates_cftime[i].hour, dates_cftime[i].minute)) # Append the dates.
    # Make dates into array.
    dates_final = np.array(dt_date_lst)

    return dates_final, load_final


def read_late_load(year1, year2, months=['01', '02', '03', '04', '10', '11', '12'], month_bnd = [3, 11], div = 'OKGE'):
    """Retrieves hourly SPP load for 2011 onwards.
        
    Parameters
    ---------
    year1: int. 
        The first year of the data to retrieve.
    year2: int. 
        The last year of the data to retrieve.
    months: list. 
        Strings containing each month indentifier to retrieve but with a buffer. If you want NDJFM, select ['01', '02', '03', '04', '10', '11', '12'] with month_bound = [3, 11].
    month_bnd: list, length 2.
        Ints contains the month bounds for retrieval. Will be less than or equal to the first position, and greater than or equal to the last position.
    div: string.
        The selected division name.
        
    Returns
    ---------
    final_dates: numpy.ndarray.
        1D NumPy array of datetimes, the times associated with the load values.
    final_load: numpy.ndarray.
        1D NumPy array, the load values for the given SPP division.
    """
    # Go to directory.
    dir = '/share/data1/Students/ollie/CAOs/Data/Energy/Load/SPP/Divisional'
    path = os.chdir(dir)
    # Select parameters.
    year_select = np.arange(year1, year2+1, 1) # Year range array.
    month_select = months # Month list of strings.

    # Set out lists to store data.
    date_list = [] # Empty list for date data.
    load_list = [] # Empty list for load data.

    # Loop through years and months.
    for i in tqdm(range(len(year_select))): # Loop through selected years.
        for j in range(len(month_select)): # Loop through selected months.
            # Get filename.
            filename =f"{year_select[i]}/HOURLY_LOAD-{year_select[i]}{month_select[j]}.csv"
            # Open the file.
            energy_frame = pd.read_csv(filename, skipinitialspace=True, skip_blank_lines=True)
            # Drop any lines that have NaN due to empty spaces.
            energy_frame.dropna(how="all", inplace=True)
            # Read in times.
            dates = energy_frame['MarketHour'].values
            # If KCPL then add INDN load due to merger in this period.
            if div == 'KCPL':
                load = energy_frame[div].values + energy_frame['INDN'].values # INDN + KCPL if division is KCPL.
            else: # Otherwise just read in load.
                load = energy_frame[div].values
            # If it is after March 2014, format is slightly different for date.
            if ((year_select[i] == 2014)&(int(month_select[j])>= 3)):
                # Get dates.
                date_lst = []
                for k in range(len(dates)): # Loop through dates.
                    if len(dates[k]) <= 10: # Get dates, check for length of date stamp because of reporting differences.
                        date_lst.append(datetime.strptime(dates[k]+' 0:00', "%m/%d/%Y %H:%M")) # If length of str date is less than 10, it is midnight so add 0:00.
                    else:
                        date_lst.append(datetime.strptime(dates[k], "%m/%d/%Y %H:%M")) # Else continues as normal for this condition, append the datetimes to the list.
            # If it is 2015 or 2016, slightly different format for date.
            elif (year_select[i] > 2014)&(year_select[i] < 2017):
                # Get dates.
                date_lst = []
                for k in range(len(dates)): # Loop through dates.
                    if len(dates[k]) <= 10: # Get dates, check for length of date stamp because of reporting differences.
                        date_lst.append(datetime.strptime(dates[k]+' 0:00', "%m/%d/%Y %H:%M")) # If length of str date is less than 10, it is midnight so add 0:00.
                    else:
                        date_lst.append(datetime.strptime(dates[k], "%m/%d/%Y %H:%M")) # Else continues as normal for this condition, append the datetimes to the list.
            # If year is greater than 2017, different format for date.
            elif (year_select[i] >= 2017):
                # Get dates.
                date_lst = []
                for k in range(len(dates)): # Loop through dates.
                    date_lst.append(datetime.strptime(dates[k], "%m/%d/%Y %H:%M:%S")) # Append the datetimes to the list.
            else:
                # Get dates.
                date_lst = []
                for k in range(len(dates)): # Loop through dates.
                    date_lst.append(datetime.strptime(dates[k], "%m/%d/%y %H:%M")) # Append the datetimes to the list.

            date_arr = np.array(date_lst) # Make dates into an array (will be all hours for the selected year and month).

            date_list.append(date_arr) # Append array of dates for selected month, year, and division to main date list.
            load_list.append(load) # Append array of loads for division to main list.
    
    # Now concatenate the separate arrays in the list to get continuous data.
    load_array = np.concatenate((load_list))
    date_array = np.concatenate((date_list))

    # Turn into xarray dataset.
    ds = xr.Dataset(
        data_vars=dict(
            load=(["time"], load_array),
        ),
        coords=dict(
            time=date_array
        ),
        attrs=dict(description=f"Load (MW) for {div}"),
    )
    # Now retrieve only data for months that you want to keep.
    grouped_ds = ds.sel(time=is_months(ds['time.month'], month_bnd=month_bnd))

    # Sort by time.
    sorted_ds = grouped_ds.sortby('time')

    # Remove leap days.
    leap_remove = sorted_ds.convert_calendar('noleap')

    # Get the data and dates out of the dataset.
    load_data = leap_remove['load'].values
    load_dates = leap_remove['time'].values

    # Find where it is not the first or last day of the dataset, accounts for incomplete data on these days.
    ind1 = np.where((load_dates >= cftime.DatetimeNoLeap(year1, 1, 2))&(load_dates < cftime.DatetimeNoLeap(year2+1, 1, 1)))[0]

    # Restrict the data.
    adj_load = load_data[ind1]
    adj_dates = load_dates[ind1]

    # Convert to python datetimes.
    dt_date_lst = [] # List to append dates.
    for i in range(len(adj_dates)):
        dt_date_lst.append(datetime(adj_dates[i].year, adj_dates[i].month, adj_dates[i].day, adj_dates[i].hour, adj_dates[i].minute)) # Append the dates.
    # Make dates into array.
    dt_dates = np.array(dt_date_lst)

    # Now get the days we know that are incomplete.
    incomplete_dt = np.array([datetime(2018, 3, 6), datetime(2018, 12, 12), datetime(2021, 11, 10),\
                               datetime(2022, 11, 3), datetime(2022, 12, 12), datetime(2022, 12, 20), datetime(2023, 3, 12)])

    years_arr = np.array([d.year for d in dt_dates]) # Get the year array of dt_dates.
    months_arr = np.array([d.month for d in dt_dates]) # Get the month array of dt_dates.
    days_arr = np.array([d.day for d in dt_dates]) # Get the day array of dt_dates.

    # Now let's loop through and get indices to remove.
    ind_list = [] # Empty list to append to.
    for i in range(len(incomplete_dt)):
        ind = np.where((years_arr == incomplete_dt[i].year)&(months_arr == incomplete_dt[i].month)&(days_arr == incomplete_dt[i].day))[0] # Finds all indices for given incomplete days.
        ind_list.append(ind) # Appends missing data day indices.
    # Concatenate the index lists.
    conc_inds = np.concatenate((ind_list))
    # Delete the indices from the load and date arrays.
    final_load, final_dates = np.delete(adj_load, conc_inds), np.delete(dt_dates, conc_inds)  

    return final_dates, final_load


def subcategorybar(X, vals, errors, error_col, width=0.8, bt = 1, capsize = 10, colors = ['darkblue', 'gray', 'darkred']):
    """Plots a subcategory bar.
        
    Parameters
    ---------
    X: numpy.ndarray.
        1-D NumPy array of floats/ints, shape should resemble that of the number of x-axis ticks.
    vals: list. 
        List of 1-D numpy.ndarrays, the size of the list should resemble the number of bars and the shape of the arrays should resemble the shape of X.
    errors: list.
        List of 1-D numpy.ndarrays, the size of the list should resemble the number of bars and the shape of the arrays should be 2 (absolute value of min and max error bar for each category).
    error_col: string.
        The color for error bars.
    width: float.
        The bar widths.
    bt: float or int. 
        The reference value of the plot, i.e., where the "zero" line is.
    capsize: float or int
        The size of the error bar caps.
    colors: list. 
        List of strings, should be the size of the number of bars (size of vals).
    """
    n = len(vals) # n is number of bars.
    colors = colors # colors is list of colors for bars.
    _X = np.arange(len(X))
    for i in range(n): # Loop through each bar and plot.
        plt.bar(_X - width/2. + i/float(n)*width, vals[i]-bt, bottom = bt, 
                width=width/float(n), align="edge", color = colors[i], edgecolor = 'black', yerr = errors[i], ecolor = error_col, capsize = capsize)   
    plt.xticks(_X, X) # x-ticks.

def plot_fit(X, Y):
    """Fits a second degree polynomial between two datasets.
        
    Parameters
    ---------
    X: numpy.nadarray.
        1-D NumPy array, the X data to fit the model to.
    Y: numpy.ndarray.
        1-D NumPy array, the Y data to fit the model to.
        
    Returns
    ---------
    X_fit: numpy.nadarray.
        1-D NumPy array, the X values to use to predict the Y value.
    Y_fit: numpy.ndarray.
        1-D NumPy array, the Y values that are predicted for the model based on X.
    a: float.
        The quadratic coefficient in ax^2+bx+c.
    b: float.
        The linear coefficient in ax^2+bx+c.
    c: float.
        The y-intercept coefficient in ax^2+bx+c.
    fit_equation: function object. 
        The polyfit object used to hold the model.
    """
    # Use polyfit to fit a 2nd degree polynomial to the data.
    a, b, c = np.polyfit(X, Y, 2)
    # Fit the model to the data.
    fit_equation = lambda x: (a*x**2) + (b*x) + c
    # Generate an array of X values for the model prediction.
    X_fit = np.linspace(np.nanmin(X), np.nanmax(X), 1000)
    # Predict the Y values based off the selected X values.
    Y_fit = fit_equation(X_fit)

    return X_fit, Y_fit, a, b, c, fit_equation

def read_cust_data(div = 'OKGE', year1 = 1999, year2 = 2022, param = 'Customers'):
    """Reads SPP customer data by division.
        
    Parameters
    ---------
    div: string.
        The balancing authority to use.
    year1: int.
        The first year to use.
    year2: int. 
        The last year to use.
    param: string.
        The parameter to read, in this case Customers.
        
    Returns
    ---------
    year_restr: numpy.ndarray.
        1-D NumPy array, contains array of years for customer data.
    data_restr: numpy.ndarray.
        1-D NumPy array of shape (year_restr,), customer data for each year.
    """
    # Go to directory.
    dir = '/share/data1/Students/ollie/CAOs/Data/Energy/NCustomers/SPP/Divisional/'
    path = os.chdir(dir)

    # Open the data.
    filename = 'n_customers_ba.csv'
    customer_frame = pd.read_csv(filename, skiprows = [0], usecols=[0, 1, 2, 3, 4], names = ['Region', 'Year', 'Revenue', 'Sales', 'Customers'])

    # Get array for year.
    year_arr = customer_frame['Year'].values

    # Now get data and region.
    region_arr = customer_frame['Region'].values
    data_arr = customer_frame[param].values

    # Select data based on region.
    if div == 'KCPL': # If it is KCPL, then add INDN to it to account for a merger.
        ind_region1 = np.where(region_arr == div)[0] # Where it is the region KCPL.
        ind_region2 = np.where(region_arr == 'INDN')[0] # Where it is the region INDN.
        # Get years.
        year_region = year_arr[ind_region1]
        # Add customers for KCPL and INDN due to reporting differences.
        data_region = data_arr[ind_region1]+data_arr[ind_region2]
    else:
        ind_region = np.where(region_arr == div)[0] # Get indices for the division.

        # Restrict the year and data to that division.
        year_region, data_region = year_arr[ind_region], data_arr[ind_region]
    # Now find where the year is in your bounded range.
    year_ind = np.where((year_region >= year1)&(year_region <= year2))[0]
    # Restrict the data and years further by selected year.
    year_restr, data_restr = year_region[year_ind], data_region[year_ind]

    return year_restr, data_restr

def rsquared(ytrue, ypred):
    """Calculates the coefficient of determination (R2) for a given model and obs.
        
    Parameters
    ---------
    ytrue: numpy.ndarray.
        1-D NumPy array, contains the true data.
    ypred: numpy.ndarray
        1-D NumPy array, contains the predicted data from the model.
        
    Returns
    ---------
    R2: float.
        The coefficient of determination for a given model.
    """
    # SST (Sum of Squares Total) calculation.
    mean_ytrue = np.nanmean(ytrue) # Get mean of true values.
    sst = np.sum((ytrue - mean_ytrue)**2) # SST

    # SSE (Sum of Squares Error) calculation.
    sse = np.nansum((ytrue - ypred)**2)

    # Coefficient of determination calculation.
    R2 = 1-(sse/sst)

    return R2

def read_regimes(fname):
    """Reads in regimes and associated dates from the txt file.
        
    Parameters
    ---------
    fname: string. 
        The filename of the regime file.
        
    Returns
    ---------
    regimes: numpy.ndarray.
        1-D NumPy array, contains strings with the regimes.
    dates_regime: numpy.ndarray.
        1-D NumPy array, contains datetimes of the dates for the regimes.
    """
    # Get filename of regimes file.
    filename = fname
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

    return regimes, dates_regime