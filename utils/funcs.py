###### Import modules ######
import numpy as np
from netCDF4 import Dataset, num2date
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from matplotlib import cm, colors
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.path as mpath
import cftime
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr


# Define a normal colormap function.
def NormColorMap(cmap_select, my_bounds):
    """Creates a normalized color map.

    Parameters
    ---------
    cmap_select: String, the cmap to normalize.
    bounds: Numpy array, the array of levels in the colormap. Has to be diverging.

    Returns
    ---------
    my_cmap: The normalized colormap.
    norm: The normalization array.
    """
    cmap = cm.get_cmap(cmap_select)
    bounds = my_bounds
    numColors = bounds.size - 1
    my_colors = []
    for i in range(1,numColors+1):
        if (i != (numColors / 2.)):
            my_colors.append(cmap(i/float(numColors)))
        else:
            my_colors.extend(['white']*2)

    my_cmap = colors.ListedColormap(my_colors)
    my_cmap.set_under(cmap(0.0))
    my_cmap.set_over(cmap(1.0))
    norm = colors.BoundaryNorm(bounds, my_cmap.N)

    return my_cmap, norm

def format_daily_ERA5(var = 'hgt', level = 500, lat_bounds = [90, 0], lon_bounds = [0, 359.5], year_bounds = [1996, 2014], months = ['05','06','07','08','09'], ndays = 153, isentropic = True):
    '''
    Function to format 3D ERA5 daily data into (year, day in season, lat, lon) at a specific pressure level.
    This just makes it easier to subtract climatology.

    Parameters
    -------
    var: string, states the 3D variable to use.
    level: integer, states the pressure level to use.
    lat_bounds: list, length 2, floats/integers with the latitude bounds in from highest to lowest.
    lon_bounds: list, length 2, floats/integers with the longitude bounds in from west to east in degrees east.
    year_bounds: list, length 2, integers representing the year bounds to use.
    months: list, strings of the month labels to use.
    ndays: integer, describes how many days there are in your month range.
    lev_type: boolean, describes whether it is an isentropic level or not.

    Outputs
    -------
    era_var: numpy.ndarray, shape (years, days of season, lat, lon).
        4-D NumPy array of ERA5 daily data for the 3D variable of choice.
    era_time: numpy.ndarray, shape (years, days of season).
        4-D NumPy array of datetimes for the ERA5 data.
    latitude: numpy.ndarray, shape (lat,).
        1-D NumPy array of floats representing latitude.
    longitude: numpy.ndarray, shape (lon,).
        1-D NumPy array of floats representing longitude.
    '''
    if level == None:
        # Go to the data.
        dir = f'/data/deluge/reanalysis/REANALYSIS/ERA5/2D/4xdaily/{var}'
        path = os.chdir(dir)
        # Read in one test file's lat, lon, and level:
        track_file = f'{var}.200001.nc'
        nc_track = Dataset(track_file, 'r')
        lat_track = nc_track.variables['latitude'][:]
        lon_track = nc_track.variables['longitude'][:]
        nc_track.close()

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
        era_var = np.zeros((len(years_ra),ndays,len(latitude),len(longitude)))
        era_time = np.zeros((len(years_ra),ndays),dtype=object)
        # Now loop through the years and months and format the ERA5 data.
        for y, year in enumerate(tqdm(years_ra)):
            season_list = [] # List to append data to.
            dates_list = [] # List to append datetimes to.
            for m, month in enumerate(months):
                fname = f"{var}.{year}{month}.nc" # Set filename.
                print(fname, year, month)
                era = Dataset(fname,'r') # Open file.
                data = era.variables[var][:,lat_index_1:lat_index_2+1,lon_index_1:lon_index_2+1] # Read in the restricted data.
                time = num2date(era.variables['time'][:],era.variables['time'].units,era.variables['time'].calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True) # Read in datetimes.
                era.close() # Close the file.
                reshape = data.reshape(int(len(time)/4), 4, data.shape[1], data.shape[2]) # Reshape to (days, hourly steps, lat, lon)
                daily_ave = np.nanmean(reshape, axis = 1) # Daily average.
                time_select = time[::4] # Get time stamp for every day but only once.
                if daily_ave.shape[0] == 29:
                    daily_ave = daily_ave[:-1]
                    time_select = time_select[:-1]
                else:
                    daily_ave = daily_ave
                    time_select = time_select
                season_list.append(daily_ave) # Append data to the data list.
                # Append the times to the time list.
                for i in range(len(time_select)):
                    dates_list.append(time_select[i])

            # Vertically stack the season_list and convert dates to an array and assign to the arrays.
            era_var[y] = np.vstack(season_list)
            era_time[y] = np.asarray(dates_list)
    else:
        # Go to the data.
        if isentropic == True:
            dir = f'/data/deluge/reanalysis/REANALYSIS/ERA5/isentropic/4xdaily/{var}'
            path = os.chdir(dir)
        else:
            dir = f'/data/deluge/reanalysis/REANALYSIS/ERA5/3D/4xdaily/{var}'
            path = os.chdir(dir)
        # Read in one test file's lat, lon, and level:
        track_file = f'{var}.200001.nc'
        nc_track = Dataset(track_file, 'r')
        lat_track = nc_track.variables['latitude'][:]
        lon_track = nc_track.variables['longitude'][:]
        level_track = nc_track.variables['level'][:]
        nc_track.close()

        # Get lat bound indices.
        lat_index_1 = np.where(lat_track == lat_bounds[0])[0][0]
        lat_index_2 = np.where(lat_track == lat_bounds[1])[0][0]
        # Get lon bound indices.
        lon_index_1 = np.where(lon_track == lon_bounds[0])[0][0]
        lon_index_2 = np.where(lon_track == lon_bounds[1])[0][0]
        level_index = np.where(level_track == level)[0][0] # Get level index.
        # Restrict lat and lon to the bounds.
        latitude = lat_track[lat_index_1:lat_index_2+1]
        longitude = lon_track[lon_index_1:lon_index_2+1]

        # Set array of years.
        years_ra = np.arange(year_bounds[0], year_bounds[1]+1, 1)
        # Set arrays to store the era5 data and times.
        era_var = np.zeros((len(years_ra),ndays,len(latitude),len(longitude)))
        era_time = np.zeros((len(years_ra),ndays),dtype=object)
        # Now loop through the years and months and format the ERA5 data.
        for y, year in enumerate(tqdm(years_ra)):
            season_list = [] # List to append data to.
            dates_list = [] # List to append datetimes to.
            for m, month in enumerate(months):
                fname = f"{var}.{year}{month}.nc" # Set filename.
                print(fname, year, month)
                era = Dataset(fname,'r') # Open file.
                data = era.variables[var][:,level_index,lat_index_1:lat_index_2+1,lon_index_1:lon_index_2+1] # Read in the restricted data.
                time = num2date(era.variables['time'][:],era.variables['time'].units,era.variables['time'].calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True) # Read in datetimes.
                era.close() # Close the file.
                reshape = data.reshape(int(len(time)/4), 4, data.shape[1], data.shape[2]) # Reshape to (days, hourly steps, lat, lon)
                daily_ave = np.nanmean(reshape, axis = 1) # Daily average.
                time_select = time[::4] # Get time stamp for every day but only once.
                if daily_ave.shape[0] == 29:
                    daily_ave = daily_ave[:-1]
                    time_select = time_select[:-1]
                else:
                    daily_ave = daily_ave
                    time_select = time_select
                season_list.append(daily_ave) # Append data to the data list.
                # Append the times to the time list.
                for i in range(len(time_select)):
                    dates_list.append(time_select[i])
            # Vertically stack the season_list and convert dates to an array and assign to the arrays.
            era_var[y] = np.vstack(season_list)
            era_time[y] = np.asarray(dates_list)
    # Format the hours.
    #for i in range(era_time.shape[0]):
        #for j in range(era_time.shape[1]):
            #era_time[i, j] = era_time[i, j].replace(hour = 0)


    return era_var, era_time, latitude, longitude

def minmax_2d_ERA5(var = 'Tmin', lat_bounds = [90, 0], lon_bounds = [0, 359.5], year_bounds = [1996, 2014], months = ['01','02','03','11','12'], ndays = 151):
    '''
    Function to format ERA5 daily min or max data into (year, day in season, lat, lon).
    This just makes it easier to subtract climatology.

    Parameters
    -------
    var: string, states the 2D variable to use.
    lat_bounds: list, length 2, floats/integers with the latitude bounds in from highest to lowest.
    lon_bounds: list, length 2, floats/integers with the longitude bounds in from west to east in degrees east.
    year_bounds: list, length 2, integers representing the year bounds to use.
    months: list, strings of the month labels to use.
    ndays: integer, describes how many days there are in your month range.

    Outputs
    -------
    era_var: numpy.ndarray, shape (years, days of season, lat, lon).
        4-D NumPy array of ERA5 daily min or max data for the variable of choice.
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
    # Read in one test file's lat, lon, and level:
    track_file = f'{var}.200001.nc'
    nc_track = Dataset(track_file, 'r')
    lat_track = nc_track.variables['latitude'][:]
    lon_track = nc_track.variables['longitude'][:]
    nc_track.close()

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
    era_var = np.zeros((len(years_ra),ndays,len(latitude),len(longitude)))
    era_time = np.zeros((len(years_ra),ndays),dtype=object)
    # Now loop through the years and months and format the ERA5 data.
    for y, year in enumerate(tqdm(years_ra)):
        season_list = [] # List to append data to.
        dates_list = [] # List to append datetimes to.
        for m, month in enumerate(months):
            if year >= 2022:
                fname = f"/data/deluge/scratch/tmp/Ollie_ERA5/Tmin/{var}.{year}{month}.nc" # Set filename.
            else:
                fname = f"{var}.{year}{month}.nc" # Set filename.
            print(fname, year, month)
            era = Dataset(fname,'r') # Open file.
            if var == 'Tmin':
                var_ind = 't2m'
            else:
                var_ind = var
            data = era.variables[var_ind][:,lat_index_1:lat_index_2+1,lon_index_1:lon_index_2+1] # Read in the restricted data.
            time = num2date(era.variables['time'][:],era.variables['time'].units,era.variables['time'].calendar, only_use_cftime_datetimes=False, only_use_python_datetimes=True) # Read in datetimes.
            era.close() # Close the file.
            if data.shape[0] == 29:
                data = data[:-1]
                time = time[:-1]
            else:
                data = data
                time = time
            season_list.append(data) # Append data to the data list.
            # Append the times to the time list.
            for i in range(len(time)):
                dates_list.append(time[i])

        # Vertically stack the season_list and convert dates to an array and assign to the arrays.
        era_var[y] = np.vstack(season_list)
        era_time[y] = np.asarray(dates_list)

    return era_var, era_time, latitude, longitude

# Define a draw polygon function
def DrawPolygon(ax, lat_bounds = [-8, -40], lon_bounds = [295, 325], res = 0.5, color = 'black', lw = 2):
    '''
    Function to plot a polygon on a cartopy map.

    Parameters
    -------
    ax: Cartopy axes object for subplot.
    lat_bounds: list of length 2, containing the highest and lowest latitude bounds.
    lon_bounds: list of length 2, containing the furthest west and east longitude bounds (degrees east).
    res: float, grid resolution.
    color: string, color of the box.
    lw: float, width of box lines.

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
    # Define lats.
    lat_restr = np.arange(lat_bounds[0], lat_bounds[1]-res, -res)
    # Define lons.
    lon_restr = np.arange(lon_bounds[0], lon_bounds[1]+res, res)

    # Now get the left side of the box.
    lon_rep_left = np.repeat(lon_bounds[0], lat_restr.shape[0])

    # Now get the right side of the box.
    lon_rep_right = np.repeat(lon_bounds[1], lat_restr.shape[0])
    
    # Now get the upper bound of the box.
    lat_rep_top = np.repeat(lat_bounds[0], lon_restr.shape[0])

    # Now get the lower bound of the box.
    lat_rep_bottom = np.repeat(lat_bounds[1], lon_restr.shape[0])

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

# Define a SP Stereographic Map.
def SP_StereMap(fig, subplot = [1, 1, 1], lon_bounds = [0, 360], lat_bounds = [20, -90]):
    # Circle trajectories.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1 = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection = ccrs.SouthPolarStereo(central_longitude=-80))
    ax1.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    return ax1

# Define a NP Stereographic Map.
def NP_StereMap(fig, subplot = [1, 1, 1], lon_bounds = [0, 360], lat_bounds = [90, 20], cen_lon = -100):
    # Circle trajectories.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax1 = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection = ccrs.NorthPolarStereo(central_longitude=cen_lon))
    ax1.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS)
    ax1.set_boundary(circle, transform=ax1.transAxes)
    return ax1


# Define a Plate Carree Map.
def PlateCarreeMap(fig, subplot = [1, 1, 1], lon_bounds = [260, 330], lat_bounds = [20, -60], cen_lon = 295):
    ax1 = fig.add_subplot(subplot[0], subplot[1], subplot[2], projection=ccrs.PlateCarree(central_longitude = cen_lon))
    ax1.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())
    ax1.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS)
    return ax1

# Define the linear detrend function.
def LinearDetrend(y):
    """Linearly detrends an array of data with time in the first dimension.

    Parameters
    ---------
    y: The array to be detrended.

    Returns
    ---------
    detrendedData: The detrended data.
    longitude: The slope of the trend line.
    """
    time = np.arange(y.shape[0])

    E = np.ones((y.shape[0],2))
    E[:,0] = time

    invETE = np.linalg.inv(np.dot(E.T,E))
    xhat = invETE.dot(E.T).dot(y)
    trend = np.dot(E,xhat)
    detrendedData = y - trend
    return detrendedData,trend

def LinearRegression(x_data, y_data):
    # Set up the matrix E.
    E = np.zeros((y_data.size, 2))
    # We know y is the peak load.
    y = y_data.copy()
    # We know y = Ex + n. x contains the regression coefficient.
    # Set first column of E to be the temperature, and second column as ones.
    E[:, 0] = x_data
    E[:, 1] = 1

    # Now calculate xhat to get slope and y-intercept.

    inv_term = np.linalg.inv(np.dot(E.T, E))
    xhat = np.dot(np.dot(inv_term, E.T), y)

    # For equation y=ax+b:
    a = xhat[0]
    b = xhat[1]

    # Now createa straight line covering all the data you just had.
    min_data = np.nanmin(x_data)
    max_data = np.nanmax(x_data)

    x_graph = np.arange(min_data, max_data+0.1, 0.1)
    y_line = (a*x_graph)+b

    return x_graph, y_line

def is_months(month, month_bnd = [3, 11]):
        return (month <= month_bnd[0]) | (month >= month_bnd[1])


def read_early_load(div = 'OKGE', year1 = 1999, month_bnd = [3, 11]):
    # Go to directory.
    dir = '/share/data1/Students/ollie/CAOs/Data/Energy/Load/SPP/Divisional/1999_2010'
    path = os.chdir(dir)

    # Open the data.
    filename = 'spp_load_1999_2010_ba.csv'
    energy_frame = pd.read_csv(filename, skiprows = [0], usecols=[0, 1, 2, 4, 5, 6], names = ['Region', 'Year', 'Month', 'Day', 'Hour', 'Load'])

    # Get arrays for date.
    year_arr = energy_frame['Year'].values
    month_arr = energy_frame['Month'].values
    day_arr = energy_frame['Day'].values
    hour_arr = energy_frame['Hour'].values

    # Now get load and region.
    region_arr = energy_frame['Region'].values
    #print(set(region_arr))
    load_arr = energy_frame['Load'].values

    # We want just OKGE to start.
    ind_region = np.where(region_arr == div)[0] # Get indices.
    # Restrict.
    year_region, month_region, day_region, hour_region = year_arr[ind_region], month_arr[ind_region], day_arr[ind_region], hour_arr[ind_region]
    load_region = load_arr[ind_region]
    # Now we only want selected period.
    ind_month = np.where((month_region <= month_bnd[0])|(month_region >= month_bnd[1]))[0]
    # Restrict.
    year_period, month_period, day_period, hour_period = year_region[ind_month], month_region[ind_month], day_region[ind_month], hour_region[ind_month]
    load_period = load_region[ind_month] # This is now load for DJF between 1999-2010.

    # Now get datetimes for this period.
    dates_list = []
    for i in range(len(load_period)):
        dates_list.append(datetime(year_period[i], month_period[i], day_period[i], hour_period[i]))
    dates_arr = np.asarray(dates_list) # Make to an array.

    # Now restrict to start year you want.
    year_track = np.array([d.year for d in dates_arr])
    year_ind = np.where(year_track >= year1)[0]

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
    # Go to directory.
    dir = '/share/data1/Students/ollie/CAOs/Data/Energy/Load/SPP/Divisional'
    path = os.chdir(dir)
    # Select parameters.
    year_select = np.arange(year1, year2+1, 1)
    month_select = months

    # Set out lists to store data.
    date_list = []
    load_list = []

    # Loop through years and months.
    for i in tqdm(range(len(year_select))):
        for j in range(len(month_select)):
            # Get filename.
            filename =f"{year_select[i]}/HOURLY_LOAD-{year_select[i]}{month_select[j]}.csv"
            # Open the file.
            energy_frame = pd.read_csv(filename, skipinitialspace=True, skip_blank_lines=True)
            # Drop any lines that have NaN due to empty spaces.
            energy_frame.dropna(how="all", inplace=True)
            # Read in times and load values.
            dates = energy_frame['MarketHour'].values
            # If KCPL then add INDN due to merger.
            if div == 'KCPL':
                load = energy_frame[div].values + energy_frame['INDN'].values
            else: # Otherwise just read in load.
                load = energy_frame[div].values
            # If it is December 2014, format is slightly different for date.
            if ((year_select[i] == 2014)&(int(month_select[j])>= 3)):
                # Get dates.
                date_lst = []
                for k in range(len(dates)): # Loop through dates.
                    if len(dates[k]) <= 10: # Get dates.
                        date_lst.append(datetime.strptime(dates[k]+' 0:00', "%m/%d/%Y %H:%M"))
                    else:
                        date_lst.append(datetime.strptime(dates[k], "%m/%d/%Y %H:%M"))
            # If it is 2015 or 2016, slightly different format for date.
            elif (year_select[i] > 2014)&(year_select[i] < 2017):
                # Get dates.
                date_lst = []
                for k in range(len(dates)): # Loop through dates.
                    if len(dates[k]) <= 10:
                        date_lst.append(datetime.strptime(dates[k]+' 0:00', "%m/%d/%Y %H:%M"))
                    else:
                        date_lst.append(datetime.strptime(dates[k], "%m/%d/%Y %H:%M"))
            # If year is greater than 2017, different format date.
            elif (year_select[i] >= 2017):
                # Get dates.
                date_lst = []
                for k in range(len(dates)): # Loop through dates.
                    date_lst.append(datetime.strptime(dates[k], "%m/%d/%Y %H:%M:%S"))
            else:
                # Get dates.
                date_lst = []
                for k in range(len(dates)): # Loop through dates.
                    date_lst.append(datetime.strptime(dates[k], "%m/%d/%y %H:%M"))

            date_arr = np.array(date_lst) # Make dates into an array.

            date_list.append(date_arr) # Append array of dates for division to main list.
            load_list.append(load) # Append array of loads for division to main list.
    
    # Now stack the arrays.
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

    grouped_ds = ds.sel(time=is_months(ds['time.month'], month_bnd=month_bnd))

    # Sort by time.
    sorted_ds = grouped_ds.sortby('time')
    # Remove leap days.
    leap_remove = sorted_ds.convert_calendar('noleap')

    # Get the data and dates out of the dataset.
    load_data = leap_remove['load'].values
    load_dates = leap_remove['time'].values

    # Find where it is not the first or last day of the dataset.
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

    # Now let's loop through and get indices to remove.
    ind_list = [] # Empty list to append to.
    for i in range(len(incomplete_dt)):
        years_arr = np.array([d.year for d in dt_dates])
        months_arr = np.array([d.month for d in dt_dates])
        days_arr = np.array([d.day for d in dt_dates])

        ind = np.where((years_arr == incomplete_dt[i].year)&(months_arr == incomplete_dt[i].month)&(days_arr == incomplete_dt[i].day))[0]

        ind_list.append(ind)

    conc_inds = np.concatenate((ind_list))

    final_load, final_dates = np.delete(adj_load, conc_inds), np.delete(dt_dates, conc_inds)  

    return final_dates, final_load

def read_ercot(year1, year2):
    # Go to directory.
    dir = '/share/data1/Students/ollie/CAOs/Data/Energy/Load/ERCOT/'
    path = os.chdir(dir)
    # Select parameters.
    year_select = np.arange(year1, year2+1, 1)
    # Empty lists to store load data.
    date_list = []
    load_list = []

    for i in tqdm(range(len(year_select))):
        if (year_select[i] > 2014)&(year_select[i] < 2017):
            # Get filename.
            filename =f"{year_select[i]}/Native_Load_{year_select[i]}.csv"
            # Open the file.
            energy_frame = pd.read_csv(filename, skipinitialspace=True)
            energy_frame.dropna(how="all", inplace=True)
            # Read in times and load values.
            dates = energy_frame['Hour_End'].values
            load = energy_frame['ERCOT'].values

            date_init = []
            for k in range(len(dates)): # Loop through dates.
                date_init.append(datetime.strptime(dates[k], "%d/%m/%Y %H:%M"))
        
            date_arr = np.array(date_init) # Make dates into an array.

            date_list.append(date_arr)
            load_list.append(load)
    
        elif (year_select[i] >= 2017):
            # Get filename.
            filename =f"{year_select[i]}/Native_Load_{year_select[i]}.csv"
            # Open the file.
            energy_frame = pd.read_csv(filename, skipinitialspace=True, thousands=r',')
            energy_frame.dropna(how="all", inplace=True)
            # Read in times and load values.
            if (year_select[i] > 2017)&(year_select[i] < 2021):
                dates = energy_frame['HourEnding'].values
            else:
                dates = energy_frame['Hour Ending'].values
            load = energy_frame['ERCOT'].values

            date_init = []
            for k in range(len(dates)): # Loop through dates.
                if dates[k][11:] == '24:00':
                    month_part = dates[k][0:2]
                    day_part = dates[k][3:5]
                    new_str_prev = f"{month_part}/{day_part}/{year_select[i]} 23:00"
                    date_init.append(datetime.strptime(new_str_prev, "%m/%d/%Y %H:%M")+timedelta(hours = 1))
                elif ('DST' in dates[k]):
                    date_init.append(datetime.strptime(dates[k][:-4], "%m/%d/%Y %H:%M"))
                else:
                    date_init.append(datetime.strptime(dates[k], "%m/%d/%Y %H:%M"))
        
            date_arr = np.array(date_init) # Make dates into an array.

            date_list.append(date_arr)
            load_list.append(load)
        else:
            # Get filename.
            filename =f"{year_select[i]}/{year_select[i]}_ercot_hourly_load_data.csv"
            # Open the file.
            energy_frame = pd.read_csv(filename, skipinitialspace=True)
            energy_frame.dropna(how="all", inplace=True)
            # Read in times and load values.
            dates = energy_frame['Hour_End'].values
            load = energy_frame['ERCOT'].values

            date_init = []
            for k in range(len(dates)): # Loop through dates.
                date_init.append(datetime.strptime(dates[k], "%d/%m/%Y %H:%M"))
        
            date_arr = np.array(date_init) # Make dates into an array.

            date_list.append(date_arr)
            load_list.append(load)

    # Now stack the arrays.
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
        attrs=dict(description=f"Load (MW) for ERCOT"),
    )

    # Group by December-February.
    grouped_ds = ds.sel(ds['time.season']=='DJF')

    # Sort by time.
    sorted_ds = grouped_ds.sortby('time')

    # Remove leap days.
    leap_remove = sorted_ds.convert_calendar('noleap')

    # Get the data and dates out of the dataset.
    load_data = leap_remove['load'].values
    load_dates = leap_remove['time'].values

    # Find where it is not the first or last day of the dataset.
    ind1 = np.where((load_dates >= cftime.DatetimeNoLeap(year1, 1, 2))&(load_dates < cftime.DatetimeNoLeap(year2+1, 1, 1)))[0]
    # Restrict the data.
    final_load = load_data[ind1]
    adj_dates = load_dates[ind1]

    # Convert to python datetimes.
    final_date_lst = [] # List to append dates.
    for i in range(len(adj_dates)):
        final_date_lst.append(datetime(adj_dates[i].year, adj_dates[i].month, adj_dates[i].day, adj_dates[i].hour, adj_dates[i].minute)) # Append the dates.
    # Make dates into array.
    final_dates = np.array(final_date_lst)

    return final_load, final_dates

def subcategorybar(X, vals, errors, error_col, width=0.8, bt = 1, capsize = 10, colors = ['darkblue', 'gray', 'darkred']):
    n = len(vals)
    colors = colors
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i]-bt, bottom = bt, 
                width=width/float(n), align="edge", color = colors[i], edgecolor = 'black', yerr = errors[i], ecolor = error_col, capsize = capsize)   
    plt.xticks(_X, X)

# Define function to take consecutive array indices meeting a certain criteria.
def consec(arr, stepsize=1):
    """Splits an array where it meets a condition consecutively.

    Parameters
    ---------
    arr: numpy array 1d.
    stepsize: how many steps to split by.

    Returns
    ---------
    List of split arrays of indices in original array that meet certain criteria consecutively.
    """
    return np.split(arr, np.where(np.diff(arr) != stepsize)[0]+1)

def plot_fit(X, Y, minX, maxX, deg = 2):
    a, b, c = np.polyfit(X, Y, 2)
    fit_equation = lambda x: (a*x**2) + (b*x) + c
    
    X_fit = np.linspace(minX, maxX, 1000)
    Y_fit = fit_equation(X_fit)

    return X_fit, Y_fit, a, b, c, fit_equation

def read_cust_data(div = 'OKGE', year1 = 1999, year2 = 2022, param = 'Customers'):
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
    if div == 'KCPL':
        ind_region1 = np.where(region_arr == div)[0]
        ind_region2 = np.where(region_arr == 'INDN')[0]
        # Get years.
        year_region = year_arr[ind_region1]
        # Add customers for KCPL and INDN due to reporting differences.
        data_region = data_arr[ind_region1]+data_arr[ind_region2]
    else:
        ind_region = np.where(region_arr == div)[0] # Get indices.

        # Restrict.
        year_region, data_region = year_arr[ind_region], data_arr[ind_region]

    year_ind = np.where((year_region >= year1)&(year_region <= year2))[0]

    year_restr, data_restr = year_region[year_ind], data_region[year_ind]

    return year_restr, data_restr