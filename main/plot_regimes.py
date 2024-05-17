###### Import modules ######
import sys
sys.path.insert(2, '/share/data1/Students/ollie/CAOs/project-cold-load/')
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import funcs
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from shapely.geometry import shape

###### Open the hgt and T2M data ######
# Go to data.
dir = '/share/data1/Students/ollie/CAOs/Data/Energy/Regime_Comps/'
path = os.chdir(dir)

# Open the nc file.
nc = Dataset("regime_comps.nc", 'r')
regime_composite = nc.variables['hgt'][:] # In dam for hgt.
regime_composite_t2m = nc.variables['t2m'][:] # T2M
latitude = nc.variables['lat'][:] # Latitude
longitude = nc.variables['lon'][:] # Longitude
ratios_sorted = nc.variables['ratio'][:] # Ratios
nc.close() # Close nc file.

###### Now get the Southwest Power Pool shapefile ######
dir = '/share/data1/Students/ollie/CAOs/Data/Energy/Shapefiles/'
path = os.chdir(dir) # Go to directory.

# Open the shapefile
shpfile = gpd.read_file('Control__Areas.shp')
shpfile = shpfile.to_crs('EPSG:4326') # Convert to standard, i.e., PlateCarree.

# Select the multipolygon.
geometry = shape(shpfile['geometry'].iloc[-1]) # SPP is in the last index.

###### Now plot hgt first ######
# Set color maps.
clevs_hgt = np.arange(-15, 16, 1)
my_cmap_hgt, norm_hgt = funcs.NormColorMap('RdBu_r', clevs_hgt)

# Set bounds for lat and lon.
lat1, lat2 = 80, 20
lon1, lon2 = 180, 330

# Mesh the grid.
lons, lats = np.meshgrid(longitude, latitude)

# Plot hgt.
fig = plt.figure(figsize=(5, 4)) # Set figure up.

ax1 = plt.subplot2grid(shape = (6,4), loc = (0,0), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax1.contourf(lons, lats, regime_composite[0], clevs_hgt, norm = norm_hgt, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_hgt) # Contourf first regime.
ax1.coastlines() # Add coastlines.
ax1.add_feature(cfeature.BORDERS) # Add borders.
ax1.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree()) # Set extent.
plt.title(f"(a) AkR: {np.round(ratios_sorted[0], 1)}%",weight="bold") # Set title.

ax2 = plt.subplot2grid(shape = (6,4), loc = (0,2), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax2.contourf(lons, lats, regime_composite[1], clevs_hgt, norm = norm_hgt, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_hgt) # Contourf second regime.
ax2.coastlines() # Add coastlines.
ax2.add_feature(cfeature.BORDERS) # Add borders.
ax2.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree()) # Set extent.
plt.title(f"(b) ArH: {np.round(ratios_sorted[1], 1)}%",weight="bold") # Set title.

ax3 = plt.subplot2grid(shape = (6,4), loc = (2,0), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax3.contourf(lons, lats, regime_composite[2], clevs_hgt, norm = norm_hgt, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_hgt) # Contourf third regime.
ax3.coastlines() # Add coastlines.
ax3.add_feature(cfeature.BORDERS) # Add borders.
ax3.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree()) # Set extent.
plt.title(f"(c) PT: {np.round(ratios_sorted[2], 1)}%",weight="bold") # Set title.

ax4 = plt.subplot2grid(shape = (6,4), loc = (2,2), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax4.contourf(lons, lats, regime_composite[3], clevs_hgt, norm = norm_hgt, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_hgt) # Contourf fourth regime.
ax4.coastlines() # Add coastlines.
ax4.add_feature(cfeature.BORDERS) # Add borders.
ax4.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree()) # Set extent.
plt.title(f"(d) WCR: {np.round(ratios_sorted[3],1)}%",weight="bold") # Set title.

ax5 = plt.subplot2grid(shape = (6,4), loc = (4,1), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax5.contourf(lons, lats, regime_composite[4], clevs_hgt, norm = norm_hgt, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_hgt) # Contourf fifth regime.
ax5.coastlines() # Add coastlines.
ax5.add_feature(cfeature.BORDERS) # Add borders.
ax5.set_extent([lon1, lon2, lat1, lat2], crs=ccrs.PlateCarree()) # Set extent.
plt.title(f"(e) ArL: {np.round(ratios_sorted[4], 1)}%",weight="bold") # Set title.

cb_ax = fig.add_axes([0.05, -0.02, 0.91, 0.04]) # Add figure axes.
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks=np.arange(-15, 18, 3),extend="both",spacing='proportional') # Figure colorbar.
cbar.set_label("500 hPa Geopotential Height Anomaly (dam)") # Figure cbar label.
cbar.ax.tick_params(labelsize=7) # Figure cbar tick size.
fig.tight_layout() # Figure layout.
plt.savefig("/share/data1/Students/ollie/CAOs/project-cold-load/Figures/Regimes/regimes_hgt.png", bbox_inches = 'tight', dpi = 500) # Save figure.


###### Now plot T2M patterns with SPP overlaid ######
# Set updated lats and lons.
upd_lon1, upd_lon2 = 233, 294
upd_lat1, upd_lat2 = 52, 27.5

# Set colormap.
clevs_t2m = np.arange(-6, 6.5, 0.5)
my_cmap_t2m, norm_t2m = funcs.NormColorMap('RdBu_r', clevs_t2m)

# Plot T2M next
fig = plt.figure(figsize=(5, 4)) # Set up figure.

ax1 = plt.subplot2grid(shape = (6,4), loc = (0,0), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax1.contourf(lons, lats, regime_composite_t2m[0], clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf first regime.
for poly in geometry.geoms: # Loop through geometries in shape file and plot only the largest region.
    if poly.area > 5: # If area larger than 5, plot.
        x,y = poly.exterior.xy
        ax1.plot(x,y,transform=ccrs.PlateCarree(), color = 'black', lw = 1) # Plot shapefile outline.
    else:
        pass
ax1.coastlines() # Add coastlines.
ax1.add_feature(cfeature.BORDERS) # Add borders.
ax1.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
plot1, plot2, plot3, plot4 = funcs.DrawPolygon(ax1, lat_bounds = [43, 31], lon_bounds = [256, 268], res = 0.5, color = 'purple', lw = 1.5) # Plot polygon for the T2M region.
plt.title("(a) AkR T2M",weight="bold") # Set title.

ax2 = plt.subplot2grid(shape = (6,4), loc = (0,2), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax2.contourf(lons, lats, regime_composite_t2m[1], clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf second regime.
for poly in geometry.geoms: # Loop through geometries in shape file and plot only the largest region.
    if poly.area > 5: # If area larger than 5, plot.
        x,y = poly.exterior.xy
        ax2.plot(x,y,transform=ccrs.PlateCarree(), color = 'black', lw = 1) # Plot shapefile outline.
    else:
        pass
ax2.coastlines() # Add coastlines.
ax2.add_feature(cfeature.BORDERS) # Add borders.
ax2.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
plot1, plot2, plot3, plot4 = funcs.DrawPolygon(ax2, lat_bounds = [43, 31], lon_bounds = [256, 268], res = 0.5, color = 'purple', lw = 1.5) # Plot polygon for the T2M region.
plt.title("(b) ArH T2M",weight="bold") # Set title.

ax3 = plt.subplot2grid(shape = (6,4), loc = (2,0), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax3.contourf(lons, lats, regime_composite_t2m[2], clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf third regime.
for poly in geometry.geoms: # Loop through geometries in shape file and plot only the largest region.
    if poly.area > 5: # If area larger than 5, plot.
        x,y = poly.exterior.xy
        ax3.plot(x,y,transform=ccrs.PlateCarree(), color = 'black', lw = 1) # Plot shapefile outline.
    else:
        pass
ax3.coastlines() # Add coastlines.
ax3.add_feature(cfeature.BORDERS) # Add borders.
ax3.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
plot1, plot2, plot3, plot4 = funcs.DrawPolygon(ax3, lat_bounds = [43, 31], lon_bounds = [256, 268], res = 0.5, color = 'purple', lw = 1.5) # Plot polygon for the T2M region.
plt.title("(c) PT T2M",weight="bold") # Set title.

ax4 = plt.subplot2grid(shape = (6,4), loc = (2,2), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax4.contourf(lons, lats, regime_composite_t2m[3], clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf fourth regime.
for poly in geometry.geoms: # Loop through geometries in shape file and plot only the largest region.
    if poly.area > 5: # If area larger than 5, plot.
        x,y = poly.exterior.xy
        ax4.plot(x,y,transform=ccrs.PlateCarree(), color = 'black', lw = 1) # Plot shapefile outline.
    else:
        pass
ax4.coastlines() # Add coastlines.
ax4.add_feature(cfeature.BORDERS) # Add borders.
ax4.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
plot1, plot2, plot3, plot4 = funcs.DrawPolygon(ax4, lat_bounds = [43, 31], lon_bounds = [256, 268], res = 0.5, color = 'purple', lw = 1.5) # Plot polygon for the T2M region.
plt.title("(d) WCR T2M",weight="bold") # Set title.

ax5 = plt.subplot2grid(shape = (6,4), loc = (4,1), colspan = 2, rowspan = 2,projection=ccrs.PlateCarree(central_longitude = 255)) # Subplot format.
cs = ax5.contourf(lons, lats, regime_composite_t2m[4], clevs_t2m, norm = norm_t2m, extend='both', transform=ccrs.PlateCarree(), cmap = my_cmap_t2m) # Contourf fifth regime.
for poly in geometry.geoms: # Loop through geometries in shape file and plot only the largest region.
    if poly.area > 5: # If area larger than 5, plot.
        x,y = poly.exterior.xy
        ax5.plot(x,y,transform=ccrs.PlateCarree(), color = 'black', lw = 1) # Plot shapefile outline.
    else:
        pass
ax5.coastlines() # Add coastlines.
ax5.add_feature(cfeature.BORDERS) # Add borders.
ax5.set_extent([upd_lon1, upd_lon2, upd_lat1, upd_lat2], crs=ccrs.PlateCarree()) # Set extent.
plot1, plot2, plot3, plot4 = funcs.DrawPolygon(ax5, lat_bounds = [43, 31], lon_bounds = [256, 268], res = 0.5, color = 'purple', lw = 1.5) # Plot polygon for the T2M region.
plt.title("(e) ArL T2M",weight="bold") # Set title.

cb_ax = fig.add_axes([0.05, -0.02, 0.91, 0.04]) # Add figure axes.
cbar = fig.colorbar(cs, cax=cb_ax,orientation="horizontal",ticks=np.arange(-6, 8, 2),extend="both",spacing='proportional') # Figure colorbar.
cbar.set_label("2m Temperature Anomaly ($^\circ$C)") # Figure cbar label.
cbar.ax.tick_params(labelsize=7) # Figure cbar tick size.
fig.tight_layout() # Figure layout.
plt.savefig("/share/data1/Students/ollie/CAOs/project-cold-load/Figures/Regimes/regimes_t2m.png", bbox_inches = 'tight', dpi = 500) # Save figure.
