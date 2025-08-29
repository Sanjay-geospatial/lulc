import joblib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import xarray as xr
import rioxarray as rxr
import pandas as pd
import numpy as np
import os
import random
import geopandas as gpd
import pystac_client
import odc.stac
import data
import skops.io as sio
import leafmap.foliumap as leafmap

st.set_page_config(
    page_title="Land cover app",
    page_icon="☘🌳🌴",
    layout="wide",
    initial_sidebar_state="expanded")

model = sio.load("lulc_model.skops")
# model = joblib.load('lulc_model.pkl')

st.title("☘🌳🌴 Land cover App")
st.write("Prototype for Land cover analysis")

st.sidebar.title("Select Date")
year = st.sidebar.selectbox("Year", [2020, 2021, 2022, 2023])
month = st.sidebar.selectbox("Month", list(range(1, 6)))

path = os.path.join('data', 'Chapuralapalli.shp')

s2_monthly, s1_monthly, dem = data.get_satellite_data(shapefile_path = path,
                                  start_date = f"{year:04d}-{month:02d}-01",
                                  end_date = f"{year:04d}-{month+3:02d}-01")

combined_data = data.combine_data(s2_monthly, s1_monthly, dem, month)

gdf = gpd.read_file(path)

raster_df = pd.DataFrame()

for i in combined_data.band.values:
  raster_df[i] = combined_data.sel(band = i).values.flatten()
  raster_df[i].fillna(raster_df[i].mean(), inplace = True)

predicted = model.predict(raster_df)
predicted_reshaped = predicted.reshape(combined_data.shape[1], combined_data.shape[2])

predicted_array = xr.DataArray(
    data=predicted_reshaped,
    coords={
        "y": combined_data.y,
        "x": combined_data.x
    },
    dims=["y", "x"],
    name="lulc"
)

predicted_array = predicted_array.rio.write_crs(combined_data.rio.crs)

predicted_array.rio.to_raster('lulc.tif')

gdf = gdf.to_crs(predicted_array.rio.crs)

m = leafmap.Map()
m.add_basemap("Esri.WorldImagery")
m.add_gdf(
    gdf,
    layer_name="village boundary",
    style={
        "color": "#6BAEED",  
        "weight": 2,        
        "fillOpacity": 0.0 
    }
)
m.to_streamlit(height=300)

classes = {
    0: "Built-up",
    1: "Water",
    2: "Barren",
    3: "Crop",
    4: "Forest"
}

colors = ['#6E2B0C', '#1854AD', '#DB1E07', '#ED3BB7', '#118C13']
xmin, xmax = float(predicted_array.x.min()), float(predicted_array.x.max())
ymin, ymax = float(predicted_array.y.min()), float(predicted_array.y.max())
cmap = mcolors.ListedColormap(colors)

# Plot raster
fig, ax = plt.subplots(figsize=(5, 3))
ax.imshow(
    predicted_array,
    cmap=cmap,
    extent=[xmin, xmax, ymin, ymax],
    origin="upper"
)

# Create a legend
legend_elements = [Patch(facecolor=colors[i], label=classes[i]) for i in classes]
ax.legend(handles=legend_elements, loc='lower left', fontsize=8, frameon=True)

ax.set_title("Predicted Land Use / Land Cover", fontsize=12)
ax.axis("off")

st.pyplot(fig)

pixel_area_m2 = 10*10  # example if 10m resolution
unique, counts = np.unique(predicted_array, return_counts=True)
area_sqkm = counts * pixel_area_m2 / 1e6
area_dict = {classes[int(k)]: v for k,v in zip(unique, area_sqkm)}

# --- Bar Plot ---
fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.bar(area_dict.keys(), area_dict.values(), color=[colors[int(k)] for k in unique])
ax2.set_ylabel("Area (sq. km)")
ax2.set_title("Area of Each LULC Class")
ax2.set_xticklabels(area_dict.keys(), rotation=45, ha='right')

st.pyplot(fig2)
