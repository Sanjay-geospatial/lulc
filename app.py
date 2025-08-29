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
from PIL import Image
import rasterstats

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
predicted_array.rio.to_raster('predicted_lulc.tif')

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
    0: "Barren land",
    1: "Water",
    2: "Agricultural land",
    3: "Settlements",
    4: "Forest",
}
colors = ['#6E2B0C', '#1854AD', '#DB1E07', '#ED3BB7', '#118C13']

# Colormap
cmap = mcolors.ListedColormap(colors)
bounds = np.arange(-0.5, len(classes) + 0.5, 1)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# --- Plot raster + boundary ---
fig, ax = plt.subplots(figsize=(8, 6))

# get spatial extent from raster
xmin, ymin, xmax, ymax = predicted_array.rio.bounds()

# plot raster with correct geospatial extent
im = ax.imshow(
    predicted_array.values,
    cmap=cmap,
    norm=norm,
    extent=[xmin, xmax, ymin, ymax],
    origin="upper"
)

# overlay boundary
gdf.boundary.plot(ax=ax, edgecolor="cyan", linewidth=2)

ax.set_title("Predicted LULC")
ax.axis("off")

# Legend
legend_elements = [Patch(facecolor=colors[i], label=classes[i]) for i in classes]
ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

# Save to PNG
png_path = "lulc_plot.png"
plt.savefig(png_path, bbox_inches="tight", dpi=150)
plt.close(fig)

# --- Download button in Streamlit ---
with open(png_path, "rb") as f:
    st.download_button(
        label="📥 Download LULC Map (PNG)",
        data=f,
        file_name="lulc_classification.png",
        mime="image/png"
    )

stats = rasterstats.zonal_stats(
    gdf,
    "predicted_lulc.tif",
    categorical=True,
    geojson_out=True
)
class_dict = {
    0 : 'barren',
    1 : 'water',
    2 : 'agriculture land',
    3 : 'built up',
    4 : 'forest'
}

records = []
for class_id, class_name in class_dict.items():
    count = stats[0]['properties'].get(class_id, 0) 
    area_ha = (count * 100) / 10000          
    records.append({
        "Class ID": class_id,
        "Class": class_name,
        "Count": count,
        "Area (ha)": area_ha
    })

st.write('Area statistics')
st.table(records)

fig, ax = plt.subplots(figsize=(6,6))

wedges, texts, autotexts = ax.pie(
    df["Area (ha)"],
    labels=df["Class"],
    autopct="%.1f%%",
    startangle=90,
    wedgeprops=dict(width=0.4)  # <-- donut style
)

plt.setp(autotexts, size=10, weight="bold", color="white")
ax.set_title("LULC Area Distribution", fontsize=14)

st.pyplot(fig)
