
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import rioxarray as rxr 
import rioxarray as rxr
import planetary_computer
import pystac_client
import odc.stac
from datetime import datetime
import skops.io as sio
import streamlit as st
import data
import calendar
import rasterstats

# --- Page config ---
st.set_page_config(
    page_title="Land cover app",
    page_icon="☘🌳🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("☘🌳🌴 Land cover App")
st.write("Prototype for Land cover analysis")

# --- Year and month options ---
years = ['None', 2023, 2024, 2025]
months = ['None', 'January', 'February', 'March', 'April', 'May', 'June', 'July',
          'August', 'September', 'October', 'November', 'December']

months_dict = {
    'January': "01", 'February': "02", 'March': "03", 'April': "04",
    'May': "05", 'June': "06", 'July': "07", 'August': "08",
    'September': "09", 'October': "10", 'November': "11", 'December': "12"
}

# --- Sidebar selections ---
year_selected = st.sidebar.selectbox('Select year', years, index=0)
start_month_selected = st.sidebar.selectbox('Select start month', months, index=0)
end_month_selected = st.sidebar.selectbox('Select end month', months, index=0)

# --- Proceed if valid selections ---
if (
    year_selected != "None" 
    and start_month_selected != "None" 
    and end_month_selected != "None"):
  
  # --- Shapefile ---
  shapefile_path = os.path.join('data', 'Chapuralapalli.shp')
  gdf = gpd.read_file(shapefile_path)
  st.success("✅ Shapefile loaded")
  bounds = gdf.total_bounds
  bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

  # --- STAC catalog ---
  STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
  catalog = pystac_client.Client.open(STAC_URL)

  # --- Dates ---
  start_date = f"{year_selected}-{months_dict[start_month_selected]}-01T00:00:00Z"
  
  last_day = calendar.monthrange(int(year_selected), int(months_dict[end_month_selected]))[1]
  end_date = f"{year_selected}-{months_dict[end_month_selected]}-{last_day:02d}T23:59:59Z"

  st.write(f"📅 Searching between **{start_date}** and **{end_date}**")

  # --- Sentinel-2 search ---

  s2_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        
  s2_search = catalog.search(
      collections=["sentinel-2-l2a"],
      datetime=f"{start_date}/{end_date}",
      bbox=bbox,
      query={"eo:cloud_cover": {"lt": 10}}
  )
  s2_items = list(s2_search.get_all_items())
  s2_items_signed = [planetary_computer.sign(i) for i in s2_items]

  s2_dates = sorted([i.datetime.strftime("%Y-%m-%d") for i in s2_items_signed])

  s1_search = catalog.search(
      collections=["sentinel-1-grd"],
      datetime=f"{start_date}/{end_date}",
      bbox=bbox)
  
  s1_items = list(s1_search.get_all_items())
  s1_items_signed = [planetary_computer.sign(i) for i in s1_items]

  s1_dates = sorted([i.datetime.strftime("%Y-%m-%d") for i in s1_items_signed])
  
  # User picks a single date
  selected_date_first_s2 = st.sidebar.selectbox("Select first Sentinel-2 date",['None'] +  s2_dates, index=0)
  selected_date_last_s2 = st.sidebar.selectbox("Select last Sentinel-2 date",['None'] +  s2_dates, index=0)

  selected_date_first_s1 = st.sidebar.selectbox("Select first Sentinel-1 date", ['None'] + s1_dates, index=0)
  selected_date_last_s1 = st.sidebar.selectbox("Select last Sentinel-1 date",['None'] +  s1_dates, index=0)

  if all(d != 'None' for d in [selected_date_first_s2, selected_date_last_s2,
                             selected_date_first_s1, selected_date_last_s1]):
    
    chosen_item_first_s2 = next(i for i in s2_items_signed if i.datetime.strftime("%Y-%m-%d") == selected_date_first_s2)
    chosen_item_last_s2  = next(i for i in s2_items_signed if i.datetime.strftime("%Y-%m-%d") == selected_date_last_s2)

    chosen_item_first_s1 = next(i for i in s1_items_signed if i.datetime.strftime("%Y-%m-%d") == selected_date_first_s1)
    chosen_item_last_s1  = next(i for i in s1_items_signed if i.datetime.strftime("%Y-%m-%d") == selected_date_last_s1)

    # --- Load datasets ---
    s2_ds_first = odc.stac.load([chosen_item_first_s2], bands = s2_bands, bbox=bbox, crs=32643, resolution=10)
    s2_ds_last  = odc.stac.load([chosen_item_last_s2], bands = s2_bands, bbox=bbox, crs=32643, resolution=10)

    s1_ds_first = odc.stac.load([chosen_item_first_s1], bands=['vv','vh'], bbox=bbox, crs=32643, resolution=10)
    s1_ds_last  = odc.stac.load([chosen_item_last_s1],  bands=['vv','vh'], bbox=bbox, crs=32643, resolution=10)

    # --- DEM ---
    dem_search = catalog.search(collections='cop-dem-glo-30', bbox=bbox)
    dem_items_signed = [planetary_computer.sign(i) for i in dem_search.get_all_items()]
    dem_da = odc.stac.load(dem_items_signed, bbox=bbox, crs=32643, resolution=10)

    s2_first_array = s2_ds_first.squeeze().to_array(dim = 'band')                            
    s2_last_array = s2_ds_last.squeeze().to_array(dim = 'band')
    s1_first_array = s1_ds_first.squeeze().to_array(dim = 'band')
    s1_last_array = s1_ds_last.squeeze().to_array(dim = 'band')
    dem_array = dem_da.to_array().squeeze().expand_dims({'band' : ['dem']}) 

    # st.write('Sentinel 2 first bands', s2_first_array.band)
    # st.write('Sentinel 2 last bands', s2_last_array.band)
    # st.write('Sentinel 1 first bands', s1_first_array.band)
    # st.write('Sentinel 1 last bands', s1_last_array.band)
    # st.write('DEM bands', dem_array.band)
                                 
    # --- Combine datasets ---
    total_ds_first = xr.concat([s2_first_array, s1_first_array, dem_array], dim = 'band')
    total_ds_last  = xr.concat([s2_last_array,  s1_last_array,  dem_array], dim = 'band')

    # st.write('Total dataset first bands', total_ds_first.band.values)
    # st.write('Total dataset last bands', total_ds_last.band.values)

    gdf = gpd.read_file(shapefile_path)
    model = sio.load('lulc_model.skops')

    raster_df_first = pd.DataFrame()

    for i in total_ds_first.band.values:
      raster_df_first[i] = total_ds_first.sel(band = i).values.flatten()
      raster_df_first[i].fillna(raster_df_first[i].mean(), inplace = True)

    raster_df_last = pd.DataFrame()

    for i in total_ds_last.band.values:
      raster_df_last[i] = total_ds_last.sel(band = i).values.flatten()
      raster_df_last[i].fillna(raster_df_last[i].mean(), inplace = True)

    predicted_first = model.predict(raster_df_first)
    predicted_reshaped_first = predicted_first.reshape(total_ds_first.shape[1], total_ds_first.shape[2])

    predicted_array_first = xr.DataArray(
        data=predicted_reshaped_first,
        coords={
            "y": total_ds_first.y,
            "x": total_ds_first.x
        },
        dims=["y", "x"],
        name="lulc_first"
    )

    predicted_array_first = predicted_array_first.rio.write_crs(total_ds_first.rio.crs)
    predicted_array_first.rio.to_raster('predicted_lulc_first.tif')

    predicted_last = model.predict(raster_df_last)
    predicted_reshaped_last = predicted_last.reshape(total_ds_last.shape[1], total_ds_last.shape[2])

    predicted_array_last = xr.DataArray(
        data=predicted_reshaped_last,
        coords={
            "y": total_ds_last.y,
            "x": total_ds_last.x
        },
        dims=["y", "x"],
        name="lulc_last"
    )

    predicted_array_last = predicted_array_last.rio.write_crs(total_ds_last.rio.crs)
    predicted_array_last.rio.to_raster('predicted_lulc_last.tif')

    st.write('First image', np.unique(predicted_array_first, return_counts = True))
    st.write('Last image', np.unique(predicted_array_last, return_counts = True))

    class_dict = {
    0 : 'barren',
    1 : 'water',
    2 : 'agriculture land',
    3 : 'built up',
    4 : 'forest'}

    def area_stats(tif, zone, class_dict):
        # Open raster with rioxarray
        da = rxr.open_rasterio(tif, masked=True)
        
        # Get pixel resolution (in map units, usually meters)
        res_x, res_y = da.rio.resolution()
        pixel_area = abs(res_x * res_y)      # m²
        pixel_area_ha = pixel_area / 10000   # hectares
        
        # Run zonal stats
        stats = raster_stats.zonal_stats(
            zone,
            tif,
            categorical=True,
            geojson_out=False
        )
        
        # Build results table
        records = []
        for class_id, class_name in class_dict.items():
            count = stats[0].get(class_id, 0)  # pixel count for class
            area_ha = count * pixel_area_ha
            records.append({
                "Class ID": class_id,
                "Class": class_name,
                "Count": count,
                "Area (ha)": area_ha
            })
        
        df = pd.DataFrame(records)
        return df

    area_first = area_stats('predicted_lulc_first.tif', gdf, class_dict)
    area_last = area_stats('predicted_lulc_last.tif', gdf, class_dict)

    st.write(f'Area statistics as of {selected_date_first_s2}')
    st.table(area_first)

    st.write(f'Area statistics as of {selected_date_last_s2}')
    st.table(area_last)   

  else:
    st.info("👆 Please select valid dates for both Sentinel-1 and Sentinel-2.")      
      

