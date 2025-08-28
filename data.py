import os
import streamlit as st
import geopandas as gpd
import pystac_client
import odc.stac
import planetary_computer

def get_satellite_data(shapefile_path, start_date, end_date):
    # Check shapefile
    if not os.path.exists(shapefile_path):
        st.error(f"Shapefile not found: {shapefile_path}. Ensure all .shp, .shx, .dbf, .prj files are present in data/")
        return None, None, None

    gdf = gpd.read_file(shapefile_path)
    st.write("Shapefile loaded")
    bounds = gdf.total_bounds
    bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

    STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
    catalog = pystac_client.Client.open(STAC_URL)

    # --- Sentinel-2 ---
    s2_search = catalog.search(
        collections='sentinel-2-l2a',
        datetime=f"{start_date}/{end_date}",
        bbox=bbox,
        query={'eo:cloud_cover': {'lt': 10}}
    )
    s2_items = list(s2_search.get_all_items())
    st.write(f"Found {len(s2_items)} Sentinel-2 items")

    s2_bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']

    s2_ds = odc.stac.load(
        items=s2_items,
        bands=s2_bands,
        bbox=bounds,
        crs=32643,
        resolution=10
    )
    s2_monthly = s2_ds.groupby('time.month').median(dim='time')
    s2_monthly = s2_monthly.to_array(dim='band')

    # --- Sentinel-1 ---
    s1_search = catalog.search(
        collections='sentinel-1-grd',
        datetime=f"{start_date}/{end_date}",
        bbox=bbox
    )
    s1_items = [planetary_computer.sign(item) for item in s1_search.get_all_items()]
    st.write(f"Found {len(s1_items)} Sentinel-1 items")

    s1_da = odc.stac.load(
        items=s1_items,
        bands=['vv', 'vh'],
        bbox=bounds,
        crs=32643,
        resolution=10
    )

    # --- DEM ---
    dem_search = catalog.search(
        collections='cop-dem-glo-30',
        bbox=bbox
    )
    dem_items = list(dem_search.get_all_items())
    dem_da = odc.stac.load(
        items=dem_items,
        bbox=bounds,
        crs=32643,
        resolution=10
    )

    return s2_monthly, s1_da, dem_da
