def get_satellite_data(shapefile_path, start_date, end_date):
  if not os.path.exists(shapefile_path):
    st.error(f"Shapefile not found: {shapefile_path}. Did you upload all shapefile components in the data/ folder?")
  else:
      try:
          gdf = gpd.read_file(shapefile_path)
      except Exception as e:
          st.error(f"Failed to read shapefile: {e}")
      else:
          st.write("Shapefile successfully loaded! Here's the bounding box:", gdf.total_bounds)
  bounds = gdf.total_bounds
  bbox = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))

  STAC_URL = "https://earth-search.aws.element84.com/v1"

  catalog = pystac_client.Client.open(STAC_URL)

  search = catalog.search(
      collections = 'sentinel-2-l2a',
      datetime = start_date+'/'+end_date,
      bbox = bbox,
      query = {'eo:cloud_cover': {'lt': 10}}
  )

  items = list(search.get_all_items())
  print(len(items))

  bands = ['red', 'green', 'blue', 'nir', 'nir08', 'nir09', 'rededge1', 'rededge2', 'rededge3', 'swir16', 'swir22']

  ds = odc.stac.load(
      items = items,
      bands = bands,
      bbox= bounds,
      crs= 32643,
      resolution=10
  )

  monthly_data = ds.groupby('time.month').median(dim = 'time')
  monthly_data = monthly_data.to_array(dim = 'band')

  s1_search = catalog.search(
    collections = 'sentinel-1-grd',
    datetime = '2023-01-01/2023-04-01',
    bbox = bbox
)

  s1_items = list(s1_search.get_all_items())

  s1_da = odc.stac.load(
      items = s1_items,
      bands = ['vv', 'vh'],
      bbox = bbox,
      crs = 32643,
      resolution = 10
  )

  s1_da_monthly = s1_da.groupby('time.month').median(dim = 'time')

  dem_search = catalog.search(
    collections=["cop-dem-glo-30"],
    bbox=bbox
)

  # Get items from DEM search
  dem_items = list(dem_search.get_all_items())

  # Load DEM with odc.stac
  dem_da = odc.stac.load(
      items=dem_items,
      bbox=bbox,
      crs="EPSG:32643",   # UTM zone 43N
      resolution=10       # resample DEM to 10 m (original is 30 m)
  )
  dem_da

  return monthly_data, s1_da_monthly, dem_da

def combine_data(s2, s1, dem, month):
  x1 = s2.sel(month = month).drop_vars(['month'], errors='ignore')
  x2 = s1.sel(month = month).to_array(dim = 'band').drop_vars(['month'], errors='ignore')
  x3 = dem.to_array().squeeze().expand_dims(dim  = {'band' : ['dem']}).drop_vars(['time', 'variable'], errors='ignore')

  combined = xr.concat([x1, x2, x3], dim = 'band')
  return combined

