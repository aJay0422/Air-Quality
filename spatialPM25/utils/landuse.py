import geopandas as gpd
from shapely.geometry import Point, Polygon


class LandUseMap():
    def __init__(self):
        self.supported_cities = {"Fresno"}
    
    def loadmap(self, city):
        if city not in  self.supported_cities:
            raise Exception("Unsupported city")
        else:
            file_path = f"./landuse_data/{city}_Zoning.geojson"
            self.gdf = gpd.read_file(file_path)
        
    def __call__(self, coord):
        if len(coord) == 2:
            # single point
            lon, lat = coord
            point = gpd.GeoDataFrame({"geometry": [Point(lon, lat)]}, crs=self.gdf.crs)
            point_in_gdf = gpd.sjoin(point, self.gdf, how="inner", op='intersects')
            zoning_description = point_in_gdf.iloc[0]["ZoningDescription"] if not point_in_gdf.empty else "No Value"
        elif len(coord) == 4:
            # bounding box
            lon_min, lat_min, lon_max, lat_max = coord
            bbox = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
            bbox = gpd.GeoDataFrame({"geometry": [bbox]}, crs=self.gdf.crs)
            bbox_in_gdf = gpd.sjoin(bbox, self.gdf, how="inner", op="intersects")
            zoning_description = bbox_in_gdf["ZoningDescription"].tolist() if not bbox_in_gdf.empty else "No Value"
        return zoning_description