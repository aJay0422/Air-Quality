import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import os


script_dir = os.path.dirname(os.path.realpath(__file__))


class LandUseMap():
    def __init__(self):
        self.supported_cities = {"fresno": os.path.join(script_dir, "map_data/Fresno_Zoning.geojson")}

    def loadmap(self, city):
        """
        Load the map of the city
        :param city: str, name of the city
        """
        city = city.lower()
        if city not in self.supported_cities:
            raise ValueError("City not supported")
        else:
            self.gdf = gpd.read_file(self.supported_cities[city])
            print("Map of {} loaded".format(city))

    def __call__(self, coord):
        """
        Get the land use information of a give coordinate
        :param coord: tuple, if len(coord) == 2, then quesy a single point(lon, lat)
                             if len(coord) == 4, then query a rectangle(lon_min, lat_min, lon_max, lat_max)
        """
        if len(coord) == 2:
            # single point
            lon, lat = coord
            point = gpd.GeoDataFrame({"geometry": [Point(lon, lat)]}, crs=self.gdf.crs)
            point_in_gdf = gpd.sjoin(point, self.gdf, how="left", predicate="intersects")
            zoning_description = point_in_gdf.iloc[0]["ZoningDescription"]
            zoning_description = zoning_description if isinstance(zoning_description, str) else "No Value"
        elif len(coord) == 4:
            # bouding box
            lon_min, lat_min, lon_max, lat_max = coord
            bbox = Polygon([(lon_min, lat_min), (lon_min, lat_max), (lon_max, lat_max), (lon_max, lat_min)])
            bbox = gpd.GeoDataFrame({"geometry": [bbox]}, crs=self.gdf.crs)
            bbox_in_gdf = gpd.sjoin(bbox, self.gdf, how="left", predicate="intersects")

        return zoning_description


if __name__ == "__main__":
    landusemap = LandUseMap()
    landusemap.loadmap("fresno")
    landusemap((-119.77737, 36.710896 ))