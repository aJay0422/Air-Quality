import geopandas as gpd
import pandas as pd
from fiona.crs import from_string
import os

city_set = {
    "Merced": {'Merced'},
    "Fresno": {'Fresno'},
    "San Jose": {'San Jose city', 'Santa Clara city', 'Sunnyvale city', 'Campbell city', 'Cupertino city',
                 'Saratoga city', 'Los Gatos town', 'Monte Sereno city', 'Lexington Hills CDP', 'Mountain View city'},
    "Palo Alto": {'Palo Alto', 'Los Altos', 'East Palo Alto', "Stanford", 'Los Altos Hills'}
}


def read_ca_map():
    ca_map = gpd.read_file("./map_data/ca-places-boundaries/CA_Places_TIGER2016.shp")
    crs = from_string("+proj=longlat +datum=WGS84 +no_defs")
    ca_map = ca_map.to_crs(crs)
    return ca_map


def get_city_map(city="Merced"):
    ca_map = read_ca_map()
    city_map = ca_map[ca_map["NAME"].isin(city_set[city])]
    return city_map


def load_city_data(city="Merced"):
    city_folder = f"./pm25_data/PurpleAir/{city.lower()}_folder/"
    file_list = [file_name for file_name in os.listdir(city_folder) if file_name.endswith(".csv")]
    data_dict = {}
    for file_name in file_list:
        df = pd.read_csv(city_folder + file_name)
        df.sort_values("time_stamp", inplace=True)
        data_dict[file_name[:-4]] = df["pm2.5_alt"].to_numpy()
    df = pd.DataFrame(data_dict)
    return df


def get_sensor_locations(sensor_list):
    purpleair_sensors = pd.read_csv("./map_data/PurpleAir SJ,PA,MER, FRE.csv")
    purpleair_sensors = purpleair_sensors.astype({"Sensor ID": str})
    city_sensors = purpleair_sensors[purpleair_sensors["Sensor ID"].isin(sensor_list)]
    return city_sensors.reset_index(drop=True)


if __name__ == "__main__":
    load_city_data("Fresno")