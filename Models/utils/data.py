import os
import torch
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing

from tqdm import tqdm


def average_hour(df, columns=["longitude", "latitude", "pm25"]):
    """
    Set negative values to zero and average readings for each hour
    Input:
        df: a dataframe with a column named "timestamp" and another column named "pm25
        columns: a list of columns to be averaged
    Ouput:
        df: a dataframe with averaged pm25 readings for each hour
    """

    # decompose timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    # df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['weekday'] = df['timestamp'].dt.weekday

    # set negative pm25 to 0
    df.loc[df["pm25"] < 0, "pm25"] = 0

    # averaging
    df = df.loc[:, columns + ["year", "month", "day", "hour", "weekday"]]
    df = df.groupby(["year", "month", "day", "hour", "weekday"]).mean().reset_index(drop=False)
    
    return df

def pm25_to_aqi(pm25):
    """
    Convert pm25 readings to AQI
    Input:
        pm25: a numpy array of pm25 readings
    Output:
        aqi: a numpy array of AQI
    """
    c = np.array([0, 12, 35.5, 55.5, 150.5, 250.5, 350.5, 500.5])
    i = np.array([0, 50, 100, 150, 200, 300, 400, 500])
    aqi = np.zeros_like(pm25)
    for j in range(1, len(c)):
        mask = (pm25 > c[j-1]) & (pm25 <= c[j])
        aqi[mask] = (i[j] - i[j-1]) / (c[j] - c[j-1]) * (pm25[mask] - c[j-1]) + i[j-1]
    return aqi

def load_data(train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, window=24, batch_size=32, seed=42,
              aqi=False, shuffle=False):
    data_dir = '/Users/shangjiedu/Desktop/aJay/Merced/Research/Air Quality/InterpolationBaseline/data/Oct0123_Jan3024/'
    data_dir = '../InterpolationBaseline/data/Oct0123_Jan3024/'

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    files.sort()
    data = []
    for i, file in enumerate(files):
        df = pd.read_csv(data_dir + file)
        df = average_hour(df)

        # remove sensors with missing data
        if len(df) < 2928:
            print("File{}: {} contains missing hours".format(i, file))
            continue

        # remove sensors with outliers
        if df["pm25"].max() > 500:
            print("File{}: {} contains outliers".format(i, file))
            continue

        data.append(df.loc[:, ['pm25', 'longitude', 'latitude']])

    data = np.array(data).transpose(1, 0, 2)

    if aqi:
        data[:, :, 0] = pm25_to_aqi(data[:, :, 0])

    # reading wind data
    wind_path = "./data/FresnoWeatherOct0123_Jan3124.csv"
    wind_df = pd.read_csv(wind_path)
    wind_df = wind_df.drop_duplicates(subset='datetime')
    wind_data = wind_df.loc[:2928, ['windspeed', 'winddir']].to_numpy()
    wind_data = np.repeat(np.expand_dims(wind_data, axis=1), data.shape[1], axis=1)
    data = np.concatenate([data, wind_data], axis=2)
    
    # train-val-test split
    np.random.seed(seed)
    perm = np.random.permutation(data.shape[1])
    n_train = int(train_ratio * data.shape[1])
    n_val = int(val_ratio * data.shape[1])
    n_test = int(test_ratio * data.shape[1])
    train_idx = perm[:n_train]
    val_idx = perm[n_train: n_train + n_val]
    test_idx = perm[n_train + n_val:]
    idx_list = [train_idx, val_idx, test_idx]

    train_dataset = AirQualityDataset(data, window=window, idx_list=idx_list, dataset_type="train")
    val_dataset = AirQualityDataset(data, window=window, idx_list=idx_list, dataset_type="val")
    test_dataset = AirQualityDataset(data, window=window, idx_list=idx_list, dataset_type="test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader, val_loader, test_loader


def load_data2(train_ratio=0.7, test_ratio=0.3, batch_size=256, seed=42):
    data_dir = '/Users/shangjiedu/Desktop/aJay/Merced/Research/Air Quality/InterpolationBaseline/data/Oct0123_Jan3024/'
    data_dir = '../InterpolationBaseline/data/Oct0123_Jan3024/'

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    files.sort()
    data = []
    for i, file in enumerate(files):
        df = pd.read_csv(data_dir + file)
        df = average_hour(df)

        # remove sensors with missing data
        if len(df) < 2928:
            print("File{}: {} contains missing hours".format(i, file))
            continue

        # remove sensors with outliers
        if df["pm25"].max() > 500:
            print("File{}: {} contains outliers".format(i, file))
            continue

        data.append(df.loc[:, ['pm25', 'longitude', 'latitude']])

    data = np.array(data).transpose(1, 0, 2)

    # reading wind data
    wind_path = "data/FresnoWeatherOct0123_Jan3024.csv"
    wind_df = pd.read_csv(wind_path)
    wind_df = wind_df.drop_duplicates(subset='datetime')
    wind_data = wind_df.loc[:2928, ['windspeed', 'winddir']].to_numpy()
    wind_data = np.repeat(np.expand_dims(wind_data, axis=1), data.shape[1], axis=1)
    data = np.concatenate([data, wind_data], axis=2)

    np.random.seed(seed)
    perm = np.random.permutation(data.shape[1])
    n_train = int(train_ratio * data.shape[1])
    n_test = int(test_ratio * data.shape[1])
    train_idx = perm[:n_train]
    val_idx = None
    test_idx = perm[n_train:]
    idx_list = [train_idx, val_idx, test_idx]

    train_dataset = AirQualityDataset(data, idx_list=idx_list, dataset_type="train")
    test_dataset = AirQualityDataset(data, idx_list=idx_list, dataset_type="test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class AirQualityDataset(torch.utils.data.Dataset):
    def __init__(self, data, window=24, idx_list=None, dataset_type="train"):
        """
        Input:
            data: a 3D numpy array with shape (n_steps, n_sensors, 5), where the last dimension contains pm25 readings, longitude and latitude, wind speed and wind direction
            window: the number of lookback hours
        """
        # min-max normalization of location
        self.locations = torch.from_numpy(data[0, :, 1:3]).float()   # (n_sensors, 2)
        self.locations_unnormalized = self.locations.clone()
        self.locations[:, 0] = (self.locations[:, 0] - self.locations[:, 0].min()) / (self.locations[:, 0].max() - self.locations[:, 0].min())
        self.locations[:, 1] = (self.locations[:, 1] - self.locations[:, 1].min()) / (self.locations[:, 1].max() - self.locations[:, 1].min())

        # normalization of readings
        self.readings = torch.from_numpy(data[:, :, 0]).float()   # (n_steps, n_sensors)
        
        # normalization of wind speed and wind direction
        self.wind = torch.from_numpy(data[:, 0, 3:5]).float()
        self.wind[self.wind == 360] = 0
        self.wind_unnormalized = self.wind.clone()
        self.wind[:, 1] = self.wind[:, 1] / 360

        # calculate the wind features for each pair of sensors
        # self.wind_features = self.get_wind_feature(self.locations, self.wind)
        self.wind_features = torch.load('./data/wind_features.pt')

        self.window = window
        self.train_idx = idx_list[0]
        self.val_idx = idx_list[1]
        self.test_idx = idx_list[2]
        self.type= dataset_type

    def __len__(self):
        if self.type == "train":
            return (self.readings.shape[0] - self.window + 1) * len(self.train_idx)
        elif self.type == "val":
            return (self.readings.shape[0] - self.window + 1) * len(self.val_idx)
        else:
            return (self.readings.shape[0] - self.window + 1) * len(self.test_idx)
        
    def __getitem__(self, idx):
        if self.type == "train":
            target_idx = self.train_idx[idx % len(self.train_idx)]
            monitored_idx = np.concatenate([self.train_idx[: idx % len(self.train_idx)], self.train_idx[idx % len(self.train_idx) + 1:]])
            start = idx // len(self.train_idx)
            end = start + self.window
        elif self.type == "val":
            target_idx = self.val_idx[idx % len(self.val_idx)]
            monitored_idx = self.train_idx
            start = idx // len(self.val_idx)
            end = start + self.window
        else:
            target_idx = self.test_idx[idx % len(self.test_idx)]
            monitored_idx = self.train_idx
            start = idx // len(self.test_idx)
            end = start + self.window

        target_location = self.locations[target_idx, :]
        target_reading = self.readings[start:end, target_idx]
        monitored_locations = self.locations[monitored_idx, :]
        monitored_readings = self.readings[start:end, monitored_idx]
        all_idx = np.concatenate([monitored_idx, [target_idx]])
        wind_features = self.wind_features[all_idx][:, all_idx, start:end, :]

        return monitored_locations, monitored_readings, target_location, target_reading, wind_features
    
    def get_wind_feature(self, locs, winds):
        """
        : param locs: a torch tensor of size (num_nodes, 2)
        : param winds: a torch tensor of size (T, 2)
        : return: a torch tensor of size (num_nodes, num_nodes, T, 2)                                             
        """
        N, _ = locs.size()
        T, _ = winds.size()

        def wind_scale(angles):
            """
            rescale angles from [-1, 1] with east as 0 counterclockwise
            to [0, 1] with north as 0 clockwise
            : param angles a torch tensor of size (*)
            """
            angles = (angles + 1) / 2 * 360
            angles = (-angles + 90) % 360
            return angles / 360
        
        # get distance and angle features between each pair of nodes
        locs_feature = torch.zeros(N, N, 2)
        for i in range(N):
            for j in range(N):
                dist = torch.sqrt(torch.sum((locs[i] - locs[j]) ** 2))
                angle = torch.atan2(locs[j, 1] - locs[i, 1], locs[j, 0] - locs[i, 0])
                angle = angle / np.pi
                angle = wind_scale(angle)
                locs_feature[i, j] = torch.tensor([dist, angle])

        # calculate wind features
        wind_features = torch.zeros(N, N, T, 3)
        for t in tqdm(range(T)):
            for i in range(N):
                for j in range(N):
                    speed, direction = winds[t]
                    dist, angle = locs_feature[i, j]
                    alpha = torch.abs(angle - direction)
                    cos_alpha = torch.cos(alpha * np.pi)
                    wind_features[i, j, t] = torch.tensor([dist, speed, cos_alpha])
        torch.save(wind_features, './data/wind_features.pt')
        return wind_features