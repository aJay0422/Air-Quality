import pandas as pd


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
