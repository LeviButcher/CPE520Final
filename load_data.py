from os.path import join
import numpy as np
import pandas as pd
from os import listdir
import datetime
import pytz


data_dir = "pump-and-dump-dataset/data"
pumpDumpInfo = "pump-and-dump-dataset/pump_telegram.csv"


def get_file_paths(dir):
    return [f for f in listdir(dir)]


def get_data():
    files = get_file_paths(data_dir)
    filePaths = [join(data_dir, f) for f in files]
    data = [pd.read_csv(f) for f in filePaths]

    print(data)
    return data


def get_pump_and_dump_info():
    return pd.read_csv(pumpDumpInfo)


def get_pump_and_dump_info_binance():
    pumps = get_pump_and_dump_info()
    return pumps.loc[pumps.exchange == "binance"]


# Possibly need to squash data that takes place at the same time

# TODO: INVESTIGATE IF this labels correctly WRITE TEST
def label_time_series(X, k, pumpStartDateTime):
    dateDiff = abs(pumpStartDateTime - X.datetime)
    return any(dateDiff <= np.timedelta64(k, 's'))


def load_csv_data(pump_info):
    p = pump_info
    time = '.'.join(p.hour.split(":"))
    filePath = join(data_dir, f"{p.symbol}_{p.date} {time}.csv")
    pump_data = pd.read_csv(filePath, parse_dates=["datetime"])

    return pump_data


def split_csv_data(pump_info, pump_data, split_size):
    p = pump_info
    year, month, day = [int(x) for x in p.date.split("-")]
    hour, minutes = [int(x) for x in p.hour.split(":")]
    pumpStartDateTime = datetime.datetime(
        year, month, day, hour, minutes, 0, 0, tzinfo=pytz.utc)

    n, _ = pump_data.shape
    split_pump_data = np.array_split(pump_data, round(n / split_size))
    split_pump_data = [
        x for x in split_pump_data if x.shape[0] == split_size]

    Y = np.array([label_time_series(x, split_size, pumpStartDateTime)
                  for x in split_pump_data])
    X = np.stack(split_pump_data)
    return X, Y


cached_data = []
dataset_size = 150


def get_dataset(split_size):
    print("Running load data")
    global cached_data
    if len(cached_data) == 0:
        pump_infos = get_pump_and_dump_info_binance()
        cached_data = [(p, load_csv_data(p))
                       for _, p in pump_infos[:dataset_size].iterrows()]

    data = [split_csv_data(p, d, split_size) for p, d in cached_data]
    Xs, Ys = zip(*data)
    Y = np.concatenate(Ys)
    X = np.vstack(Xs)
    return X, Y


# X, Y = get_dataset(5)
# print(X.shape)
# tfRatio = (Y == True).sum() / len(Y) * 100
# print(tfRatio)
