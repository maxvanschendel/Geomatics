from utils import group_dataset_by_datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
import seaborn as sn

# load multiple data sets into one datagrame
print("Loading data")
merged_rooms = pd.DataFrame(columns=["datetime", "mac", "empty",
                                     "ssid", "rssi", "channel",
                                     "empty2", "freq", "amount",
                                     "chipset", "room"])
for i in os.listdir("./data"):
    print("-", i)
    df = pd.read_csv("./data/"+i, names=["datetime", "mac", "empty",
                                         "ssid",  "rssi", "channel",
                                         "empty2", "freq", "amount",
                                         "chipset", "room"], delimiter="\t")
    df["room"] = i[:-4]
    merged_rooms = merged_rooms.append(df)


# preprocess data
print("Preprocessing data")
print("- Filtering eduroam")
filtered_dataset = merged_rooms[merged_rooms["ssid"] == "eduroam"]

print("- Shuffling")
shuffled_dataset = filtered_dataset.sample(frac=1).reset_index(drop=True)

print("- Grouping by datetime")
grouped_by_datetime = group_dataset_by_datetime(shuffled_dataset)
grouped_dataframe = pd.DataFrame(grouped_by_datetime).transpose()

X_train = grouped_dataframe["data"]
Y_train = grouped_dataframe["room"].reset_index()

# convert dictionaries to dataframes
print("- Converting dicts to dataframe")
unique_macs = filtered_dataset["mac"].unique()

X_columnized_train = pd.DataFrame(columns=unique_macs,dtype=int)
for i in enumerate(X_train.tolist()):
    for j in i[1]:
        X_columnized_train.loc[i[0], j["mac"]] = j["rssi"]


# set rssi of all mac addresses that haven"t been registered to zero
print("- Setting NaN to zero")
X_columnized_train = X_columnized_train.fillna(0)

X_columnized_train['room'] = Y_train['room']
grouped_by_room = X_columnized_train.groupby('room').mean()
res = pdist(grouped_by_room, 'euclidean')
pairwise = pd.DataFrame(squareform(res), index=grouped_by_room.index, columns= grouped_by_room.index)

plt.figure(figsize=(10, 10))
sn.heatmap(pairwise, cmap='Blues', linewidth = 0, square=True)
plt.savefig('./figures/EDM.svg')
