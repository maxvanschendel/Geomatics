from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from utils import group_dataset_by_datetime
from sklearn.neural_network import MLPClassifier

room_b = pd.read_csv(
    "B.txt",
    names=['datetime', 'mac', 'empty', 'ssid', 'rssi', 'channel', 'empty2', 'freq', 'amount', 'chipset']
)

room_f = pd.read_csv(
    "F.txt",
    names=['datetime', 'mac', 'empty', 'ssid', 'rssi', 'channel', 'empty2', 'freq', 'amount', 'chipset']
)

room_athok = pd.read_csv(
    "athok.txt",
    names=['datetime', 'mac', 'empty', 'ssid', 'rssi', 'channel', 'empty2', 'freq', 'amount', 'chipset']
)

room_b["room"] = "B"
room_f["room"] = "F"
room_athok["room"] = "@hok"

merged_rooms = room_b.append(room_f).append(room_athok)
filtered_dataset = merged_rooms[merged_rooms['ssid'] == 'eduroam']
shuffled_dataset = filtered_dataset.sample(frac=1).reset_index(drop=True)
grouped_by_datetime = group_dataset_by_datetime(shuffled_dataset)
grouped_dataframe = pd.DataFrame(grouped_by_datetime).transpose()

X = grouped_dataframe['data']
Y = grouped_dataframe['room']

unique_macs = filtered_dataset['mac'].unique()
columnized = pd.DataFrame(columns=unique_macs, dtype=int)

# convert dictionaries to dataframe
for i in enumerate(X.tolist()):
    for j in i[1]:
        columnized.loc[i[0], j['mac']] = j['rssi']

# set rssi of all mac addresses that haven't been registered to zero
columnized = columnized.fillna(0)

# construct and train neural network
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 2), random_state=1)
NN.fit(columnized, Y)

wrong = 0
for i in range(Y.size):
    if Y.iloc[i] != NN.predict(columnized.iloc[i:i+1, :])[0]:
        wrong += 1

print(wrong)
