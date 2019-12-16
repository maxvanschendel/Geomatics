from utils import group_dataset_by_datetime
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from numpy.random import rand
import pandas as pd
import os
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

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

# split data into training and testing data
print("- Splitting train/test")
msk = rand(len(grouped_dataframe)) < 0.7
train = grouped_dataframe[msk]
test = grouped_dataframe[~msk]

X_train, X_test = train["data"], test["data"]
Y_train, Y_test = train["room"], test["room"]

# convert dictionaries to dataframes
print("- Converting dicts to dataframe")
unique_macs = filtered_dataset["mac"].unique()
X_columnized_train = pd.DataFrame(columns=unique_macs, dtype=int)
X_columnized_test = pd.DataFrame(columns=unique_macs, dtype=int)

for i in enumerate(X_train.tolist()):
    for j in i[1]:
        X_columnized_train.loc[i[0], j["mac"]] = j["rssi"]

for i in enumerate(X_test.tolist()):
    for j in i[1]:
        X_columnized_test.loc[i[0], j["mac"]] = j["rssi"]

# set rssi of all mac addresses that haven"t been registered to zero
print("- Setting NaN to zero")
X_columnized_train = X_columnized_train.fillna(0)
X_columnized_test = X_columnized_test.fillna(0)


# construct and train neural network, random forest, support vector machine and logistic regression
print("Fitting models")
print("- Neural Network")
NN = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(150, 15), random_state=1, max_iter=2500)\
    .fit(X_columnized_train, Y_train)

print("- Random Forest")
RF = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=0)\
    .fit(X_columnized_train, Y_train)

print("- Support Vector Machine")
SVM = svm.SVC(decision_function_shape="ovo", max_iter=2500)\
    .fit(X_columnized_train, Y_train)

print("- Logistic Regression")
LR = LogisticRegression(random_state=0, solver="lbfgs", multi_class="ovr", max_iter=2500)\
    .fit(X_columnized_train, Y_train)

# load multiple data sets into one datagrame
print("Loading data")
merged_rooms = pd.DataFrame(columns=["datetime", "mac", "empty",
                                     "ssid", "rssi", "channel",
                                     "empty2", "freq", "amount",
                                     "chipset", "room"])
for i in os.listdir("./test_measurements"):
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

# split data into training and testing data
print("- Splitting train/test")
msk = rand(len(grouped_dataframe)) < 0.7
train = grouped_dataframe[msk]
test = grouped_dataframe[~msk]

X_train, X_test = train["data"], test["data"]
Y_train, Y_test = train["room"], test["room"]

# convert dictionaries to dataframes
print("- Converting dicts to dataframe")
X_columnized_train = pd.DataFrame(columns=unique_macs, dtype=int)
X_columnized_test = pd.DataFrame(columns=unique_macs, dtype=int)

for i in enumerate(X_train.tolist()):
    for j in i[1]:
        X_columnized_train.loc[i[0], j["mac"]] = j["rssi"]

for i in enumerate(X_test.tolist()):
    for j in i[1]:
        X_columnized_test.loc[i[0], j["mac"]] = j["rssi"]

# set rssi of all mac addresses that haven"t been registered to zero
print("- Setting NaN to zero")
X_columnized_train = X_columnized_train.fillna(0)
X_columnized_test = X_columnized_test.fillna(0)

# test prediction accuracy on test dataset
classifiers = {'NN': NN, 'RF': RF, 'SVM': SVM, 'LR': LR}
print("Testing accuracy")
print("- No. classes:", len(set(Y_test)))
print("- Classes:", set(Y_test))

np.set_printoptions(precision=1)
for i in classifiers:
    print('- ' + i + ' score:', round(classifiers[i].score(X_columnized_test, Y_test), 3))
    cf = plot_confusion_matrix(classifiers[i], X_columnized_test, Y_test, cmap='Blues',
                               normalize='true', xticks_rotation='vertical', include_values=False)
    plt.title(i+' confusion matrix')
    plt.savefig('./figures/' + i + '_cm.svg')
    plt.close()
