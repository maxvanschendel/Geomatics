from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from utils import group_dataset_by_datetime
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
import os


merged_rooms = pd.DataFrame(columns=['datetime', 'mac', 'empty', 'ssid', 'rssi', 'channel', 'empty2', 'freq', 'amount', 'chipset',"room"])
for i in os.listdir('./data'):
    df = pd.read_csv('./data/'+i, names=['datetime', 'mac', 'empty', 'ssid', 'rssi', 'channel', 'empty2', 'freq', 'amount', 'chipset'], delimiter="\t")
    df['room'] = i[:-4]
    merged_rooms = merged_rooms.append(df)

# preprocess data
filtered_dataset = merged_rooms[merged_rooms['ssid'] == 'eduroam']
shuffled_dataset = filtered_dataset.sample(frac=1).reset_index(drop=True)
grouped_by_datetime = group_dataset_by_datetime(shuffled_dataset)
grouped_dataframe = pd.DataFrame(grouped_by_datetime).transpose()

# split data into training and testing data
msk = np.random.rand(len(grouped_dataframe)) < 0.7
train = grouped_dataframe[msk]
test = grouped_dataframe[~msk]

X_train, X_test = train['data'], test['data']
Y_train, Y_test = train['room'], test['room']
print(Y_train.unique())

unique_macs = filtered_dataset['mac'].unique()
X_columnized_train = pd.DataFrame(columns=unique_macs, dtype=int)
X_columnized_test = pd.DataFrame(columns=unique_macs, dtype=int)

# convert dictionaries to dataframe
for i in enumerate(X_train.tolist()):
    for j in i[1]:
        X_columnized_train.loc[i[0], j['mac']] = j['rssi']

for i in enumerate(X_test.tolist()):
    for j in i[1]:
        X_columnized_test.loc[i[0], j['mac']] = j['rssi']

# set rssi of all mac addresses that haven't been registered to zero
X_columnized_train = X_columnized_train.fillna(0)
X_columnized_test = X_columnized_test.fillna(0)

# construct and train neural network, random forest, support vector machine and logistic regression
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 5), random_state=1,verbose=1).fit(X_columnized_train, Y_train)
RF = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=0,verbose=1).fit(X_columnized_train, Y_train)
SVM = svm.SVC(decision_function_shape="ovo",verbose=1).fit(X_columnized_train, Y_train)
LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr',verbose=1).fit(X_columnized_train, Y_train)

# test prediction accuracy on test dataset
print('NN score:', round(NN.score(X_columnized_test, Y_test), 4))
print('RF score:', round(RF.score(X_columnized_test, Y_test), 4))
print('SVM score:', round(SVM.score(X_columnized_test, Y_test), 4))
print('LR score:', round(LR.score(X_columnized_test, Y_test), 4))
