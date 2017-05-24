#!/usr/bin/python
# -*- coding:utf-8 -*-

from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def load_data(file_name, is_train):
    df = pd.read_csv(file_name)
    df.Sex = df.Sex.map({'female': 0, 'male': 1}).astype(int)

    #Fare
    if len(df.Fare[df.Fare.isnull()]) > 0:
        fare = []
        for i in range(0, 3):
            fare[i] = df.loc[df.Pclass==i+1, 'Fare'].dropna().median()
        for i in range(0, 3):
            df.loc[(df.Fare.isnull()) & (df.Pclass == i+1), 'Fare'] = fare[i]

    if is_train:
        print('随机森林预测缺失年龄：--start--')
        data = df[['Age', 'Survived', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_y = data.loc[data.Age.notnull()]
        age_n = data.loc[data.Age.isnull()]
        x, y = age_y.values[:, 1:], age_y.values[:, 0]
        rfg = RandomForestRegressor(n_estimators=100)
        rfg.fit(x, y)
        age_pred = rfg.predict(age_n.values[:, 1:])
        df.loc[df.Age.isnull(), 'Age'] = age_pred
        print('随机森林预测缺失年龄：--end--')
    else:
        print('随机森林预测缺失年龄2：--start--')
        data = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
        age_y = data.loc[data.Age.notnull()]
        age_n = data.loc[data.Age.isnull()]
        x, y = age_y.values[:, 1:], age_y.values[:, 0]
        rfg = RandomForestRegressor(n_estimators=50)
        rfg.fit(x, y)
        age_pred = rfg.predict(age_n.values[:, 1:])
        df.loc[df.Age.isnull(), 'Age'] = age_pred
        print('随机森林预测缺失年龄2：--end--')
    df.loc[df.Embarked.isnull(), 'Embarked'] = 'U'
    embarked = pd.get_dummies(df.Embarked)
    embarked = embarked.rename(columns=lambda x:'Embarked_'+str(x))
    df = pd.concat((df, embarked), axis=1)
    print(df.describe())
    x = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']]
    # x = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = None
    if 'Survived' in df:
        y = df['Survived']
    x = np.array(x)
    y = np.array(y)
    # x = np.tile(x, (5, 1))
    # y = np.tile(y, (5, ))
    return x, y

if __name__ == '__main__':
    t0 = time()
    x, y = load_data('14.Titanic.train.csv', True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(x_train, y_train)
    y_pred1 = rfc.predict(x_test)
    rate1 = accuracy_score(y_test, y_pred1)

    train_data = xgb.DMatrix(x_train, label=y_train)
    test_data = xgb.DMatrix(x_test, label=y_test)
    params = {'max_depth': 6, 'eta': 0.8, 'silent': 1, 'objective':'binary:logistic'}
    watch_list = [(test_data, 'eval'), (train_data, 'train')]
    bst = xgb.train(params, train_data, num_boost_round=50, evals=watch_list)
    y_pred2 = bst.predict(test_data)
    y_pred2[y_pred2>0.5] = 1
    y_pred2[~(y_pred2 > 0.5)] = 0
    rate2 = accuracy_score(y_test, y_pred2)

    print('随机森林正确率：', rate1)
    print('xgboost正确率：', rate2)
    print('Elapsed time is', time()-t0)
