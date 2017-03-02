# -*- coding: utf-8 -*-
# @Time    : 2017/2/19 15:50
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

'''
Taitanic   加载并处理数据
'''
import pandas as pd
import numpy as np
import re

column_name = ['Survived','Pclass', 'Age', 'SibSp', 'Parch', 'Sex','Fare', 'Embarked']


def getRequire_data(data):
    '根据取出对应数据'
    Y = data.iloc[:,0]
    data = data.drop(['Survived'], axis=1)
    return data, Y



##1、加载训练数据
filename = 'train.csv'
Data = pd.read_csv('../DataFile/' + filename);Data = Data.drop(['PassengerId'], axis=1)
training_X, training_Y = getRequire_data(Data)


##2、训练数据预处理
#2.1处理'Sex'
temp = training_X['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
training_X.loc[:,('Sex')] = temp

#2.2处理'Embarked' 三个值转为对应数字0，1，2
if len(training_X.Embarked[ training_X.Embarked.isnull() ]) > 0:
    training_X.loc[(training_X.Embarked.isnull()), 'Embarked'] = training_X.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(training_X['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
training_X.Embarked = training_X.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#2.3处理'Age' 空值填充中位数
median_age = training_X['Age'].dropna().median()  #中位数
if len(training_X.Age[ training_X.Age.isnull() ]) > 0:
    training_X.loc[(training_X.Age.isnull()), 'Age'] = median_age

#2.4处理'Name'
training_X['Name_title'] = training_X['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
training_X['Name_title'][training_X.Name_title=='Jonkheer'] = 'Master'
training_X['Name_title'][training_X.Name_title.isin(['Ms','Mlle'])] = 'Miss'
training_X['Name_title'][training_X.Name_title == 'Mme'] = 'Mrs'
training_X['Name_title'][training_X.Name_title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
training_X['Name_title'][training_X.Name_title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
training_X['Name_title_id'] = pd.factorize(training_X.Name_title)[0]+1



##3、加载测试数据
testData = pd.read_csv('../DataFile/test.csv')
test_index = testData['PassengerId']
test_X = testData.drop(['PassengerId'], axis=1)      #备注：测试集中'fare'一列有行数值为空



##4、测试数据预处理
#4.1处理'Sex'
temp = test_X['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test_X.loc[:,('Sex')] = temp

#4.2处理'Embarked'
if len(test_X.Embarked[ test_X.Embarked.isnull() ]) > 0:
    test_X.loc[(test_X.Embarked.isnull()), 'Embarked'] = test_X.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(test_X['Embarked'])))    # determine all values of Embarked,
test_X.Embarked = test_X.Embarked.map( lambda x: Ports_dict[x]).astype(int)

#4.3处理'Age' 空值填充中位数
median_age = test_X['Age'].dropna().median()  #中位数
if len(test_X.Age[ test_X.Age.isnull() ]) > 0:
    test_X.loc[(test_X.Age.isnull()), 'Age'] = median_age

#4.4处理'Fare'
if len(test_X.Fare[ test_X.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_X[ test_X.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_X.loc[ (test_X.Fare.isnull()) & (test_X.Pclass == f+1 ), 'Fare']= median_fare[f]

#4.5处理'Name'
test_X['Name_title'] = test_X['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])
test_X['Name_title'][test_X.Name_title=='Jonkheer'] = 'Master'
test_X['Name_title'][test_X.Name_title.isin(['Ms','Mlle'])] = 'Miss'
test_X['Name_title'][test_X.Name_title == 'Mme'] = 'Mrs'
test_X['Name_title'][test_X.Name_title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
test_X['Name_title'][test_X.Name_title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
test_X['Name_title_id'] = pd.factorize(test_X.Name_title)[0]+1



##5、加载测试数据的标签
testdata = pd.read_csv('../DataFile/gendermodel.csv')
true_result = testdata['Survived']

##预先删除字符列
training_X = training_X.drop(['Name','Name_title','Ticket','Cabin'], axis=1)
test_X = test_X.drop(['Name','Name_title','Ticket','Cabin'], axis=1)

column_name = training_X.columns

# ##根据特征选择的结果 删除指定的列
# drop_columns = ['Parch', 'Embarked']
# training_X = training_X.drop(drop_columns, axis=1)
# test_X = test_X.drop(drop_columns, axis=1)
# #print 'training_X.columns:', training_X.columns, '\n', 'test_X.columns:', test_X.columns
#
#
#
#标准化训练集和测试集
from sklearn import preprocessing
training_X = pd.DataFrame(preprocessing.scale(training_X, axis=0))
test_X = pd.DataFrame(preprocessing.scale(test_X))






