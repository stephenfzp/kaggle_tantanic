# -*- coding: utf-8 -*-
# @Time    : 2017/2/19 15:50
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

from sklearn.ensemble import RandomForestClassifier

from Load_data import training_X, training_Y, test_X, true_result, column_name

def random_forest_classifier(train_x, train_y):
    '随机森林分类器'
    model = RandomForestClassifier()
    model.fit(train_x, train_y)
    return model


#使用随机森林评估特征重要性
randomForest_model = random_forest_classifier(training_X, training_Y)    #randomForest_model.feature_importances_  #越高说明特征越重要
print sorted(zip(map(lambda x: round(x, 3), randomForest_model.feature_importances_), column_name[1:]))


##使用RFE(递归特征消除)
print '\n'
from sklearn.feature_selection import RFE
model1 = RandomForestClassifier()
rfe2 = RFE(model1, n_features_to_select=1)
rfe2 = rfe2.fit(training_X, training_Y)
print zip(column_name[1:], map(lambda x: x, rfe2.support_))
print sorted(zip(map(lambda x: round(x, 4), rfe2.ranking_), column_name[1:]))   #ranking_最好的等级为1


# #使用RFECV   #每次运行结果都不太相同
# print '\n'
# from sklearn.feature_selection import RFECV
# model2 = RandomForestClassifier()
# rfecv = RFECV(model2, step=1, cv=3)
# rfecv = rfecv.fit(training_X, training_Y)
# print rfecv.n_features_
# print zip(column_name[1:], map(lambda x: x, rfecv.support_))
# print sorted(zip(map(lambda x: round(x, 4), rfecv.ranking_), column_name[1:]))


#FRE结果：  去除 'Parch' 和 'Embarked' 两列
# [('Pclass', True), ('Age', True), ('SibSp', True), ('Parch', False), ('Sex', True), ('Fare', True), ('Embarked', False)]
# [(1.0, 'Age'), (1.0, 'Fare'), (1.0, 'Pclass'), (1.0, 'Sex'), (1.0, 'SibSp'), (2.0, 'Parch'), (3.0, 'Embarked')]
