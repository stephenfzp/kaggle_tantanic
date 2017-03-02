# -*- coding: utf-8 -*-
# @Time    : 2017/2/19 16:50
# @Author  : stephenfeng
# @Software: PyCharm Community Edition

from Load_data import training_X, training_Y, test_X, true_result, test_index
import pandas as pd

def neural_network_classifier(train_x, train_y):
    '神经网络'
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(10,5), alpha=0.01)
    model.fit(train_x, train_y)
    return model

##训练模型
neural_network_model = neural_network_classifier(training_X, training_Y)
yPre4 = pd.DataFrame(neural_network_model.predict(test_X), columns=['Survived'])


##预测评分
from sklearn import metrics
print '神经网络：', metrics.accuracy_score(yPre4, true_result)


##把预测结果导入文件中
index_people = pd.DataFrame(test_index)
final_data = pd.merge(index_people, yPre4, left_index=True, right_index=True, how='outer')
#final_data.to_csv('result.csv', index=False)  #生成结果文件