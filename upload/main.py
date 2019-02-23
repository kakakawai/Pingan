# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
from sklearn import neural_network
from collections import defaultdict


path_train = "/data/dm/train.csv"  # 训练文件
#path_train = "train.csv"
path_test = "/data/dm/test.csv"  # 测试文件
#path_test = "train.csv"

path_test_out = "model/"  # 预测结果输出路径为model/xx.csv,有且只能有一个文件并且是CSV格式。


def read_csv():
    """
    文件读取模块，头文件见columns.
    :return: 
    """
    # for filename in os.listdir(path_train):
    tempdata = pd.read_csv(path_train)
    tempdata.columns = ["TERMINALNO", "TIME", "TRIP_ID", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED",
                        "CALLSTATE", "Y"]
    return tempdata


def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return:
    """
    ####################
    output = defaultdict(float)
    ####################

    data = np.array(read_csv())
    trainData = data[:,[1,3,4,6,7,8]]
    print(trainData.shape)
    trainLabel = data[:,-1]
    print(trainLabel.shape)

    model = neural_network.MLPRegressor(hidden_layer_sizes=(13,),solver="adam",learning_rate_init=0.001,max_iter=510,learning_rate="adaptive")
    model.fit(trainData,trainLabel)

    data = np.array(pd.read_csv(path_test))
    testData = data[:,[1,3,4,6,7,8]]
    print(testData.shape)
    rowNum,colNum = testData.shape
    #testLabel = data[:,-1]
    #print(testLabel.shape)

    predict = model.predict(testData)

    for row in range(rowNum):
        user = str(data[row,0])
        if output[user] < predict[row]:
            output[user] = predict[row]
    print(len(output))

    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for Id,Pred in output.items():
            writer.writerow([Id,float(Pred)])

if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    process()
