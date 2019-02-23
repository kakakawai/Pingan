# -*- coding:utf8 -*-
import os
import csv
import time
import pandas as pd
import numpy as np
import itertools
import random
from collections import defaultdict
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn import neural_network

path_train = "/data/dm/train.csv"  # 训练文件
path_test = "/data/dm/test.csv"  # 测试文件
#path_train = "train.csv"
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
    ###########################
    allList = []
    modelList = []
    output = defaultdict(lambda:100.0)
    ###########################

    data = pd.DataFrame(read_csv())
    testData = pd.read_csv(path_test)
    trainLabel = data["Y"]
    newTrainData = np.zeros((data.shape[0], 1))
    newTestData = np.zeros((testData.shape[0],1))

    #print(trainLabel.shape)

    features = ["TIME", "LONGITUDE", "LATITUDE", "DIRECTION", "HEIGHT", "SPEED","CALLSTATE"]
    # print(data)
    #print(features)

    gen = time.time()
    for i in range(len(features)-1):
        item = list(map(list, itertools.combinations(features, i + 1)))
        allList.extend(item)

    for item in allList:
        if ("LONGITUDE" in item) ^ ("LATITUDE" in item):
            allList.remove(item)
    print("[+]Gen:" + str(time.time() - gen))

    #allList = allList[::]
    #print(len(allList))
    for i in range(33):
        index = int(random.uniform(0,len(allList)))
        del(allList[index])
    print(len(allList))

    OneLay = time.time()
    for featureItem in allList:
        #if ("LONGITUDE" in featureItem) ^ ("LATITUDE" in featureItem):  continue
        #print(featureItem)
        trainData = data.loc[:,featureItem]
        #print(trainData.shape)
        model = DecisionTreeRegressor()
        model.fit(trainData,trainLabel)
        #modelList.append(model)
        predict = model.predict(trainData)
        newTrainData = np.column_stack((newTrainData,predict))
        predict = model.predict(testData.loc[:, featureItem])
        newTestData = np.column_stack((newTestData, predict))
    print(np.shape(newTrainData))
    print(np.shape(newTestData))
    print("[+]OneLayer:" + str(time.time() - OneLay))

    TwoLay = time.time()
    resultModel = neural_network.MLPRegressor()
    resultModel.fit(newTrainData,trainLabel)
    predict = resultModel.predict(newTestData)
    print(np.shape(predict))
    print("[+]TwoLayer:" + str(time.time() - TwoLay))

    outBegin = time.time()
    for row in range(predict.shape[0]):
        user = testData.iat[row, 0]
        if output[user] > predict[row]:
            output[user] = predict[row]

    with(open(os.path.join(path_test_out,"test.csv"),mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id","Pred"])
        for Id,Pred in output.items():
            Pred = 0 if Pred<0.2 else Pred
            writer.writerow([Id,Pred])
    print("[+]Output:"+str(time.time() - outBegin))



if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    a = time.time()
    process()
    b = time.time()
    print("[+]time:"+str(b-a))
