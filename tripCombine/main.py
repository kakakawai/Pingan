# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from sklearn import ensemble

#path_train = "/data/dm/train.csv"  # 训练文件
#path_test = "/data/dm/test.csv"  # 测试文件
path_train = "train.csv"
path_test = "train.csv"

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

def LabelProcess(data):
    #trainLabel = np.array(data.drop_duplicates(["TERMINALNO","TRIP_ID"])["Y"])
    trainLabel = np.array(data.drop_duplicates(["TERMINALNO","TRIP_ID"])["Y"])
    #print(trainLabel.shape)
    return trainLabel

def DataProcess(data):
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return:
    """
    #trainData = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    #data = pd.DataFrame(read_csv())
    tripData = data.groupby(["TERMINALNO","TRIP_ID"])
    tripDataIndices = pd.DataFrame(list(tripData.groups.keys()))
    #print(tripDataIndices.dtypes)
    #for (name,tripData) in groupData:
    time = tripData["TIME"].max() - tripData["TIME"].min()
    speedMax = tripData["SPEED"].max()
    LongitudeMax = tripData["LONGITUDE"].max()
    LongitudeMin = tripData["LONGITUDE"].min()
    LatitudeMax = tripData["LONGITUDE"].max()
    LatitudeMin = tripData["LONGITUDE"].min()
    centerLongitude = (LongitudeMax + LongitudeMin) / float(2)
    centerLatitude = (LatitudeMax + LatitudeMin) / float(2)
    spaceLongitude = LongitudeMax - LongitudeMin
    spaceLatitude = LatitudeMax - LatitudeMin
    heightMean = tripData["HEIGHT"].mean()
    speedMean = tripData["SPEED"].mean()
    callstate = tripData["CALLSTATE"].min() == 4
    trainData = pd.concat([time, speedMax,speedMean,LongitudeMax,LongitudeMin,LatitudeMax,LatitudeMin,centerLongitude,centerLatitude,spaceLongitude,spaceLatitude,heightMean,callstate], axis=1)
    #trainData = data.drop_duplicates(["TERMINALNO","TRIP_ID"])#["TERMINALNO","TRIP_ID"]
    #print(tripData.indices.keys())
    #print(type(trainData))
    #print(type(tripDataIndices))
    #print(trainData.shape)
    #print(tripDataIndices.shape)
    #trainData = pd.concat([tripDataIndices,trainData],axis=1)
    #trainData = pd.merge(tripDataIndices, trainData, left_index=True, right_index=True, how='outer')
    #tripDataIndices.append(trainData)
    #print(trainData)
    #print(tripDataIndices.shape)


    return trainData,tripDataIndices

def process():
    output = defaultdict(lambda:500.0)
    data = pd.DataFrame(read_csv())

    dataPreBegin = time.time()
    trainData,trainDataIndices = DataProcess(data)
    print("[+]Data Preprocess:" + str(time.time() - dataPreBegin))

    labelPreBegin = time.time()
    trainLabel = LabelProcess(data)
    print("[+]Label Preprocess:" + str(time.time() - labelPreBegin))

    modelBegin = time.time()
    model = ensemble.RandomForestRegressor(n_estimators=22, n_jobs=2)
    model.fit(trainData, trainLabel)
    print("[+]Model:" + str(time.time() - modelBegin))

    data = pd.read_csv(path_test)
    testDataBegin = time.time()
    testData,testDataIndices = DataProcess(data)
    print("[+]TestData:" + str(time.time() - testDataBegin))

    predictBegin = time.time()
    predict = model.predict(testData)
    print("[+]Predict:" + str(time.time() - predictBegin))

    outBegin = time.time()
    for i in range(testData.shape[0]):
        user = testDataIndices.iloc[i,0]
        if output[user] > predict[i]:
            output[user] = predict[i]

    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for id,pred in output.items():
            #pred = 0 if pred < 0.5 else pred
            writer.writerow([id, pred])
    print("[+]Output:" + str(time.time() - outBegin))


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    a = time.time()
    process()
    print("[+]total:"+str(time.time()-a))