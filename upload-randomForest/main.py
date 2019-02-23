# -*- coding:utf8 -*-
import os
import csv
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn import ensemble

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

def DataProcess(data):
    #a = time.time()
    trainData = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    userSet = set(data["TERMINALNO"])
    # print(userSet)

    for user in userSet:
        userData = data.loc[data["TERMINALNO"] == user]
        tripSet = set(userData["TRIP_ID"])
        # print(tripSet)
        #b = time.time()
        for trip in tripSet:

            tripData = userData.loc[userData["TRIP_ID"] == trip]
            dataLength = float(tripData.shape[0])
            # print(tripData)
            # newFeatures = ["TERMINALNO","TRIP_ID","TIME","CenterLongitude","CenterLatitude","LongitudeSpace","LatitudeSpace","AverageDirection"
            # ,"AverageHeight","maxSpeed","minSpeed","AverageSpeed","callState"]
            comData = [user, trip, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


            #LongitudeMax = tripData["LONGITUDE"].max()
            #LongitudeMin = tripData["LONGITUDE"].min()
            #LatitudeMax = tripData["LONGITUDE"].max()
            #LatitudeMin = tripData["LONGITUDE"].min()
            comData[2] = tripData["TIME"].max() - tripData["TIME"].min()
            #comData[3] = (tripData["LONGITUDE"].max() + tripData["LONGITUDE"].min()) / float(2)
            #comData[4] = (tripData["LATITUDE"].max() + tripData["LATITUDE"].min()) / float(2)
            #comData[5] = tripData["LONGITUDE"].max() - tripData["LONGITUDE"].min()
            #comData[6] = tripData["LATITUDE"].max() - tripData["LATITUDE"].min()
            #comData[3] = (LongitudeMax + LongitudeMin) / float(2)
            #comData[4] = (LatitudeMax + LatitudeMin) / float(2)
            #comData[5] = LongitudeMax - LongitudeMin
            #comData[6] = LatitudeMax - LatitudeMin
            #comData[7] = tripData["DIRECTION"].mean()
            comData[8] = tripData["HEIGHT"].mean()
            comData[9] = tripData["SPEED"].max()
            #comData[10] = tripData["SPEED"].min()
            comData[10] = tripData["SPEED"].mean()
            #callFlag = tripData[tripData["CALLSTATE"] < 4]
            #comData[11] = 0 if callFlag.shape[0] == 0 else 1
            #comData[11] = 1 if tripData["CALLSTATE"].max() < 4 else 0
            #comData[11] = 1 if tripData["CALLSTATE"].min() == 4 else 0
            comData[11] = tripData["CALLSTATE"].min() == 4
            #comData[11] = 1 if tripData["CALLSTATE"].mean() == 4 else 0
            # print(comData)
            trainData = np.row_stack((trainData, comData))
        #f =time.time()
        #print("[+]f:" + str(f - b))
    #print(trainData.shape)
    #g = time.time()
    #print("[+]g:" + str(g - a))
    return trainData[1:]

def LabelProcess(data):
    trainLabel = np.array(data.drop_duplicates(["TERMINALNO","TRIP_ID"])["Y"])
    #print(trainLabel.shape)
    return trainLabel

def process():
    """
    处理过程，在示例中，使用随机方法生成结果，并将结果文件存储到预测结果路径下。
    :return: 
    """
    output = defaultdict(float)
    data = pd.DataFrame(read_csv())
    # print(data)
    trainDataBegin = time.time()
    trainData = DataProcess(data)
    print("[+]TrainData:" + str(time.time() - trainDataBegin))
    trainLabelBegin = time.time()
    trainLabel = LabelProcess(data)
    print("[+]trainLabel:" + str(time.time() - trainLabelBegin))


    modelBegin = time.time()
    model = ensemble.RandomForestRegressor(n_estimators=20, n_jobs=2)
    model.fit(trainData[:,[2,8,9,10,11]], trainLabel)
    print("[+]Model:" + str(time.time() - modelBegin))

    data = pd.read_csv(path_test)
    testDataBegin = time.time()
    testData = DataProcess(data)
    print("[+]TestData:" + str(time.time() - testDataBegin))

    predict = model.predict(testData[:,[2,8,9,10,11]])

    outBegin = time.time()
    for row in range(predict.shape[0]):
        user = testData[row,0]
        if output[user] < predict[row]:
            output[user] = predict[row]


    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id","Pred"])
        for Id,Pred in output.items():
            if Pred < 0.5:
                Pred = 0
            writer.writerow([int(Id),Pred])
    print("[+]Output:"+str(time.time() - outBegin))


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    a = time.time()
    process()
    b = time.time()
    print("[+]time:"+str(b-a))
