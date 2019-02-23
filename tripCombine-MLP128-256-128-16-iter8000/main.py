# -*- coding:utf8 -*-
import os
import csv
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.neural_network import MLPRegressor

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

def get_hour(value):
    format = '%H'
    value = time.localtime(value)
    #print(value)
    dt = int(time.strftime(format, value))
    if 0<=dt<6:  dt = 1
    elif 23 <= dt <= 24:dt = 1
    elif 7 <= dt < 9: dt = 2
    elif 11 <= dt < 13: dt = 3
    elif 17 <= dt < 20: dt = 4
    else: dt = 0

    return dt

def speed_analysis(data):
    lowSpeedRate = data.loc[data<10].shape[0]/float(data.shape[0])
    #print(lowSpeedRate)
    return lowSpeedRate

def get_tripLong(data):
    print(data)
    return 11

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
    speedMean = data[data["SPEED"]>=0]['SPEED'].mean()
    directionMean = data[data["DIRECTION"]>=0]['DIRECTION'].mean()

    data['SPEED'].replace(-1, speedMean, inplace=True)
    data['DIRECTION'].replace(-1, directionMean, inplace=True)  # 异常值赋平均

    tripData = data.groupby(["TERMINALNO","TRIP_ID"])
    tripDataIndices = pd.DataFrame(list(tripData.groups.keys()))

    #print(tripDataIndices.dtypes)
    #for (name,tripData) in groupData:
    #Time = tripData["TIME"].max() - tripData["TIME"].min()
    #print(tripData["TIME"].get_group((1,1)))
    TimeCost = tripData["TIME"].max()-tripData["TIME"].min()
    Time = tripData["TIME"].mean()
    Time = Time.apply(get_hour)
    enc = OneHotEncoder()
    enc.fit([[0],[1],[2],[3],[4]])
    #print(pd.DataFrame(np.array(Time)))
    Time = pd.DataFrame(enc.transform(pd.DataFrame(np.array(Time))).toarray(),dtype="int")

    speedMax = tripData["SPEED"].max()
    lowSpeedRate = tripData["SPEED"].apply(speed_analysis)

    LongitudeMax = tripData["LONGITUDE"].max()
    LongitudeMin = tripData["LONGITUDE"].min()
    LongitudeStd = tripData["LONGITUDE"].std()
    LongitudeStdMean = LongitudeStd.mean()
    LongitudeStd = LongitudeStd.fillna(LongitudeStdMean)

    LatitudeMax = tripData["LATITUDE"].max()
    LatitudeMin = tripData["LATITUDE"].min()
    LatitudeStd = tripData["LATITUDE"].std()
    LatitudeStdMean = LatitudeStd.mean()
    LatitudeStd = LatitudeStd.fillna(LatitudeStdMean)

    centerLongitude = (LongitudeMax + LongitudeMin) / float(2)
    centerLatitude = (LatitudeMax + LatitudeMin) / float(2)
    #tripDiffData = tripData.diff()
    #print(type(tripDiffData))
    #tripLong = np.sqrt(np.square(tripDiffData["LONGITUDE"]) + np.square(tripDiffData["LATITUDE"]))
    #print(tripLong)
    #spaceLongitude = LongitudeMax - LongitudeMin
    #spaceLatitude = LatitudeMax - LatitudeMin
    heightMean = tripData["HEIGHT"].mean()
    heightStd = tripData["HEIGHT"].std()
    heightStdMean = heightStd.mean()
    heightStd = heightStd.fillna(heightStdMean)
    speedMean = tripData["SPEED"].mean()
    callstate = tripData["CALLSTATE"].min() == 4
    #enc = OneHotEncoder()
    #enc.fit([[0],[1]])
    #print(enc.n_values_)
    #callstate = pd.DataFrame(enc.transform(pd.DataFrame(callstate)))
    #print(type(callstate))
    trainData = pd.concat([TimeCost,speedMax,speedMean,lowSpeedRate,LongitudeMax,LongitudeMin,LongitudeStd,LatitudeMax,LatitudeMin,LatitudeStd,centerLongitude,centerLatitude,heightMean,heightStd,callstate], axis=1)
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
    trainData = pd.DataFrame(np.array(trainData))
    trainData = pd.concat([Time,trainData],axis=1)
    #print(trainData)
    return trainData,tripDataIndices

def process():
    #output = defaultdict(lambda:500.0)
    output = defaultdict(float)


    data = pd.DataFrame(read_csv())
    dataPreBegin = time.time()
    trainData,trainDataIndices = DataProcess(data)
    print("[+]Data Preprocess:" + str(round(time.time() - dataPreBegin,4)))

    labelPreBegin = time.time()
    trainLabel = LabelProcess(data)
    print("[+]Label Preprocess:" + str(round(time.time() - labelPreBegin,4)))

    data = pd.read_csv(path_test)
    testDataBegin = time.time()
    testData,testDataIndices = DataProcess(data)
    print("[+]TestData:" + str(round(time.time() - testDataBegin,4)))

    modelBegin = time.time()

    scaler = StandardScaler()
    scaler.fit(testData)
    trainData = scaler.transform(trainData)
    testData = scaler.transform(testData)

    #model = ensemble.RandomForestRegressor(n_estimators=22, n_jobs=2)
    model = MLPRegressor(hidden_layer_sizes=(128,256,128,16),learning_rate_init=0.001,max_iter=8000)
    model.fit(trainData, trainLabel)
    predict = model.predict(testData)
    print("[+]Model:" + str(round(time.time() - modelBegin,4)))

    outBegin = time.time()
    for i in range(testData.shape[0]):
        user = testDataIndices.iloc[i,0]
        if output[user] < predict[i]:
            output[user] = predict[i]

    with(open(os.path.join(path_test_out, "test.csv"), mode="w")) as outer:
        writer = csv.writer(outer)
        writer.writerow(["Id", "Pred"])
        for id,pred in output.items():
            #pred = 0 if pred < 0.5 else pred
            writer.writerow([id, pred])
    print("[+]Output:" + str(round(time.time() - outBegin,4)))


if __name__ == "__main__":
    print("****************** start **********************")
    # 程序入口
    a = time.time()
    process()
    print("[+]total:"+str(round(time.time()-a,4)))
