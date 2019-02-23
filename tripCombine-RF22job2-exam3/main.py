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
import gc

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
    return lowSpeedRate

def get_low_speed_time(data):
    lowSpeedData = (data.loc[data["SPEED"]<10].shape[0])*60
    return lowSpeedData

def get_SB_num(data):
    num = data.loc[data["SPEED"].abs()>6].shape[0]
    return num

def get_tripLong(data):
    tripLong = np.sum(np.sqrt(np.square(data["LONGITUDE"]) + np.square(data["LATITUDE"]))) * 100
    return tripLong


def LabelProcess(data):
    #trainLabel = np.array(data.drop_duplicates(["TERMINALNO","TRIP_ID"])["Y"])
    trainLabel = np.array(data.drop_duplicates(["TERMINALNO","TRIP_ID"])["Y"])
    #print(trainLabel.shape)
    return trainLabel

def DataProcess(data):
    tripSpeedMean = data[data["SPEED"]>=0]['SPEED'].mean()
    directionMean = data[data["DIRECTION"]>=0]['DIRECTION'].mean()

    data['SPEED'].replace(-1, tripSpeedMean, inplace=True)
    data['DIRECTION'].replace(-1, directionMean, inplace=True)  # 异常值赋平均
    #del(tripSpeedMean)
    #del(directionMean)
    #gc.collect()

    tripData = data.groupby(["TERMINALNO","TRIP_ID"])
    tripDataIndices = pd.DataFrame(list(tripData.groups.keys()))
    #TERandTRIPID = data[["TERMINALNO","TRIP_ID"]]
    #tripDiffData = tripData["SPEED"].diff()
    #del(data)
    #gc.collect()

    #tripDiffData = pd.concat([data[["TERMINALNO","TRIP_ID"]], tripDiffData],axis=1)
    #tripDiffData = tripDiffData.groupby(["TERMINALNO","TRIP_ID"])
    #del(TERandTRIPID)
    #gc.collect()

    TimeCost = tripData["TIME"].max()-tripData["TIME"].min()
    Time = tripData["TIME"].mean()
    Time = Time.apply(get_hour)
    enc = OneHotEncoder()
    enc.fit([[0],[1],[2],[3],[4]])
    Time = pd.DataFrame(enc.transform(pd.DataFrame(np.array(Time))).toarray(),dtype="int")

    speedMax = tripData["SPEED"].max()
    speedMean = tripData["SPEED"].mean()
    speedMin = tripData["SPEED"].mean()
    speedStd = tripData["SPEED"].std()
    speedStdMean = speedStd.mean()
    speedStd = speedStd.fillna(speedStdMean)
    #diffSpeedMax = tripDiffData["SPEED"].max().fillna(0)
    #diffSpeedMin = tripDiffData["SPEED"].min().fillna(0)
    #diffSpeedMean = tripDiffData["SPEED"].mean().fillna(0)
    #diffSpeedStd = tripDiffData["SPEED"].std().fillna(0)
    #lowSpeedRate = tripData["SPEED"].apply(speed_analysis)
    lowSpeedTime = tripData.apply(get_low_speed_time)
    lowSpeedTime = lowSpeedTime / (TimeCost+60)
    #suddenBrake = tripDiffData.apply(get_SB_num)

    #tripLong = tripData["LONGITUDE","LATITUDE"].apply(get_tripLong)
    #tripLong = tripDiffData.apply(get_tripLong)

    LongitudeMax = tripData["LONGITUDE"].max()
    LongitudeMin = tripData["LONGITUDE"].min()
    LongitudeStd = tripData["LONGITUDE"].std()
    LongitudeStdMean = LongitudeStd.mean()
    LongitudeStd = LongitudeStd.fillna(LongitudeStdMean)
    #del(LongitudeStdMean)
    #gc.collect()

    LatitudeMax = tripData["LATITUDE"].max()
    LatitudeMin = tripData["LATITUDE"].min()
    LatitudeStd = tripData["LATITUDE"].std()
    LatitudeStdMean = LatitudeStd.mean()
    LatitudeStd = LatitudeStd.fillna(LatitudeStdMean)
    #del(LatitudeStdMean)
    #gc.collect()

    centerLongitude = (LongitudeMax + LongitudeMin) / float(2)
    centerLatitude = (LatitudeMax + LatitudeMin) / float(2)

    heightMax = tripData["HEIGHT"].max()
    heightMin = tripData["HEIGHT"].min()
    heightMean = tripData["HEIGHT"].mean()
    heightStd = tripData["HEIGHT"].std()
    heightStdMean = heightStd.mean()
    heightStd = heightStd.fillna(heightStdMean)
    #del (heightStdMean)
    #gc.collect()
    #diffHeightMax = tripDiffData["HEIGHT"].max().fillna(0)
    #diffHeightMin = tripDiffData["HEIGHT"].min().fillna(0)
    #diffHeightMean = tripDiffData["HEIGHT"].mean().fillna(0)
    #diffHeightStd = tripDiffData["HEIGHT"].std().fillna(0)

    DirMean = tripData["DIRECTION"].mean()
    DirMax = tripData["DIRECTION"].max()
    DirMin = tripData["DIRECTION"].min()
    DirStd = tripData["DIRECTION"].std()
    DirStdMean = DirStd.mean()
    DirStd = DirStd.fillna(DirStdMean)

    #diffDirMax = tripDiffData["DIRECTION"].max().fillna(0)
    #diffDirMin = tripDiffData["DIRECTION"].min().fillna(0)
    #diffDirMean = tripDiffData["DIRECTION"].mean().fillna(0)
    #diffDirStd = tripDiffData["DIRECTION"].std().fillna(0)

    callstate = tripData["CALLSTATE"].min() == 4

    trainData = pd.concat([TimeCost,lowSpeedTime,speedMax,speedMean,speedMin,speedStd,#diffSpeedMean,
                           #diffSpeedMax,diffSpeedMin,diffSpeedStd,diffDirMax,diffDirMean,
                           #diffDirMin,diffDirStd,diffHeightMax,diffHeightMean,diffHeightMin,diffHeightStd,
                           LongitudeMax,#suddenBrake,tripLong,lowSpeedRate,
                           LongitudeMin,LongitudeStd,LatitudeMax,LatitudeMin,LatitudeStd,
                           DirMax,DirMean,DirMin,DirStd,
                           centerLongitude,centerLatitude,heightMean,heightMax,heightMin,heightStd,callstate], axis=1)

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
    gc.collect()
    print("[+]Data Preprocess:" + str(round(time.time() - dataPreBegin,4)))

    labelPreBegin = time.time()
    trainLabel = LabelProcess(data)
    print("[+]Label Preprocess:" + str(round(time.time() - labelPreBegin,4)))

    data = pd.read_csv(path_test)
    testDataBegin = time.time()
    testData,testDataIndices = DataProcess(data)
    gc.collect()
    print("[+]TestData:" + str(round(time.time() - testDataBegin,4)))

    modelBegin = time.time()

    scaler = StandardScaler()
    scaler.fit(testData)
    trainData = scaler.transform(trainData)
    testData = scaler.transform(testData)

    model = ensemble.RandomForestRegressor(n_estimators=22, n_jobs=2)
    #model = MLPRegressor(hidden_layer_sizes=(23,),learning_rate_init=0.001,max_iter=10000)
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
