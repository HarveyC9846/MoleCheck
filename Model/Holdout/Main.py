# -*- coding: UTF-8 -*-
from sklearn.model_selection import ParameterGrid
import os
import pandas
import PIL.Image as pil
import numpy, datetime
import random
import keras
import tensorflow
import pdb
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix
from keras import backend
import sys
import re
import datetime
from skimage.feature import hog
from keras.models import load_model
import scikitplot
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import matplotlib.pyplot as plot
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from keras import backend
from keras.losses import categorical_crossentropy as logloss
import shutil
from keras.optimizers import Adadelta, Adam, SGD
from keras.applications.resnet50 import ResNet50 as CreateResNet
import timeit
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
strStrategy   = "Holdout/"
strOutputPath = strStrategy + "Output/"
os.makedirs(strOutputPath, exist_ok = True)
################################################################################
def CustomResNet():
    ResNet = CreateResNet(include_top=True, weights=None, input_tensor=None,
                          input_shape=(32,32,3), pooling=None, classes=2)
    ResNet.layers.pop()
    ImageInput  = ResNet.input
    VariableInput = Input(shape=(16,))
    ModelNet = keras.layers.concatenate([ResNet.output, VariableInput])
    ModelNet = Dense(256, activation ='tanh', kernel_regularizer=regularizers.l2(0.01))(ModelNet)
    PredictOutput = Dense(2, activation ='softmax')(ModelNet)
    TheModel = Model(inputs=[ImageInput, VariableInput], outputs=PredictOutput)
    return(TheModel)
def TuneResult():
    ##
    ##  Parameter
    dictObject = {}
    dictObject["Index"] = []
    dictObject["Batch"] = []
    dictObject["Epoch"] = []
    dictObject["Eta"]   = []
    dictObject["Time"]  = []
    dictObject["Optimizer"] = []
    ##
    ##  Train
    dictObject["TrainAccuracy"]            = []
    dictObject["TrainAuc"]                 = []
    ##  Valid
    dictObject["ValidAccuracy"]            = []
    dictObject["ValidAuc"]                 = []
    ##
    ##  Custom
    dictObject["Threshold"] = []
    dictObject["Accuracy"]  = []
    dictObject["Sensitivity(Recall)"] = []
    dictObject["Specificity"]         = []
    dictObject["Precision(PPV)"]      = []
    return(dictObject)

def CustomEval(true, pred):
    objectCustomEval = {"Threshold":[], "Accuracy":[], "Sensitivity":[], "Specificity":[], "Precision":[]}
    _, _, arrayThreshold = roc_curve(true, pred)
    for floatThreshold in arrayThreshold:
        arrayPrediction = 1 * (pred > floatThreshold)
        if(len(numpy.unique(arrayPrediction))==1):
            continue
        else:
            floatAccuracy = accuracy_score(true, arrayPrediction)
            arrayConfuse  = confusion_matrix(true, arrayPrediction)
            floatSensitivity = arrayConfuse[1,1]/sum(arrayConfuse[1,:])
            floatSpecificity = arrayConfuse[0,0]/sum(arrayConfuse[0,:])
            floatPrecision   = arrayConfuse[1,1]/sum(arrayConfuse[:,1])
            objectCustomEval["Threshold"].append(floatThreshold)
            objectCustomEval["Accuracy"].append(floatAccuracy)
            objectCustomEval["Sensitivity"].append(floatSensitivity)
            objectCustomEval["Specificity"].append(floatSpecificity)
            objectCustomEval["Precision"].append(floatPrecision)
    dataCustomEval     = pandas.DataFrame(objectCustomEval)
    dataBestCustomEval = dataCustomEval.loc[dataCustomEval["Accuracy"]==max(dataCustomEval["Accuracy"])].iloc[0]
    return(dataCustomEval, dataBestCustomEval)
################################################################################
listVariable = ['mole_size_no', 'mole_size_yes',
                'period_1個月內', 'period_1個月～1年', 'period_1年以上', 'period_不記得',
                'change_1month_不記得', 'change_1month_有變化', 'change_1month_無變化',
                'gender_不想回答', 'gender_女性', 'gender_男性',
                'age_21~40歲', 'age_21歲以下','age_40~65歲', 'age_65歲以上']
tupleResize = (32, 32)
##
##  Train
dataTrain           = pandas.read_csv("Holdout/Train/Table.csv")
dataTrainResult     = dataTrain["result"]
arrayTrainVariable  = numpy.array(dataTrain[listVariable])
arrayTrainResult    = to_categorical(dataTrainResult)
listTrainImage = []
for index, data in dataTrain.iterrows():
    strResult = str(data["result"])
    strImage = str(data["id"]) + ".jpg"
    objectImage = pil.open("Holdout/Train/Image/" + strResult + "/" + strImage).resize(tupleResize)
    arrayImage  = numpy.array(objectImage) / 255
    listTrainImage.append(arrayImage)
arrayTrainImage = numpy.array(listTrainImage).astype("float32")
##
##  Valid
dataValid           = pandas.read_csv("Holdout/Valid/Table.csv")
dataValidResult     = dataValid["result"]
arrayValidVariable  = numpy.array(dataValid[listVariable])
arrayValidResult    = to_categorical(dataValidResult)
listValidImage = []
for index, data in dataValid.iterrows():
    strResult = str(data["result"])
    strImage = str(data["id"]) + ".jpg"
    objectImage = pil.open("Holdout/Valid/Image/" + strResult + "/" + strImage).resize(tupleResize)
    arrayImage  = numpy.array(objectImage) / 255
    listValidImage.append(arrayImage)
arrayValidImage = numpy.array(listValidImage).astype("float32")
##
##  Parameter
dictParameter = {}
dictParameter["Batch"] = [64]
dictParameter["Epoch"] = [2, 2, 2]
dictParameter["Eta"] = [0.0001]
dictParameter["Optimizer"] = ["Adam"]
listParameter = list(ParameterGrid(dictParameter))
##
##  Result
dictTuneResult = TuneResult()
strResultPath  = strOutputPath + str.split(str(timeit.default_timer()), ".")[1] + "/"
os.makedirs(strResultPath)
##
##  Model
for i, p in enumerate(listParameter):
    ##
    ##  Initial
    floatStart = timeit.default_timer()
    random.seed(2)
    numpy.random.seed(2018)
    tensorflow.set_random_seed(2018)
    os.environ['PYTHONHASHSEED'] = "1"
    backend.get_session()

    objectModel = CustomResNet()
    if(p["Optimizer"]=="Adadelta"):
        objectOptimizer = Adadelta
    if(p["Optimizer"]=="Adam"):
        objectOptimizer = Adam
    if(p["Optimizer"]=="SGD"):
        objectOptimizer = SGD
    objectModel.compile(loss=logloss,optimizer=objectOptimizer(lr=p["Eta"]), metrics=["acc"])
    objectStop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto', restore_best_weights=True)
    ##
    ##  Fit
    objectModel.fit(
        [arrayTrainImage, arrayTrainVariable], arrayTrainResult,
        batch_size=p["Batch"],
        epochs=p["Epoch"],
        verbose=1,
        validation_data=([[arrayValidImage, arrayValidVariable], arrayValidResult]),
        callbacks = [objectStop]
        )
    floatEnd = timeit.default_timer()
    ##
    ##  Train result
    arrayTrainLikelihood     = objectModel.predict([arrayTrainImage, arrayTrainVariable])
    arrayTrainPrediction     = numpy.argmax(arrayTrainLikelihood, axis=1)
    floatTrainAuc            = roc_auc_score(dataTrainResult, arrayTrainLikelihood[:,1])
    floatTrainAccuracy       = accuracy_score(dataTrainResult, arrayTrainPrediction)
    ##
    ##  Valid result
    arrayValidLikelihood     = objectModel.predict([arrayValidImage, arrayValidVariable])
    arrayValidPrediction     = numpy.argmax(arrayValidLikelihood, axis=1)
    floatValidAuc            = roc_auc_score(dataValidResult, arrayValidLikelihood[:,1])
    floatValidAccuracy       = accuracy_score(dataValidResult, arrayValidPrediction)
    ##
    ##  Custom result
    dataCustomResult, dataBetterCustomResult       = CustomEval(dataValidResult, arrayValidLikelihood[:,1])
    ##
    ##  Summary
    dictTuneResult['Index'].append(i)
    dictTuneResult['Batch'].append(p['Batch'])
    dictTuneResult['Eta'].append(p['Eta'])
    dictTuneResult['Epoch'].append(max(objectModel.history.epoch))
    dictTuneResult['Time'].append(floatEnd - floatStart)
    dictTuneResult['Optimizer'].append(p['Optimizer'])
    dictTuneResult['TrainAccuracy'].append(floatTrainAccuracy)
    dictTuneResult['TrainAuc'].append(floatTrainAuc)
    dictTuneResult['ValidAccuracy'].append(floatValidAccuracy)
    dictTuneResult['ValidAuc'].append(floatValidAuc)
    dictTuneResult["Threshold"].append(dataBetterCustomResult["Threshold"].item())
    dictTuneResult["Accuracy"].append(dataBetterCustomResult["Accuracy"].item())
    dictTuneResult["Sensitivity(Recall)"].append(dataBetterCustomResult["Sensitivity"].item())
    dictTuneResult["Specificity"].append(dataBetterCustomResult["Specificity"].item())
    dictTuneResult["Precision(PPV)"].append(dataBetterCustomResult["Precision"].item())
    dataTuneResult = pandas.DataFrame(dictTuneResult)
    ##
    ##  Save
    dataTuneResult.to_csv(strResultPath + "TuneResult.csv", index=False)
    dataCustomResult.to_csv(strResultPath + str(i) + ".csv", index=False)
    objectModel.save(strResultPath + str(i) + ".h5")
