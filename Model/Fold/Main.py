from sklearn.model_selection import ParameterGrid
import os
import pandas
import PIL.Image as pil
import numpy, datetime
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
strStrategy   = "Fold/"
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
def CustomEval(true, pred):
    dictCustomEval = {}
    arrayConfuse     = confusion_matrix(dataValidResult, arrayValidPrediction)
    floatSensitivity = arrayConfuse[1,1]/sum(arrayConfuse[1,:])
    floatSpecificity = arrayConfuse[0,0]/sum(arrayConfuse[0,:])
    floatPrecision   = arrayConfuse[1,1]/sum(arrayConfuse[:,1])
    dictCustomEval["Sensitivity"] = floatSensitivity
    dictCustomEval["Specificity"] = floatSpecificity
    dictCustomEval["Precision"]   = floatPrecision
    return(dictCustomEval)
def Fold():
    dictFold = {}
    dictFold["Index"] = []
    dictFold["Batch"] = []
    dictFold["Epoch"] = []
    dictFold["Eta"]   = []
    dictFold["Time"]  = []
    dictFold["Optimizer"] = []
    dictFold["TrainAccuracy"] = []
    dictFold["TrainAuc"] = []
    dictFold["ValidAccuracy"] = []
    dictFold["ValidAuc"] = []
    dictFold["Sensitivity(Recall)"] = []
    dictFold["Specificity"]         = []
    dictFold["Precision(PPV)"]      = []
    return(dictFold)
def TuneResult():
    dictObject = {}
    dictObject["Time"]  = []
    ##
    ##  Valid
    dictObject["ValidAccuracyMean"] = []
    dictObject["ValidAccuracySd"]   = []
    dictObject["ValidAucMean"] = []
    dictObject["ValidAucSd"]   = []
    ##
    ##  Custom
    dictObject["Sensitivity(Recall)Mean"] = []
    dictObject["SpecificityMean"]         = []
    dictObject["Precision(PPV)Mean"]      = []
    dictObject["Sensitivity(Recall)Sd"] = []
    dictObject["SpecificitySd"]         = []
    dictObject["Precision(PPV)Sd"]      = []
    return(dictObject)
################################################################################
listVariable = ['mole_size_no', 'mole_size_yes',
                'period_1個月內', 'period_1個月～1年', 'period_1年以上', 'period_不記得',
                'change_1month_不記得', 'change_1month_有變化', 'change_1month_無變化',
                'gender_不想回答', 'gender_女性', 'gender_男性',
                'age_21~40歲', 'age_21歲以下','age_40~65歲', 'age_65歲以上']
tupleResize = (32, 32)
listFoldSize = list(range(3))
dictParameter = {"Batch":32, "Epoch":200, "Eta":0.00001, "Optimizer":"Adam"}
strResultPath  = strOutputPath + str.split(str(timeit.default_timer()), ".")[1] + "/"
os.makedirs(strResultPath)
################################################################################
dictFold = Fold()
for intFoldSize in listFoldSize:
    ##
    ##  Train
    dataTrain           = pandas.read_csv("Fold/" + str(intFoldSize) + "/Train/Table.csv")
    dataTrainResult     = dataTrain["result"]
    arrayTrainVariable  = numpy.array(dataTrain[listVariable])
    arrayTrainResult    = to_categorical(dataTrainResult)
    listTrainImage = []
    for index, data in dataTrain.iterrows():
        strResult = str(data["result"])
        strImage = str(data["id"]) + ".jpg"
        objectImage = pil.open("Fold/" + str(intFoldSize) + "/Train/Image/" + strResult + "/" + strImage).resize(tupleResize)
        arrayImage  = numpy.array(objectImage) / 255
        listTrainImage.append(arrayImage)
    arrayTrainImage = numpy.array(listTrainImage).astype("float32")
    ##
    ##  Valid
    dataValid           = pandas.read_csv("Fold/" + str(intFoldSize) + "/Valid/Table.csv")
    dataValidResult     = dataValid["result"]
    arrayValidVariable  = numpy.array(dataValid[listVariable])
    arrayValidResult    = to_categorical(dataValidResult)
    listValidImage = []
    for index, data in dataValid.iterrows():
        strResult = str(data["result"])
        strImage = str(data["id"]) + ".jpg"
        objectImage = pil.open("Fold/" + str(intFoldSize) + "/Valid/Image/" + strResult + "/" + strImage).resize(tupleResize)
        arrayImage  = numpy.array(objectImage) / 255
        listValidImage.append(arrayImage)
    arrayValidImage = numpy.array(listValidImage).astype("float32")
    ##
    ##  Initial
    floatStart = timeit.default_timer()
    backend.get_session()
    numpy.random.seed(2018)
    tensorflow.set_random_seed(2018)
    objectModel = CustomResNet()
    if(dictParameter["Optimizer"]=="Adadelta"):
        objectOptimizer = Adadelta
    if(dictParameter["Optimizer"]=="Adam"):
        objectOptimizer = Adam
    if(dictParameter["Optimizer"]=="SGD"):
        objectOptimizer = SGD
    objectModel.compile(loss=logloss,optimizer=objectOptimizer(lr=dictParameter["Eta"]), metrics=["acc"])
    objectStop = EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto', restore_best_weights=True)
    ##
    ##  Fit
    objectModel.fit(
        [arrayTrainImage, arrayTrainVariable], arrayTrainResult,
        batch_size=dictParameter["Batch"],
        epochs=dictParameter["Epoch"],
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
    ##  Cumstom evaluation
    dictCustomEval = CustomEval(dataValidResult, arrayValidPrediction)
    ##
    ##  Summary
    dictFold["Index"].append(intFoldSize)
    dictFold["Batch"].append(dictParameter["Batch"])
    dictFold["Epoch"].append(max(objectModel.history.epoch))
    dictFold["Eta"].append(dictParameter["Eta"])
    dictFold["Time"].append(floatEnd - floatStart)
    dictFold["Optimizer"].append(dictParameter["Optimizer"])
    dictFold["TrainAccuracy"].append(floatTrainAccuracy)
    dictFold["TrainAuc"].append(floatTrainAuc)
    dictFold["ValidAccuracy"].append(floatValidAccuracy)
    dictFold["ValidAuc"].append(floatValidAuc)
    dictFold["Sensitivity(Recall)"].append(dictCustomEval["Sensitivity"])
    dictFold["Specificity"].append(dictCustomEval["Specificity"])
    dictFold["Precision(PPV)"].append(dictCustomEval["Precision"])
    pass
dataFold = pandas.DataFrame(dictFold)
################################################################################
##
##  Summary
dictTuneResult = TuneResult()
dictTuneResult['Time'].append(numpy.sum(dataFold["Time"]))
dictTuneResult['ValidAccuracyMean'].append(numpy.mean(dataFold["ValidAccuracy"]))
dictTuneResult['ValidAccuracySd'].append(numpy.std(dataFold["ValidAccuracy"]))
dictTuneResult['ValidAucMean'].append(numpy.mean(dataFold["ValidAuc"]))
dictTuneResult['ValidAucSd'].append(numpy.std(dataFold["ValidAuc"]))
dictTuneResult['Sensitivity(Recall)Mean'].append(numpy.mean(dataFold["Sensitivity(Recall)"]))
dictTuneResult['SpecificityMean'].append(numpy.mean(dataFold["Specificity"]))
dictTuneResult['Precision(PPV)Mean'].append(numpy.mean(dataFold["Precision(PPV)"]))
dictTuneResult['Sensitivity(Recall)Sd'].append(numpy.std(dataFold["Sensitivity(Recall)"]))
dictTuneResult['SpecificitySd'].append(numpy.std(dataFold["Specificity"]))
dictTuneResult['Precision(PPV)Sd'].append(numpy.std(dataFold["Precision(PPV)"]))
pandas.DataFrame(dictTuneResult)
