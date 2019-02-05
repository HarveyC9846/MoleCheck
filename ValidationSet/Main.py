import os
import pandas
import PIL.Image as pil
import numpy
import shutil
from sklearn.model_selection import train_test_split as holdout
from sklearn.model_selection import StratifiedKFold as fold
from keras.preprocessing.image import ImageDataGenerator as idg
strOutputPath = "Output/"
try:
    os.makedirs(strOutputPath)
except:
    shutil.rmtree(strOutputPath)
    os.makedirs(strOutputPath)
################################################################################
##
## 1. Holdout
os.makedirs(strOutputPath + "Holdout/Train/", exist_ok=True)
os.makedirs(strOutputPath + "Holdout/Valid/", exist_ok=True)
dataTable = pandas.read_csv("Table.csv")
dataTrain, dataValid = holdout(dataTable, test_size = 0.3, stratify = dataTable["result"])
##
##  Train
dataTrain.to_csv(strOutputPath + "Holdout/Train/Table.csv", index =False)
for index, data in dataTrain.iterrows():
    ##
    ##  Type image1
    file   = data.image
    result = str(data.result)
    image = pil.open("Image/" + result + "/" + file)
    strImagePath = strOutputPath + "Holdout/Train/Image/" + result + "/"
    os.makedirs(strImagePath, exist_ok=True)
    image.save(strImagePath + file)
    pass
##
##  Valid
dataValid.to_csv(strOutputPath + "Holdout/Valid/Table.csv", index =False)
for index, data in dataValid.iterrows():
    ##
    ##  Type image1
    file   = data.image
    result = str(data.result)
    image = pil.open("Image/" + result + "/" + file)
    StrImagePath = strOutputPath + "Holdout/Valid/Image/" + result + "/"
    os.makedirs(StrImagePath, exist_ok=True)
    image.save(StrImagePath + file)
    pass
################################################################################
## 2. Fold
os.makedirs(strOutputPath + "Fold/", exist_ok=True)
dataTable  = pandas.read_csv("Table.csv")
objectFold = fold(n_splits=3).split(dataTable, dataTable["result"])
for intFoldIndex, (listTrainIndex, listValidIndex) in enumerate(objectFold):
    ##
    ##  Fold path
    strFoldPath = strOutputPath + "Fold/" + str(intFoldIndex) + "/"
    ##
    ##  Train
    strTrainPath = strFoldPath + "Train/"
    os.makedirs(strTrainPath, exist_ok=True)
    dataTrain = dataTable.iloc[listTrainIndex]
    dataTrain.to_csv(strTrainPath + "Table.csv" , index =False)
    for index, data in dataTrain.iterrows():
        file   = data.image
        result = str(data.result)
        image = pil.open("Image/" + result + "/" + file)
        strImagePath = strTrainPath + "Image/" + result + "/"
        os.makedirs(strImagePath, exist_ok=True)
        image.save(strImagePath + file)
        pass
    ##
    ##  Valid
    strValidPath = strFoldPath + "Valid/"
    os.makedirs(strValidPath, exist_ok=True)
    dataValid = dataTable.iloc[listValidIndex]
    dataValid.to_csv(strValidPath + "Table.csv" , index =False)
    for index, data in dataValid.iterrows():
        file   = data.image
        result = str(data.result)
        image = pil.open("Image/" + result + "/" + file)
        strImagePath = strValidPath + "Image/" + result + "/"
        os.makedirs(strImagePath, exist_ok=True)
        image.save(strImagePath + file)
        pass
    pass
