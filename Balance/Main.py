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
##  1. Balance holdout
##
##  Train data
dataTrain = pandas.read_csv("Holdout/Train/Table.csv")
intTrainBalanceSize = 100000
listClass = list(dataTrain["result"].unique())
listFakeTrain = []
for i in listClass:
    numpy.random.seed(2018)
    dataFake = dataTrain.loc[dataTrain["result"]==i].sample(intTrainBalanceSize, replace =True)
    listFakeTrain.append(dataFake)
dataFakeTrain       = pandas.concat(listFakeTrain)
dataFakeTrain["id"] = range(dataFakeTrain.shape[0])
strTrainPath = strOutputPath + "Holdout/" + "Train/"
os.makedirs(strTrainPath, exist_ok = True)
dataFakeTrain.to_csv(strTrainPath + "Table.csv", index = False)
for index, data in dataFakeTrain.iterrows():
    file = data["image"]
    result = str(data["result"])
    image = pil.open("Holdout/Train/Image/" + result + '/' + file)
    ##
    ##  Image generator
    generator = idg(rotation_range = 360, horizontal_flip=True, vertical_flip=True)
    ##
    ##  Old
    old = numpy.array(image)
    old = old.reshape((1,) + old.shape)
    ##
    ##  New
    new = generator.flow(old).next()
    new = new[0,:,:,:].astype("uint8")
    new = pil.fromarray(new)
    strImagePath = strOutputPath + "Holdout/Train/Image/" + result + "/"
    os.makedirs(strImagePath, exist_ok = True)
    new.save(strImagePath + str(data["id"]) + ".jpg")
##
##  Valid data
dataValid = pandas.read_csv("Holdout/Valid/Table.csv")
intValidBalanceSize = 10000
listClass = list(dataValid["result"].unique())
listFakeValid = []
for i in listClass:
    numpy.random.seed(2018)
    dataFake = dataValid.loc[dataValid["result"]==i].sample(intValidBalanceSize, replace =True)
    listFakeValid.append(dataFake)
dataFakeValid       = pandas.concat(listFakeValid)
dataFakeValid["id"] = range(dataFakeValid.shape[0])
strValidPath = strOutputPath + "Holdout/" + "Valid/"
os.makedirs(strValidPath, exist_ok = True)
dataFakeValid.to_csv(strValidPath + "Table.csv", index = False)
for index, data in dataFakeValid.iterrows():
    file = data["image"]
    result = str(data["result"])
    image = pil.open("Holdout/Valid/Image/" + result + '/' + file)
    ##
    ##  Image generator
    generator = idg(rotation_range = 360, horizontal_flip=True, vertical_flip=True)
    ##
    ##  Old
    old = numpy.array(image)
    old = old.reshape((1,) + old.shape)
    ##
    ##  New
    new = generator.flow(old).next()
    new = new[0,:,:,:].astype("uint8")
    new = pil.fromarray(new)
    strImagePath = strOutputPath + "Holdout/Valid/Image/" + result + "/"
    os.makedirs(strImagePath, exist_ok = True)
    new.save(strImagePath + str(data["id"]) + ".jpg")
################################################################################
##
##  1. Balance fold
listFold = os.listdir("Fold/")
for strFold in listFold:
    ##
    ##  Train data
    dataTrain = pandas.read_csv("Fold/" + strFold + "/Train/Table.csv")
    intTrainBalanceSize = 100000
    listClass = list(dataTrain["result"].unique())
    listFakeTrain = []
    for i in listClass:
        numpy.random.seed(2018)
        dataFake = dataTrain.loc[dataTrain["result"]==i].sample(intTrainBalanceSize, replace =True)
        listFakeTrain.append(dataFake)
    dataFakeTrain       = pandas.concat(listFakeTrain)
    dataFakeTrain["id"] = range(dataFakeTrain.shape[0])
    strTrainPath = strOutputPath + "Fold/" + strFold + "/Train/"
    os.makedirs(strTrainPath, exist_ok = True)
    dataFakeTrain.to_csv(strTrainPath + "Table.csv", index = False)
    for index, data in dataFakeTrain.iterrows():
        file = data["image"]
        result = str(data["result"])
        image = pil.open("Fold/" + strFold + "/Train/Image/" + result + '/' + file)
        ##
        ##  Image generator
        generator = idg(rotation_range = 360, horizontal_flip=True, vertical_flip=True)
        ##
        ##  Old
        old = numpy.array(image)
        old = old.reshape((1,) + old.shape)
        ##
        ##  New
        new = generator.flow(old).next()
        new = new[0,:,:,:].astype("uint8")
        new = pil.fromarray(new)
        strImagePath = strOutputPath + "Fold/" + strFold + "/Train/Image/" + result + "/"
        os.makedirs(strImagePath, exist_ok = True)
        new.save(strImagePath + str(data["id"]) + ".jpg")
    ##
    ##  Valid data
    dataValid = pandas.read_csv("Fold/" + strFold + "/Valid/Table.csv")
    intValidBalanceSize = 10000
    listClass = list(dataValid["result"].unique())
    listFakeValid = []
    for i in listClass:
        numpy.random.seed(2018)
        dataFake = dataValid.loc[dataValid["result"]==i].sample(intValidBalanceSize, replace =True)
        listFakeValid.append(dataFake)
    dataFakeValid       = pandas.concat(listFakeValid)
    dataFakeValid["id"] = range(dataFakeValid.shape[0])
    strValidPath = strOutputPath + "Fold/" + strFold + "/Valid/"
    os.makedirs(strValidPath, exist_ok = True)
    dataFakeValid.to_csv(strValidPath + "Table.csv", index = False)
    for index, data in dataFakeValid.iterrows():
        file = data["image"]
        result = str(data["result"])
        image = pil.open("Fold/" + strFold + "/Valid/Image/" + result + '/' + file)
        ##
        ##  Image generator
        generator = idg(rotation_range = 360, horizontal_flip=True, vertical_flip=True)
        ##
        ##  Old
        old = numpy.array(image)
        old = old.reshape((1,) + old.shape)
        ##
        ##  New
        new = generator.flow(old).next()
        new = new[0,:,:,:].astype("uint8")
        new = pil.fromarray(new)
        strImagePath = strOutputPath + "Fold/" + strFold + "/Valid/Image/" + result + "/"
        os.makedirs(strImagePath, exist_ok = True)
        new.save(strImagePath + str(data["id"]) + ".jpg")
