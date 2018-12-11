################################################################################
##
##
##  Load
from sklearn.model_selection import ParameterGrid
import os
import pandas
import PIL
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
import torch
import torch.utils.data
try:
    os.chdir(".\\TuneParameter\\")
except:
    pass
##
##
##  Define model
from torch.nn import *
from torch.optim import *
class DefineModel(Module):
    def __init__(self):
        super(DefineModel, self).__init__()
        self.The1stConv  = Conv2d(in_channels= 3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1)
        self.The2ndConv  = Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1)
        self.The1stFully = Linear(in_features=16+(64*64*64), out_features=1000, bias=True)
        self.The2ndFully = Linear(in_features=    1000, out_features=  2, bias=True)
        self.Softmax     = Softmax(dim=1)
        self.ReLu        = ReLU()
        self.sigmoid     = Sigmoid()
    def forward(self, ImageTerm, VariableTerm):
        Pip = self.The1stConv(ImageTerm)
        Pip = self.ReLu(Pip)
        Pip = self.The2ndConv(Pip)
        Pip = self.ReLu(Pip)
        Pip = Pip.view(ImageTerm.size()[0], -1)
        Pip = torch.cat((Pip, VariableTerm), 1)
        Pip = self.The1stFully(Pip)
        Pip = self.sigmoid(Pip)
        Pip = self.The2ndFully(Pip)
        Output = self.Softmax(Pip)
        return(Output)
################################################################################
##
##
##  Variable
Variable = ["OverSize_N", "OverSize_Y", "Period_Month",
 "Period_OverYear", "Period_Unknown", "Period_Year", "Change_N","Change_Unknown",
 "Change_Y", "Gender_F", "Gender_M","Gender_N","Age_Middle",
 "Age_Old","Age_Teen","Age_Youth"]
##
##
##  Resize
Resize = (64, 64)
##
##
##  Train
TrainTable = pandas.read_csv(".\\Train\\Table.csv" )
TrainId    = [i for i in TrainTable.Index]
TrainLabel =  TrainTable.Label
TrainImage = []
for i, d in TrainTable.iterrows():
    iLabel = str(d.Label)
    iImage = PIL.Image.open(".\\Train\\Image\\" + iLabel + "\\" + str(d.Index) + ".jpg")
    iImage = iImage.resize(Resize)
    iImage = numpy.array(iImage)
    iImage = iImage.astype("float32")
    iImage = iImage / 255
    TrainImage.append(iImage)
TrainImage     = numpy.array(TrainImage)
TrainVariable  = TrainTable[Variable]
##
##
##  Valid
ValidTable = pandas.read_csv(".\\Valid\\Table.csv" )
ValidId    = [i for i in ValidTable.Index]
ValidLabel =  ValidTable.Label
ValidImage = []
for i, d in ValidTable.iterrows():
    iLabel = str(d.Label)
    iImage = PIL.Image.open(".\\Valid\\Image\\" + iLabel + "\\" + str(d.Index) + ".jpg")
    iImage = iImage.resize(Resize)
    iImage = numpy.array(iImage)
    iImage = iImage.astype("float32")
    iImage = iImage / 255
    ValidImage.append(iImage)
ValidImage = numpy.array(ValidImage)
ValidVariable = ValidTable[Variable]
##
##
##  Pytorch
TrainImageTerm = torch.from_numpy(TrainImage.reshape((TrainImage.shape[0], TrainImage.shape[3]) + TrainImage.shape[1:3])).type(torch.FloatTensor)
ValidImageTerm = torch.from_numpy(ValidImage.reshape((ValidImage.shape[0], ValidImage.shape[3]) + ValidImage.shape[1:3])).type(torch.FloatTensor)
TrainVariableTerm = torch.from_numpy(numpy.array(TrainVariable)).type(torch.FloatTensor)
ValidVariableTerm = torch.from_numpy(numpy.array(ValidVariable)).type(torch.FloatTensor)
TrainLabelCode = torch.from_numpy(numpy.array(TrainLabel)).type("torch.LongTensor").view(-1)
ValidLabelCode = torch.from_numpy(numpy.array(ValidLabel)).type("torch.LongTensor").view(-1)
TrainSet = torch.utils.data.TensorDataset(TrainImageTerm, TrainVariableTerm, TrainLabelCode)
ValidSet = torch.utils.data.TensorDataset(ValidImageTerm, ValidVariableTerm, ValidLabelCode)
################################################################################
##
##
##  Parameter
Parameter    = {"Batch" : [32, 64], "Optimizer": ["Adadelta", "Adam"], "LearnRate": [0.001, 0.0001], "Epoch": [20, 40]}
ParameterSet = ParameterGrid(Parameter)
for p in ParameterSet:
    break
    ##
    ##
    ##  Load model
    Model = DefineModel()
    ##
    ##
    ##  Optimizer
    if( p["Optimizer"] == 'Adadelta' ):
        Optimizer = torch.optim.Adadelta(Model.parameters(), lr = p["LearnRate"])
    ##
    ##
    ##  Loss
    LossFunction = torch.nn.CrossEntropyLoss()
    ##
    ##
    ##  Epoch
    ValidEpochLoss     = []
    ValidEpochAccuracy = []
    for EpochIndex in range(p["Epoch"]):
        TrainBatchSet = torch.utils.data.DataLoader(TrainSet, batch_size=p["Batch"], shuffle=True)
        ValidBatchSet = torch.utils.data.DataLoader(ValidSet, batch_size=p["Batch"], shuffle=True)
        for OneTrainBatchIndex, OneTrainBatch in enumerate(TrainBatchSet):
            ##
            ##
            ##  Inital gradient
            Optimizer.zero_grad()
            OneTrainBatchScore = Model.cuda()(OneTrainBatch[0].cuda(), OneTrainBatch[1].cuda())
            OneTrainBatchLoss  = LossFunction(OneTrainBatchScore, OneTrainBatch[2].cuda())
            ##
            ##
            ##  Update gradient
            OneTrainBatchLoss.backward()
            Optimizer.step()
        ##
        ##
        ##  Check valid data
        with torch.no_grad():
            ValidNumber        = 0
            ValidAccurateCount = 0
            ValidTotalLoss     = 0
            for OneValidBatchIndex, OneValidBatch in enumerate(ValidBatchSet):
                OneValidBatchNumber          = OneValidBatch[0].size()[0]
                OneValidBatchScore           = Model.cuda()(OneValidBatch[0].cuda(), OneValidBatch[1].cuda())
                OneValidBatchLoss            = LossFunction(OneValidBatchScore, OneValidBatch[2].cuda())
                _, OneValidBatchPrediction   = torch.max(OneValidBatchScore, 1)
                OneValidBatchAccuracy        = accuracy_score(OneValidBatch[2], numpy.array(OneValidBatchPrediction))
                ValidAccurateCount           = ValidAccurateCount + (OneValidBatchAccuracy * OneValidBatchNumber)
                ValidTotalLoss               = ValidTotalLoss     + (OneValidBatchNumber * OneValidBatchLoss)
                ValidNumber                  = ValidNumber        + OneValidBatchNumber
            ValidEpochLoss.append(float(ValidTotalLoss) / ValidNumber)
            ValidEpochAccuracy.append(ValidAccurateCount / ValidNumber)
ValidEpochLoss
ValidEpochAccuracy
################################################################################
##
##
##  Epoch
# Epoch = 20
# for EpochIndex in range(Epoch):
#     TrainBatchSet = torch.utils.data.DataLoader(TrainSet, batch_size=32, shuffle=True)
#     ##
#     ##
#     ##  Batch total
#     BatchTotal = {"Size" : 0, "Loss" : 0, "AccurateCount" : 0}
#     for OneTrainBatchIndex, OneTrainBatch in enumerate(TrainBatchSet):
#         ##
#         ##
#         ##  Inital gradient
#         Optimizer.zero_grad()
#         OneTrainBatchScore = Model.cuda()(OneTrainBatch[0].cuda(), OneTrainBatch[1].cuda())
#         OneTrainBatchLoss  = LossFunction(OneTrainBatchScore, OneTrainBatch[2].cuda())
#         ##
#         ##
#         ##  Update gradient
#         OneTrainBatchLoss.backward()
#         Optimizer.step()
#         ##
#         ##
#         ##  Accuracy
#         _, OnePrediction   = torch.max(OneTrainBatchScore, 1)
#         OneAccuracy        = accuracy_score(OneTrainBatch[2], numpy.array(OnePrediction))
#         BatchTotal["Size"] = BatchTotal["Size"] + OneTrainBatch[0].size()[0]
#         BatchTotal["Loss"] = BatchTotal["Loss"] + float((OneTrainBatchLoss * OneTrainBatch[0].size()[0]))
#         BatchTotal["AccurateCount"] = BatchTotal["AccurateCount"] + OneAccuracy * OneTrainBatch[0].size()[0]
#         # print("Epoch:", EpochIndex,  "Loss:", float(OneTrainBatchLoss), "Accuracy:", float(OneAccuracy), end='\r')
#     ##
#     ##
#     ##  Epoch log
#     EpochTrainLoss = BatchTotal["Loss"]/BatchTotal["Size"]
#     EpochTrainAccuracy = BatchTotal["AccurateCount"]/BatchTotal["Size"]
#     print("")
#     print("TrainLoss: %s, TrainAccuracy: %s" % (EpochTrainLoss,EpochTrainAccuracy))
#
# ################################################################################
# ValidBatchSet = torch.utils.data.DataLoader(ValidSet, batch_size=5000, shuffle=False)
#
# with torch.no_grad():
#
#     Model.cuda()(ValidSet[0].cuda(), ValidSet[1].cuda())
#
#
# ValidSet[0]
