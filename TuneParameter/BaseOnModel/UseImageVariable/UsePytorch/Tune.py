################################################################################
##
##
##  Load
import os
import pandas
import PIL
import numpy, datetime
import pdb
import sys
import re
import datetime
import scikitplot
import matplotlib.pyplot as plot
import torch
import torch.utils.data
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix
from skimage.feature import hog
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from torch.nn import *
from torch.optim import *
import time
try:
    os.chdir(".\\TuneParameter\\")
except:
    pass
##
##
##  Define model
class DefineModel(Module):
    def __init__(self, ImageSize, VariableSize, Class):
        super(DefineModel, self).__init__()
        self.The1stConv  = Conv2d(in_channels= 3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1)
        self.The2ndConv  = Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1)
        self.The1stFully = Linear(in_features=VariableSize+(ImageSize[0]*ImageSize[1]*64), out_features=1000, bias=True)
        self.The2ndFully = Linear(in_features=    1000, out_features=  Class, bias=True)
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
VariableSize = len(Variable)
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
TrainLabelCodeTerm = torch.from_numpy(numpy.array(TrainLabel)).type("torch.LongTensor").view(-1)
ValidLabelCodeTerm = torch.from_numpy(numpy.array(ValidLabel)).type("torch.LongTensor").view(-1)
TrainSet = torch.utils.data.TensorDataset(TrainImageTerm, TrainVariableTerm, TrainLabelCodeTerm)
ValidSet = torch.utils.data.TensorDataset(ValidImageTerm, ValidVariableTerm, ValidLabelCodeTerm)
################################################################################
##
##
##  Tune table
TuneTable       = {"Batch":[], "Optimizer":[], "LearnRate":[], "Epoch":[], "Accuracy":[], "Auc":[], "Loss":[]}
##
##
##  Tune better model
TuneBetterModel = []
##
##
##  Tune result path
TuneResultPath = "BaseOnModel\\UseImageVariable\\UsePytorch\\Result\\" + str.split(str(time.time()), ".")[0] + "\\"
try:
    os.makedirs(TuneResultPath)
except:
    pass
################################################################################
##
##
##  Parameter
Parameter    = {"Batch" : [2, 4, 8, 32, 64], "Optimizer": ["Adam", "Adadelta"], "LearnRate": [0.001, 0.0001], "Epoch": [50]}
ParameterSet = ParameterGrid(Parameter)
for p in ParameterSet:
    ##
    ##
    ##  Deterministic result
    torch.manual_seed(60)
    torch.cuda.manual_seed(60)
    torch.backends.cudnn.deterministic = True
    ##
    ##
    ##  Load model
    Model = DefineModel(ImageSize=(64,64,3), VariableSize=16, Class=2)
    ##
    ##
    ##  Optimizer
    if( p["Optimizer"] == 'Adadelta' ):
        Optimizer = torch.optim.Adadelta(Model.parameters(), lr = p["LearnRate"])
    if( p["Optimizer"] == 'Adam' ):
        Optimizer = torch.optim.Adam(Model.parameters(), lr = p["LearnRate"])
    ##
    ##
    ##  Loss
    LossFunction = torch.nn.CrossEntropyLoss()
    ##
    ##
    ##  Epoch
    for EpochIndex in range(p["Epoch"]):
        ##
        ##
        ##  Batch
        TrainBatchSet = torch.utils.data.DataLoader(TrainSet, batch_size=p["Batch"], shuffle=True)
        ValidBatchSet = torch.utils.data.DataLoader(ValidSet, batch_size=p["Batch"], shuffle=True)
        for TrainMinBatchIndex, TrainMinBatch in enumerate(TrainBatchSet):
            ##
            ##
            ##  Inital gradient
            Optimizer.zero_grad()
            TrainMinBatchScore = Model.cuda()(TrainMinBatch[0].cuda(), TrainMinBatch[1].cuda())
            TrainMinBatchLoss  = LossFunction(TrainMinBatchScore, TrainMinBatch[2].cuda())
            ##
            ##
            ##  Update gradient
            TrainMinBatchLoss.backward()
            Optimizer.step()
            ##
            ##
            ##  End min batch
            pass
        ##
        ##
        ##  End epoch
        pass
    ##
    ##
    ##  Check valid data when finish all epoch
    with torch.no_grad():
        ValidNumber        = 0
        ValidAnswer        = []
        ValidPredictScore  = []
        for ValidMinBatchIndex, ValidMinBatch in enumerate(ValidBatchSet):
            ValidMinBatchScore  = Model.cuda()(ValidMinBatch[0].cuda(), ValidMinBatch[1].cuda())
            ValidNumber         = ValidNumber + ValidMinBatch[0].size()[0]
            ValidAnswer.append(ValidMinBatch[2])
            ValidPredictScore.append(ValidMinBatchScore)
    ##
    ##
    ##  Complete valid data check
    ValidAnswer          = torch.cat(ValidAnswer, dim=0)
    ValidPredictScore    = torch.cat(ValidPredictScore, dim=0)
    ValidLoss            = LossFunction(ValidPredictScore.cpu(), ValidAnswer.cpu())
    _, ValidPredictClass = torch.max(ValidPredictScore,1)
    ##
    ##
    ##  Custom statistics
    ValidLoss     = float(ValidLoss)
    ValidAccuracy = accuracy_score(ValidAnswer, ValidPredictClass)
    ValidAuc      = roc_auc_score(ValidAnswer, ValidPredictScore[:,1])
    ##
    ##
    ##  Choose model
    TuneModelScore = ValidAccuracy
    if( not TuneBetterModel ):
        TuneBetterModel      = Model.cpu()
        TuneBetterModelScore = TuneModelScore
    else:
        if(TuneBetterModelScore < TuneModelScore):
            TuneBetterModel      = Model.cpu()
            TuneBetterModelScore = TuneModelScore
    ##
    ##
    ##  Tune table
    TuneTable["Batch"].append(p["Batch"])
    TuneTable["Optimizer"].append(p["Optimizer"])
    TuneTable["LearnRate"].append(p["LearnRate"])
    TuneTable["Epoch"].append(p["Epoch"])
    TuneTable["Loss"].append(ValidLoss)
    TuneTable["Accuracy"].append(ValidAccuracy)
    TuneTable["Auc"].append(ValidAuc)
    print("Finish a parameter tune.")
##
##
##  Tune table
TuneTable = pandas.DataFrame(TuneTable)
TuneTable.to_csv(TuneResultPath + "\\TuneTable.csv")
################################################################################
##
##
##  Tune better model forward on valid
with torch.no_grad():
    ReportValidNumber        = 0
    ReportValidAnswer        = []
    ReportValidPredictScore  = []
    for ValidMinBatchIndex, ValidMinBatch in enumerate(ValidBatchSet):
        ValidMinBatchScore  = TuneBetterModel(ValidMinBatch[0].cpu(), ValidMinBatch[1].cpu())
        ##
        ##
        ##  Together
        ReportValidNumber   = ReportValidNumber + ValidMinBatch[0].size()[0]
        ReportValidAnswer.append(ValidMinBatch[2])
        ReportValidPredictScore.append(ValidMinBatchScore)
##
##
##  Valid information
ReportValidAnswer            = torch.cat(ReportValidAnswer, dim = 0)
ReportValidPredictScore      = torch.cat(ReportValidPredictScore, dim = 0)
_, ReportValidPredictClass   = torch.max(ReportValidPredictScore,1)
ReportValidAccuracy          = accuracy_score(ReportValidAnswer, ReportValidPredictClass)
##
##
##  Confust matrix
ReportValidConfuseMatrix     = confusion_matrix(ReportValidAnswer, ReportValidPredictClass)
ConfusionTable = plot_confusion_matrix(y_true = ReportValidAnswer, y_pred = ReportValidPredictClass).get_figure()
ConfusionTable.savefig(TuneResultPath + "\\ConfusionTable.png")
##
##
##  Roc curve and auc
AUC = plot_roc(y_true=ReportValidAnswer, y_probas=ReportValidPredictScore, plot_micro=False, plot_macro=False, classes_to_plot=[1]).get_figure()
AUC.savefig(TuneResultPath + "\\AUC.png")
