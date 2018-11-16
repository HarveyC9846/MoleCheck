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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
Table = pandas.read_csv("./Example/Data/Table.csv")
################################################################################
##
##
##  Input one data, one image
Model = load_model("./Example/20181116043813426948/TheBetterModel.h5")
OneData = Table.iloc[0]
ImagePath = "./Example/Data/Image/"
def Predict(OneData, ImagePath, Model):
    ##
    ##
    ##  Variable term
    VariableTerm = ()
    if( OneData.mole_size == "no" ):
        VariableTerm = VariableTerm + (1,0)
    if( OneData.mole_size == "yes" ):
        VariableTerm = VariableTerm + (0,1)
    if( OneData.period == "1個月內" ):
        VariableTerm = VariableTerm + (1,0,0,0)
    if( OneData.period == "1年以上" ):
        VariableTerm = VariableTerm + (0,1,0,0)
    if( OneData.period == "不記得" ):
        VariableTerm = VariableTerm + (0,0,1,0)
    if( OneData.period == "1個月～1年" ):
        VariableTerm = VariableTerm + (0,0,0,1)
    if( OneData.change == "無變化" ):
        VariableTerm = VariableTerm + (1,0,0)
    if( OneData.change == "不記得" ):
        VariableTerm = VariableTerm + (0,1,0)
    if( OneData.change == "有變化" ):
        VariableTerm = VariableTerm + (0,0,1)
    if( OneData.gender == "女性" ):
        VariableTerm = VariableTerm + (1,0,0)
    if( OneData.gender == "男性" ):
        VariableTerm = VariableTerm + (0,1,0)
    if( OneData.gender == "不想回答" ):
        VariableTerm = VariableTerm + (0,0,1)
    if( OneData.age == "40~65歲" ):
        VariableTerm = VariableTerm + (1,0,0,0)
    if( OneData.age == "65歲以上" ):
        VariableTerm = VariableTerm + (0,1,0,0)
    if( OneData.age == "21~40歲" ):
        VariableTerm = VariableTerm + (0,0,1,0)
    if( OneData.age == "21歲以下" ):
        VariableTerm = VariableTerm + (0,0,0,1)
    VariableTerm = numpy.array([list(VariableTerm)])
    ##
    ##
    ##  Image id
    ImageId      = OneData.image_crop
    ##
    ##
    ##  Process image
    Resize = (64, 64)
    aImage = PIL.Image.open(ImagePath + ImageId).resize(Resize)
    aImage = numpy.array(aImage).astype("float") / 255
    ImageTerm = numpy.array([aImage])
    ##
    ##
    ##  Prediction
    PredictScore = Model.predict([ImageTerm, VariableTerm])
    PredictLabel = int(PredictScore.argmax(axis = 1))
    ##
    ##
    ##  Message
    if( PredictLabel == 0 ):
        Message = "這顆痣目前沒有太大的問題。"
    if( PredictLabel == 1 ):
        Message = "這顆痣可能需要進一步給醫生診斷。"
    print(Message)
    return(Message)
Predict(OneData, ImagePath, Model)
