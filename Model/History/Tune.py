################################################################################
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
from keras.callbacks import EarlyStopping
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
try:
    os.chdir("TuneParameter/")
except:
    pass
##
##  Variable
Variable = ["OverSize_N", "OverSize_Y", "Period_Month",
 "Period_OverYear", "Period_Unknown", "Period_Year", "Change_N","Change_Unknown",
 "Change_Y", "Gender_F", "Gender_M","Gender_N","Age_Middle",
 "Age_Old","Age_Teen","Age_Youth"]
VariableSize = len(Variable)
##
##  Resize
Resize = (64, 64)
##
##  Train
Train = {}
Train["Table"] = pandas.read_csv("HoldOut/Train/Table.csv" )
Train["Id"]    = [i for i in Train["Table"].Index]
Train["Label"] =  Train["Table"].Label
Train["Image"] = []
for i, d in Train["Table"].iterrows():
    iLabel = str(d.Label)
    iImage = PIL.Image.open("HoldOut/Train/Image/" + iLabel + "/" + str(d.Index) + ".jpg")
    iImage = iImage.resize(Resize)
    iImage = numpy.array(iImage)
    iImage = iImage.astype("float32")
    iImage = iImage / 255
    Train["Image"].append(iImage)
Train["Image"] = numpy.array(Train["Image"])
Train["EncodeLabelCode"] = to_categorical(Train["Label"])
Train["Variable"] = Train["Table"][Variable]
##
##  Valid
Valid = {}
Valid["Table"] = pandas.read_csv("HoldOut/Valid/Table.csv" )
Valid["Id"]    = [i for i in Valid["Table"].Index]
Valid["Label"] =  Valid["Table"].Label
Valid["Image"] = []
for i, d in Valid["Table"].iterrows():
    iLabel = str(d.Label)
    iImage = PIL.Image.open("HoldOut/Valid/Image/" + iLabel + "/" + str(d.Index) + ".jpg")
    iImage = iImage.resize(Resize)
    iImage = numpy.array(iImage)
    iImage = iImage.astype("float32")
    iImage = iImage / 255
    Valid["Image"].append(iImage)
Valid["Image"] = numpy.array(Valid["Image"])
Valid["EncodeLabelCode"] = to_categorical(Valid["Label"])
Valid["Variable"] = Valid["Table"][Variable]
################################################################################
##  Parameter control
Parameter = {}
Parameter["Batch"] = [16, 16, 16]
Parameter["Epoch"] = [50]
Parameter["LearnRate"] = [0.0001,0.0001]
Parameter["Optimizer"] = ["Adam"]
Parameter = list(ParameterGrid(Parameter))
##
##  Tune result
TuneResult = {}
TuneResult["Batch"] = []
TuneResult["Epoch"] = []
TuneResult["LearnRate"] = []
TuneResult["TrainAUC"] = []
TuneResult["TrainAccuracy"] = []
TuneResult["ValidAUC"] = []
TuneResult["ValidAccuracy"] = []
##
##  The better model
TheBetterModel = "Empty"
print("Prepare!")
################################################################################
##
##  Tune loop
for p in Parameter:
    ##
    ##  Reproducible session
    backend.get_session()
    numpy.random.seed(2018)
    tensorflow.set_random_seed(2018)
    ##
    ##  Create model
    from BaseOnModel.UseImageVariable.UseKeras.UseHoldOut.Model import Model
    model = Model.CustomResNet(ImageSize = Resize + (3,), VariableSize = VariableSize)
    ##
    ##  Optimizer
    if(p["Optimizer"]=="Adadelta"):
        TheOptimizer =  keras.optimizers.Adadelta
    if(p["Optimizer"]=="Adam"):
        TheOptimizer =  keras.optimizers.Adam
    if(p["Optimizer"]=="SGD"):
        TheOptimizer =  keras.optimizers.SGD
    ##
    ##  Compile
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=TheOptimizer(lr=p["LearnRate"]),
                  metrics=["acc"])
    ##
    ##  Fit.
    model.fit(
        [Train["Image"], Train["Variable"]], Train["EncodeLabelCode"],
        class_weight = {0:1, 1:1},
        batch_size=p["Batch"],
        epochs=p["Epoch"],
        verbose=1,
        validation_data=([Valid["Image"], Valid["Variable"]], Valid["EncodeLabelCode"]),
        callbacks = [EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto', restore_best_weights=True)]
        )
    ##
    ## Evaluate on train
    Train["Probability"]   = model.predict([Train["Image"], Train["Variable"]])
    Train["Prediction"]    = numpy.argmax(Train["Probability"], axis=1)
    Train["AUC"]           = roc_auc_score(numpy.array(Train["Label"]), Train["Probability"][:,1])
    Train["Accuracy"]      = accuracy_score(Train["Label"], Train["Prediction"])
    Train["ConfuseMatrix"] = confusion_matrix(Train["Label"], Train["Prediction"])
    ##
    ## Evaluate on valid
    Valid["Probability"]   = model.predict([Valid["Image"], Valid["Variable"]])
    Valid["Prediction"]    = numpy.argmax(Valid["Probability"], axis=1)
    Valid["AUC"]           = roc_auc_score(numpy.array(Valid["Label"]), Valid["Probability"][:,1])
    Valid["Accuracy"]      = accuracy_score(Valid["Label"], Valid["Prediction"])
    Valid["ConfuseMatrix"] = confusion_matrix(Valid["Label"], Valid["Prediction"])
    ##
    ##  Summary to tune result
    TuneResult["Batch"].append( p["Batch"] )
    TuneResult["Epoch"].append( p["Epoch"] )
    TuneResult["LearnRate"].append( p["LearnRate"] )
    TuneResult["TrainAUC"].append( Train["AUC"] )
    TuneResult["ValidAUC"].append( Valid["AUC"] )
    TuneResult["TrainAccuracy"].append( Train["Accuracy"] )
    TuneResult["ValidAccuracy"].append( Valid["Accuracy"] )
    ##
    ##  The better model
    if( TheBetterModel=="Empty" ):
        TheBetterMetric = Valid["AUC"]
        TheBetterModel    = model
    else:
        if( TheBetterMetric < Valid["AUC"] ):
            TheBetterMetric = Valid["AUC"]
            TheBetterModel    = model
    pass
print("Finish tune loop!")
##
##  Create result folder
Time = str(datetime.datetime.now())
Time = re.sub("[- :.]", "", Time)
ResultPath = "BaseOnModel/UseImageVariable/UseKeras/UseHoldOut/Result/"
os.mkdir(ResultPath + Time)
##
##  Save tune result
pandas.DataFrame(TuneResult).to_csv(ResultPath + Time + "/Tune.csv", index = False)
##
##  Save the better model
TheBetterModel.save(ResultPath + Time + "/TheBetterModel.h5")
################################################################################
##
##  Evaluate the better model on valid
EvaluateTheBetter = {}
EvaluateTheBetter["Model"] = TheBetterModel
##
## AUC score
EvaluateTheBetter["Probability"] = EvaluateTheBetter["Model"].predict([Valid["Image"], Valid["Variable"]])
EvaluateTheBetter["AUC"]         = roc_auc_score(numpy.array(Valid["Label"]), EvaluateTheBetter["Probability"][:,1])
##
##  plot AUC
AUC = plot_roc(y_true=Valid["Label"], y_probas=EvaluateTheBetter["Probability"], plot_micro=False, plot_macro=False, classes_to_plot=[1]).get_figure()
AUC.savefig(ResultPath + Time + "/AUC.png")
EvaluateTheBetter["Prediction"]   = numpy.argmax(EvaluateTheBetter["Probability"], axis=1)
EvaluateTheBetter["ConfuseTable"] = confusion_matrix(Valid["Label"], EvaluateTheBetter["Prediction"])
ConfusionTable = plot_confusion_matrix(y_true = Valid["Label"], y_pred = EvaluateTheBetter["Prediction"]).get_figure()
ConfusionTable.savefig(ResultPath + Time + "/ConfusionTable.png")
##
##  Threshold table
_, _, Threshold = roc_curve(numpy.array(Valid["Label"]), EvaluateTheBetter["Probability"][:,1])
ThresholdTable = {"Threshold":[], "Accuracy":[], "Sensitivity":[], "Specificity":[], "Precision":[]}
for threshold in Threshold:
    prediction = EvaluateTheBetter["Probability"][:,1] > threshold
    accuracy   = accuracy_score(Valid["Label"], prediction)
    confuse    = confusion_matrix(Valid["Label"], prediction)
    sensitivity = confuse[1,1]/sum(confuse[1,:])
    specificity = confuse[0,0]/sum(confuse[0,:])
    precision   = confuse[1,1]/sum(confuse[1,:])
    ThresholdTable["Threshold"].append(threshold)
    ThresholdTable["Accuracy"].append(accuracy)
    ThresholdTable["Sensitivity"].append(sensitivity)
    ThresholdTable["Specificity"].append(specificity)
    ThresholdTable["Precision"].append(precision)
ThresholdTable = pandas.DataFrame(ThresholdTable)
ThresholdTable.to_csv(ResultPath + Time + "/ThresholdTable.csv", index=False)
##
##  Select threshold
SelectThreshold = ThresholdTable.loc[ThresholdTable["Accuracy"]==max(ThresholdTable["Accuracy"])]["Threshold"].iloc[0].item()
prediction       = EvaluateTheBetter["Probability"][:,1] > SelectThreshold
ConfusionTable = plot_confusion_matrix(y_true = Valid["Label"], y_pred = prediction).get_figure()
ConfusionTable.savefig(ResultPath + Time + "/SelectThresholdConfusionTable.png")
##
##  Log
# Log = "10000 train, 3000 valid, 64*64 image, variable and tune."
# with open(ResultPath + Time + "\\Message.txt", "w") as Message:
#     Message.write(Log)
