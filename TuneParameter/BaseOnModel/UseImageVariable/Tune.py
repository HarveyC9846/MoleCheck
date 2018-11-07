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
os.chdir(".\\TuneParameter\\")
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
Resize = (48, 48)
##
##
##  Train
Train = {}
Train["Table"] = pandas.read_csv(".\\Train\\Table.csv" )
Train["Id"]    = [i for i in Train["Table"].Index]
Train["Label"] =  Train["Table"].Label
Train["Image"] = []
for i, d in Train["Table"].iterrows():
    iLabel = str(d.Label)
    iImage = PIL.Image.open(".\\Train\\Image\\" + iLabel + "\\" + str(d.Index) + ".jpg")
    iImage = iImage.resize(Resize)
    iImage = numpy.array(iImage)
    iImage = iImage.astype("float32")
    iImage = iImage / 255
    ##
    ##
    ##  Concatenate
    # iImage = numpy.concatenate((iImage, aOriented), axis=2)
    Train["Image"].append(iImage)
Train["Image"] = numpy.array(Train["Image"])
Train["EncodeLabelCode"] = to_categorical(Train["Label"])
Train["Variable"] = Train["Table"][Variable]
##
##
##  Valid
Valid = {}
Valid["Table"] = pandas.read_csv(".\\Valid\\Table.csv" )
Valid["Id"]    = [i for i in Valid["Table"].Index]
Valid["Label"] =  Valid["Table"].Label
Valid["Image"] = []
for i, d in Valid["Table"].iterrows():
    iLabel = str(d.Label)
    iImage = PIL.Image.open(".\\Valid\\Image\\" + iLabel + "\\" + str(d.Index) + ".jpg")
    iImage = iImage.resize(Resize)
    iImage = numpy.array(iImage)
    iImage = iImage.astype("float32")
    iImage = iImage / 255
    ##
    ##
    ##  Concatenate
    # iImage = numpy.concatenate((iImage, aOriented), axis=2)
    Valid["Image"].append(iImage)
Valid["Image"] = numpy.array(Valid["Image"])
Valid["EncodeLabelCode"] = to_categorical(Valid["Label"])
Valid["Variable"] = Valid["Table"][Variable]
################################################################################
##
##
##  Parameter control
Parameter = {}
Parameter["Batch"] = [8,16]
Parameter["Epoch"] = [2]
Parameter["LearnRate"] = [1e-3]
Parameter["Optimizer"] = ["Adadelta"]
Parameter = list(ParameterGrid(Parameter))
##
##
##  Result control
TuneResult = {}
TuneResult["Batch"] = []
TuneResult["Epoch"] = []
TuneResult["LearnRate"] = []
TuneResult["TrainAUC"] = []
TuneResult["TrainAccuracy"] = []
TuneResult["ValidAUC"] = []
TuneResult["ValidAccuracy"] = []
##
##
##  TheBetterModel
TheBetterModel = "Empty"
print("Prepare!")
################################################################################
##
##
##  Tune loop
for p in Parameter:
    ##
    ##
    ##  Reproducible session
    backend.get_session()
    numpy.random.seed(2018)
    tensorflow.set_random_seed(2018)
    ##
    ##
    ##  Create model
    from BaseOnModel.UseImageVariable.Model import Model
    model = Model.I2CLMV1FLO(ImageSize = Resize + (3,), VariableSize = VariableSize)
    ##
    ##
    ##  Compile
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=p["LearnRate"]),
                  metrics=["acc"])
    ##
    ##
    ##  Fit.
    model.fit(
        [Train["Image"], Train["Variable"]], Train["EncodeLabelCode"],
        # Train["Image"], Train["EncodeLabelCode"],
        batch_size=p["Batch"],
        epochs=p["Epoch"],
        verbose=1,
        validation_data=([Valid["Image"], Valid["Variable"]], Valid["EncodeLabelCode"])
        )
    ##
    ##
    ## Evaluate on train
    Train["Probability"] = model.predict([Train["Image"], Train["Variable"]])
    Train["Prediction"]  = numpy.argmax(Train["Probability"], axis=1)
    Train["AUC"]         = roc_auc_score(numpy.array(Train["Label"]), Train["Probability"][:,1])
    Train["Accuracy"] = accuracy_score(Train["Label"], Train["Prediction"])
    Train["ConfuseMatrix"] = confusion_matrix(Train["Label"], Train["Prediction"])
    ##
    ##
    ## Evaluate on valid
    Valid["Probability"] = model.predict([Valid["Image"], Valid["Variable"]])
    Valid["Prediction"]  = numpy.argmax(Valid["Probability"], axis=1)
    Valid["AUC"]         = roc_auc_score(numpy.array(Valid["Label"]), Valid["Probability"][:,1])
    Valid["Accuracy"] = accuracy_score(Valid["Label"], Valid["Prediction"])
    Valid["ConfuseMatrix"] = confusion_matrix(Valid["Label"], Valid["Prediction"])
    ##
    ##
    ##  Summary to result
    TuneResult["Batch"].append( p["Batch"] )
    TuneResult["Epoch"].append( p["Epoch"] )
    TuneResult["LearnRate"].append( p["LearnRate"] )
    TuneResult["TrainAUC"].append( Train["AUC"] )
    TuneResult["ValidAUC"].append( Valid["AUC"] )
    TuneResult["TrainAccuracy"].append( Train["Accuracy"] )
    TuneResult["ValidAccuracy"].append( Valid["Accuracy"] )
    ##
    ##
    ##  The better model
    if( TheBetterModel=="Empty" ):
        TheBetterAccuracy = Valid["Accuracy"]
        TheBetterModel    = model
    else:
        if( TheBetterAccuracy < Valid["Accuracy"] ):
            TheBetterAccuracy = Valid["Accuracy"]
            TheBetterModel    = model
    pass
print("Finish tune loop!")
##
##
##  Create result folder
Time = str(datetime.datetime.now())
Time = re.sub("[- :.]", "", Time)
ResultPath = ".\\BaseOnModel\\UseImageVariable\\Result\\"
os.mkdir(ResultPath + Time)
##
##
##  Save tune result
pandas.DataFrame(TuneResult).to_csv(ResultPath + Time + "\\Tune.csv", index = False)
##
##
##  Save the better model
TheBetterModel.save(ResultPath + Time + "\\TheBetterModel.h5")
##
##
##  Evaluate the better model on valid
Evaluate = {}
Evaluate["Model"] = TheBetterModel
##
##
## AUC score
Evaluate["Probability"] = Evaluate["Model"].predict([Valid["Image"], Valid["Variable"]])
Evaluate["AUC"]         = roc_auc_score(numpy.array(Valid["Label"]), Evaluate["Probability"][:,1])
##
##
##  plot AUC
AUC = plot_roc(y_true=Valid["Label"], y_probas=Evaluate["Probability"], plot_micro=False, plot_macro=False, classes_to_plot=[1]).get_figure()
AUC.savefig(ResultPath + Time + "\\AUC.png")
##
##
##  Accuracy
Evaluate["Prediction"]  = numpy.argmax(Evaluate["Probability"], axis=1)
Evaluate["Accuracy"]    = accuracy_score(Valid["Label"], Evaluate["Prediction"])
##
##
##  Confusion table
Evaluate["ConfuseTable"] = confusion_matrix(Valid["Label"], Evaluate["Prediction"])
##
##
##  Plot confusion table
ConfusionTable = plot_confusion_matrix(y_true = Valid["Label"], y_pred = Evaluate["Prediction"]).get_figure()
ConfusionTable.savefig(ResultPath + Time + "\\ConfusionTable.png")
##
##
##  AASN
Evaluate["SPC"] = Evaluate["ConfuseTable"][1,1]/sum(Evaluate["ConfuseTable"][1,:])
Evaluate["NPV"] = Evaluate["ConfuseTable"][1,1]/sum(Evaluate["ConfuseTable"][:,1])
AASN = {"Accuracy":[Evaluate["Accuracy"]], "AUC":[Evaluate["AUC"]], "SPC":[Evaluate["SPC"]], "NPV":[Evaluate["NPV"]]}
pandas.DataFrame(AASN).to_csv(ResultPath + Time + "\\AASN.csv", index=False)
