##
##
##  Load and check
import pandas, os, sklearn, PIL.Image, numpy, shutil
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.image import ImageDataGenerator
try:
    os.chdir(".\\SplitTrain\\")
except:
    pass
SavePath = ".\\HoldOut\\"
if( os.path.exists(SavePath) ):
    shutil.rmtree(SavePath)
Table = pandas.read_csv(".\\CleanTable.csv")
Status = []
for i, d in Table.iterrows():
    try:
        CanLoad = PIL.Image.open("./Crop/" + d.CropImage)
        Status.append(1)
    except:
        Status.append(numpy.nan)
Table["Status"] = Status
Table = Table.dropna()
Table.shape
##
##
##  Relabel
The0Table = pandas.concat([Table.loc[Table["Label"]==0]])
The1Table = pandas.concat([Table.loc[Table["Label"]==2], Table.loc[Table["Label"]==3]])
The1Table["Label"] = The1Table["Label"].replace([2, 3], 1)
Table = pandas.concat([The0Table, The1Table])
LabelCount = pandas.DataFrame(Table.groupby("Label").count()["CropImage"])
LabelCount = LabelCount.rename(columns={"CropImage":"Count"})
LabelCount.to_csv(".\\LabelCount.csv")
##
##
##  Split by hold out
Train, Valid = train_test_split(Table, test_size = 0.3, stratify = Table.Label)
##
##
##  Balance train data
TrainBalanceSize = 15000
Class = list(Train.Label.unique())
FakeTrainList = []
for i in Class:
    numpy.random.seed(2018)
    iFakeTrain = Train.loc[Train.Label==i].sample(TrainBalanceSize, replace =True)
    FakeTrainList.append(iFakeTrain)
Train = pandas.concat(FakeTrainList)
##
##
##  Balance valid data
BalanceValid = True
if(BalanceValid):
    ValidBalanceSize = 2500
    FakeValidList = []
    Class = list(Valid.Label.unique())
    for i in Class:
        numpy.random.seed(2018)
        iFakeValid = Valid.loc[Valid.Label==i].sample(ValidBalanceSize, replace =True)
        FakeValidList.append(iFakeValid)
    Valid = pandas.concat(FakeValidList)
else:
    Valid = pandas.DataFrame(Valid)
##
##
##  Combination
Train["Type"] = "Train"
Valid["Type"] = "Valid"
TrainValid = pandas.concat([Train,Valid])
TrainValid["Index"] = range(TrainValid.shape[0])
##
##
##  Save path
os.makedirs(SavePath)
for i in Class:
    os.makedirs(SavePath + "Train\\Image\\" + str(i) + "\\")
    os.makedirs(SavePath + "Valid\\Image\\" + str(i) + "\\")
##
##
##  Balance and save image
numpy.random.seed(2018)
for i, d in TrainValid.iterrows():
    ##
    ##
    ##  Load image
    iImage = PIL.Image.open(".\\Crop\\" + d.CropImage)
    ##
    ##
    ##  Generator new image function
    GeneratorTool = ImageDataGenerator(rotation_range = 360, horizontal_flip=True, vertical_flip=True)
    iImage = numpy.array(iImage)
    iNewShape = (1,) + iImage.shape
    iImage = iImage.reshape(iNewShape)
    ##
    ##
    ##  Creat new image
    iNewImage = GeneratorTool.flow(iImage).next()
    iNewImage = iNewImage[0,:,:,:].astype("uint8")
    iNewImage = PIL.Image.fromarray(iNewImage)
    iNewImage.save(SavePath + d.Type + "\\" + "Image\\" + str(d.Label) + '\\' + str(d.Index) + ".jpg")
##
##
##  Save table
Variable = ["Label", "Index", "OverSize_N", "OverSize_Y", "Period_Month",
 "Period_OverYear", "Period_Unknown", "Period_Year", "Change_N","Change_Unknown",
 "Change_Y", "Gender_F", "Gender_M","Gender_N","Age_Middle",
 "Age_Old","Age_Teen","Age_Youth", "Type"]
TrainValid = TrainValid[Variable]
Train = TrainValid.loc[TrainValid.Type == "Train"]
Valid = TrainValid.loc[TrainValid.Type == "Valid"]
Train.to_csv(SavePath + "Train/Table.csv", index = False)
Valid.to_csv(SavePath + "Valid/Table.csv", index = False)
##
##
##  Save label count
Train[["Label", "Index"]].groupby("Label").count().rename(columns={"Index": "Count"}).to_csv(SavePath + "Train\\LabelCount.csv")
Valid[["Label", "Index"]].groupby("Label").count().rename(columns={"Index": "Count"}).to_csv(SavePath + "Valid\\LabelCount.csv")
print("Finish")
