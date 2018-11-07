##
##
##  Clean table
import os, numpy, pandas, PIL.Image, cv2, shutil
from skimage import measure
from matplotlib import pyplot as plot
os.chdir(".\\History\\CleanTable\\")
Table = pandas.read_csv('.\\Table.csv')
##
##
##  Group by "image1" and "image2"
Table = Table[["user_id", "image1", "image2", "mole_size", "period", "change_1month", "gender", "age", "result"]]
Part1 = Table[["user_id", "image1", "mole_size", "period", "change_1month", "gender", "age", "result"]]
Part2 = Table[["user_id", "image2", "mole_size", "period", "change_1month", "gender", "age", "result"]]
Part1 = Part1.rename(columns = {'image1':'image'})
Part2 = Part1.rename(columns = {'image2':'image'})
Table = pandas.concat([Part1,Part2])
##
##
##  Remove missing data
Table = Table.replace('None'  , numpy.nan).dropna()
Table = Table.replace('26-35y', numpy.nan).dropna()
Table = Table.replace('19-25y', numpy.nan).dropna()
Table = Table.replace("--待分類--, None, None", numpy.nan).dropna()
##
##
##  Rename
Rename = {"user_id":"UserId", "image":"CropImage", "mole_size":"OverSize", "period":"Period","change_1month":"Change", "gender":"Gender", "age":"Age", "result":"Label"}
Table = Table.rename(columns = Rename)
##
##
##  Gender
Table = Table.replace("男性"    , "M")
Table = Table.replace("女性"    , "F")
Table = Table.replace("不想回答", "N")
##
##
##  Age
Table = Table.replace("21歲以下", "Teen")
Table = Table.replace("21~40歲", "Youth")
Table = Table.replace("40~65歲", "Middle")
Table = Table.replace("65歲以上", "Old")
##
##
##  Over size
Table = Table.replace("no",  "N")
Table = Table.replace("yes", "Y")
##
##
##  Period
Table = Table.replace("1個月內", "Month")
Table = Table.replace("1個月～1年", "Year")
Table = Table.replace("1年以上", "OverYear")
Table = Table.replace("不記得", "Unknown")
##
##
##  Change
Table = Table.replace("有變化", "Y")
Table = Table.replace("無變化", "N")
Table = Table.replace("Unknown", "Unknown")
##
##
##  Label
Table = Table.replace("低風險, None, None", "0")
Table = Table.replace("中低風險, None, None", "1")
Table = Table.replace("中低風險, 其他, None", "1")
Table = Table.replace("中低風險, 可能是角化斑（老人斑), None", "1")
Table = Table.replace("中高風險, None, None", "2")
Table = Table.replace("高風險, None, None", "3")
Table = Table.replace("照片較模糊, None, None", "4")
Table = Table.replace("無法判斷, None, None", "5")
Table = Table.replace("無法判斷，請務必找皮膚科專科醫師做進一步診治。, None, None", "5")
Table = Table.replace("無法判斷, 可能是雀斑, None", "5")
Table = Table.replace("這不是痣, 其他, None", "6")
Table = Table.replace("這不是痣, 可能是角化斑（老人斑), None", "6")
Table = Table.replace("這不是痣, 可能是血管瘤, None", "6")
Table = Table.replace("這不是痣, 其他, scar", "6")
Table = Table.replace("這不是痣, 其他, ulcer with crust", "6")
Table = Table.replace("這不是痣, 其他, skin tag", "6")
Table = Table.replace("這不是痣, 其他, skin tag?", "6")
Table = Table.replace("這不是痣, 其他, it's a cup", "6")
Table = Table.replace("這不是痣, 其他, macbook", "6")
Table = Table.replace("這不是痣, 其他, 可能是皮膚纖維瘤 建議看皮膚科醫師", "6")
Table = Table.replace("這不是痣, None, None", "6")
Table = Table.replace("這不是痣, 可能是黑色素沈澱, None", "6")
Table = Table.replace("這不是痣, 其他, cd album", "6")
Table = Table.replace("這不是痣, 其他, evacuation map", "6")
Table = Table.replace("這不是痣, 其他, rice", "6")
Table = Table.replace("這不是痣, 可能是傷害性刺青, None", "6")
Table = Table.replace("這不是痣, 可能是雀斑, None", "6")
Table = Table.replace("這不是痣, 可能是曬斑, None", "6")
##
##
##  Label count
LabelCount = Table.groupby("Label").count()
LabelCount = pandas.DataFrame(LabelCount["UserId"]).rename(columns={"UserId": "Count"})
LabelCount.to_csv(".\\LabelCount.csv")
##
##
##  Select variable and label
Label = Table[["Label","CropImage"]]
Variable = Table[["OverSize", "Period", "Change", "Gender", "Age"]]
Variable = pandas.get_dummies(Variable)
Table = pandas.concat([Label, Variable], axis=1)
Table.shape
Table.head()
Table.to_csv("./CleanTable.csv", index =False)
