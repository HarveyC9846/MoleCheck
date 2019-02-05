##
##  Please make sure your work space
import pandas
import os
import shutil
import PIL.Image as pil
import xml.etree.ElementTree as et
import numpy
strOutputPath = "Output/"
try:
    os.makedirs(strOutputPath)
except:
    shutil.rmtree(strOutputPath)
    os.makedirs(strOutputPath)
################################################################################
##
##  1. Clean table
##
##  Original data table
dataTable = pandas.read_csv("Data/Table.csv")
##
##  Select with doctor give result
##  All row data have result
setDoctor = set(['Jack Li', 'Eric Lin 林','Christine(王筱涵)'])
dataTable = dataTable.iloc[[i in setDoctor for i in dataTable["doctor"]]]
dataTable[["user_id", "doctor"]].groupby(["doctor"]).count()
##
##  Encode result
strNotMoleTag = [
 '這不是痣, 可能是角化斑（老人斑), None',
 '這不是痣, 其他, None'
 '這不是痣, 可能是血管瘤, None',
 '這不是痣, 其他, scar',
 '這不是痣, 可能是黑斑, None',
 '這不是痣, 其他, ulcer with crust',
 '這不是痣, 其他, skin tag',
 '這不是痣, 其他, skin tag?',
 '這不是痣, 可能是血管瘤, None',
 '這不是痣, 可能是黑色素沈澱, None',
 '這不是痣, 其他, 可能是皮膚纖維瘤 建議看皮膚科醫師',
 '這不是痣, None, None',
 '這不是痣, 其他, not skin',
 '這不是痣, 其他, None']
dataTable = dataTable.replace(['低風險, None, None'], 0)
dataTable = dataTable.replace(['中低風險, None, None'],1)
dataTable = dataTable.replace(['中高風險, None, None'],2)
dataTable = dataTable.replace(["高風險, None, None"], 3)
dataTable = dataTable.replace(['照片較模糊, None, None'], 4)
dataTable = dataTable.replace(['無法判斷, None, None', '無法判斷，請務必找皮膚科專科醫師做進一步診治。, None, None'], 5)
dataTable = dataTable.replace(strNotMoleTag, 6)
##
##  Select result
dataTable = dataTable.iloc[[i in set([0, 2, 3]) for i in dataTable["result"]]]
##
##  Relabel result
dataTable["result"] = dataTable["result"].replace([2,3],1)
##
##  Select data with useful question
dataTable = dataTable.iloc[[i in set(['21~40歲', '21歲以下', '40~65歲', '65歲以上']) for i in dataTable.age]]
dataTable = dataTable.iloc[[i in set(['no', 'yes']) for i in dataTable.mole_size]]
dataTable = dataTable.iloc[[i in set(['1年以上', '不記得', '1個月～1年', '1個月內']) for i in dataTable.period]]
dataTable = dataTable.iloc[[i in set(['無變化', '不記得', '有變化']) for i in dataTable.change_1month]]
dataTable[["result", "user_id", "doctor"]].groupby(["doctor", "result"]).count()
##
##  Group image
strImage1Tag = ['user_id', 'datetime', 'image1', 'mole_size',
                'period', 'change_1month', 'gender', 'age',
                'result', 'doctor', 'time_result',
                'revision_result', 'revision_dr', 'revision_time']
strImage2Tag = ['user_id', 'datetime', 'image2', 'mole_size',
                'period', 'change_1month', 'gender', 'age',
                'result', 'doctor', 'time_result',
                'revision_result', 'revision_dr', 'revision_time']
dataImage1 = dataTable[strImage1Tag]
dataImage2 = dataTable[strImage2Tag]
dataImage1.columns = dataImage1.columns.str.replace('image1','image')
dataImage2.columns = dataImage2.columns.str.replace('image2','image')
dataTable = pandas.concat([dataImage1, dataImage2])
##
##  Encode variable
dataVariable = pandas.get_dummies(dataTable[['mole_size', 'period', 'change_1month', 'gender', 'age']])
dataTable    = pandas.concat([dataTable[['user_id', 'image', 'result']], dataVariable], axis=1)
dataTable.shape
##
##  Save data table
dataTable.to_csv(strOutputPath + "Table.csv", index = False)
################################################################################
##
##  2. Crop images base on table and location
for index, data in dataTable.iterrows():
    ##
    ##  Crop image1
    file   = data["image"]
    image  = pil.open("Data/Image/" + file)
    xml    = et.parse("Data/Location/" + str.split(file, ".")[0] + ".xml")
    bndbox     = xml.findall("object")[0].findall("bndbox")[0]
    (xmin, ymin, xmax, ymax) = tuple([float(i.text) for i in bndbox])
    center = ((xmin + xmax)/2, (ymin + ymax)/2)
    coordinate= (center[0] - 50, center[1] - 50, center[0] + 50, center[1] + 50)
    crop       = image.crop(coordinate)
    strImagePath = strOutputPath + "Image/" + str(data["result"]) + "/"
    os.makedirs(strImagePath, exist_ok=True)
    with open(strImagePath + file, 'w') as f:
        crop.save(f)
################################################################################
