##
##
##  Base on "Masked"
import os, numpy, pandas, PIL.Image, cv2, shutil
from skimage import measure
from matplotlib import pyplot as plot
try:
    os.chdir(".\\History\\CropImage\\")
except:
    pass
MaskedPath    = ".\\Masked\\"
ColorizedPath = ".\\Colorized\\"
Search = os.listdir(MaskedPath)
##
##
##  Save path
SavePath = ".\\Crop\\"
if( os.path.exists(SavePath) ):
    shutil.rmtree(SavePath)
os.makedirs(SavePath)
##
##
##  i = Search[0]
for i in Search:
    ##
    ##
    ##  Load and check
    iMasked   = PIL.Image.open(MaskedPath + i)
    if( len(numpy.array(iMasked).shape)!=2 ):
        continue

    try:
        iColorized = PIL.Image.open(ColorizedPath + str.split(i,".")[0] + ".jpg")
    except:
        continue
    ##
    ##
    ##  Crop marginal area
    if( iMasked.size == iColorized.size ):
        Size = iMasked.size
        xLU = Size[0]*(1/20)  ##  left up point of x
        yLU = Size[1]*(1/20)  ##  left up point of y
        xRD = Size[0]- xLU  ##  Right down point of x
        yRD = Size[1]- yLU  ##  Right down point of y
        Box = (xLU, yLU, xRD, yRD)
        iMasked    = iMasked.crop(Box)
        iColorized = iColorized.crop(Box)
    else:
        continue
    ##
    ##
    ##  Get contours and check
    iMasked   = numpy.array(iMasked)
    iColorized = numpy.array(iColorized)
    iContours = measure.find_contours(iMasked, 0.5)
    if(len(iContours)!=1):
        continue
    ##
    ##
    ##  Get circle range size and check
    MinX = int( min(iContours[0][:,0]) )
    MinY = int( min(iContours[0][:,1]) )
    MaxX = int( max(iContours[0][:,0]) )
    MaxY = int( max(iContours[0][:,1]) )
    RangeX = MaxX - MinX
    RangeY = MaxY - MinY
    SafeX = 120
    SafeY = 120
    if( (RangeX>SafeX) or (RangeY>SafeY) ):
        continue
    ##
    ##
    ##  Crop and check size
    iCenterX = int( numpy.mean(iContours[0][:,0]) )
    iCenterY = int( numpy.mean(iContours[0][:,1]) )
    Radius = 50
    iCropMasked     = iMasked[iCenterX - Radius : iCenterX + Radius, iCenterY - Radius : iCenterY + Radius]
    iCropColorized  = iColorized[ iCenterX - Radius : iCenterX + Radius, iCenterY - Radius : iCenterY + Radius]
    if( min(iCropMasked.shape)!=2*Radius ):
        continue
    ##
    ##
    ##  Save crop image
    iCropimage = PIL.Image.fromarray(iCropColorized)
    iCropimage.save( SavePath + str.split(i,".")[0] + ".jpg" )
