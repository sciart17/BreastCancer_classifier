# Breast Cancer classifier 

## Usage
python pomias_resnet-3cls.py

## Environment
python 3.6
tensorflow-gpu 2.0

## Data preparation
You can use public mammography datasets such as [MIAS](http://peipa.essex.ac.uk/pix/mias/), [DDSM](http://www.eng.usf.edu/cvprg/Mammography/Database.html) and [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM).  (The CBIS-DDSM dataset is an updated and standardized version of the DDSM dataset.)

You need to convert mammography into .png format and split dataset into train, val and test set.

MIAS: .pmg --> .png		Read images using PIL and save them as .png format directly.

DDSM: .LJPEG --> .png		Refer to (https://blog.csdn.net/liuxinghan1998/article/details/91493334?depth_1-utm_source=distribute.pc_relevant.none-task&utm_source=distribute.pc_relevant.none-task)

CBIS-DDSM: .dcm-->.png
````bash
#!/bin/bash
# This script is used to find .dcm files and convert them into 16-bit .png files
OLDIFS="$IFS"
IFS=$'\n'
picname=$(find $'E:/CBIS-DDSM' -name '*.dcm')    
for file in $picname   
do    
	echo $file
	dcmj2pnm +on2 $file ${file%.*}.png	  
done  
IFS="$OLDIFS"
`````````
Your directory tree should be like this:
````bash
$CODE_ROOT
├── train
│   ├── norm
│   ├── benign
│   └── malignant
│  
├── val
│   ├── norm
│   ├── benign
│   └── malignant
│  
├── test
│   ├── norm
│   ├── benign
│   └── malignant
│  
````
Train
--------------
generate a new .h5 file

Test
--------------
test model

