# Breast Cancer classifier 

## Usage
python pomias_resnet-3cls.py

## Environment
python 3.6
tensorflow-gpu 2.0

## Data preparation
Commonly used public mammography datasets include [MIAS](http://peipa.essex.ac.uk/pix/mias/),[DDSM](http://www.eng.usf.edu/cvprg/Mammography/Database.html) and [CBIS-DDSM](https://wiki.cancerimagingarchive.net/display/Public/CBIS-DDSM).

You need to convert mammography into .png format

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

