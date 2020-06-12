# Breast Cancer classifier 

## Usage
python pomias_resnet-3cls.py

## Environment
python 3.6
tensorflow-gpu 2.0

## Data preparation
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

