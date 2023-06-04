To prepare the data, run
```[language=bash]
csDownload -d ./data gtFine_trainvaltest.zip
csDownload -d ./data leftImg8bit_trainvaltest.zip

cd data
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip

export CITYSCAPES_DATASET=$(realpath .)
csCreateTrainIdLabelImgs
```

To train the model, run 
```[language=bash]
python main.py --tag DeepLabv3 --gpus 0,1,2,3,4,5 # use cross validation on training set
python main.py --tag DeepLabv3 --gpus 0,1,2,3,4,5 --test # use train set and val set
```