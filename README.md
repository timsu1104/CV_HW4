To prepare the data, run
```[language=shell]
csDownload -d ./data gtFine_trainvaltest.zip
csDownload -d ./data leftImg8bit_trainvaltest.zip

cd data
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip

export CITYSCAPES_DATASET=$(realpath .)
csCreateTrainIdLabelImgs
```