This repo is for homework 4 of computer vision. Written by Zhengyuan Su. 

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
python main.py --tag DeepLabv3 --gpus 0,1,2,3,4,5 # use cross-validation on training set
python main.py --tag DeepLabv3 --gpus 0,1,2,3,4,5 --test # train a whole model, report metrics on the validation set (used as the test set)
```
I used 6 RTX 3090 to train and one model takes about 1 hour. (Hence to run cross-validation with 5 splits takes 5 hours or so. )

After running, the logs will be synchronized online, and the checkpoints can be found under `logs/$TAG/DeepLabv3+/$LOGINDEX/checkpoints`. 

To evaluate a model, run
```[language=bash]
python main.py --eval --gpus 7 --eval_ckpt $PATH_TO_CHECKPOINT
```

To visualize, run 
```[language=bash]
python main.py --vis --num_vis 10 --gpus 7 --eval_ckpt $PATH_TO_CHECKPOINT
```
The result will be saved in ``./visualizations``. 