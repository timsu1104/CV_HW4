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
I used 6 RTX 3090 to train and one model takes about 1 hour. (Hence to run cross-validation with 5 splits takes 5 hours or so. )

To evaluate a model, run
```[language=bash]
python main.py --eval --gpus 7 --eval_ckpt /home/zhengyuan/CVhw4/logs/Adamoptlr3e-3_Test/DeepLabv3+/3tsdfmh5/checkpoints/epoch=299-step=9300.ckpt
```

To visualize, run 
```[language=bash]
python main.py --vis --gpus 7 --eval_ckpt /home/zhengyuan/CVhw4/logs/Adamoptlr3e-3_Test/DeepLabv3+/3tsdfmh5/checkpoints/epoch=299-step=9300.ckpt
```
