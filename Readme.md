# Re-implementaion of Enhanced LSTM for Natural Language Inference with Tensorflow
**"Enhanced LSTM for Natural Language Inference"**
Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei, Hui Jiang, Diana Inkpen. _ACL (2017)_

## Dependencies
To run it perfectly, you will need:
* Python 2.7
* Tensorflow (preferable 1.8)

## Running the Script
1. Download and preprocess the data set
```
cd data
bash fetch_and_preprocess.sh
```

2. Train and test model for ESIM
```
cd scripts/ESIM/
bash train.sh
```

The result are shown in `log.txt` and `log_result.txt` file.

## Note

There are three main files in the scripts folder
```
scripts/main.py
```
main.py is only compatible when tensorflow version >= 1.8.
It uses the latest tf.contrib.cudnn\_rnn.CudnnLSTM library that gives you the boosting performance when training the model.
```
scripts/main1.py
```
main1.py are implemented the same as main.py but works with lower version of tensorflow.

```
scripts/main_old.py
```
main\_old.py uses the old libraries, it's slightly slower but gives you stable results compared with the other two mentioned above
