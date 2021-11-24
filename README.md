# Install software requirements
Please run the following pip install command:
```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
If you wish to run any of the bash scripts outlined below, this command has been added to each script in order to set-up your environment on the fly. 

We have tested our networks with the following setup:
```
Python 3.7
Ubuntu 20.04.3
CUDA 11.0
PyTorch 1.7.0
```

# Download code + dataset + pre-trained models

You may either clone this repo and download the dataset and pre-trained feature encoder model separately (see below) or download all code + dataset + pre-trained models [by following this link](https://1drv.ms/u/s!Ai8vR3oqUKxTcG25XBvPyykMIJ8?e=GVZIFy).

# How to run

## Dataset
Please download the dataset from the following link [located here](https://1drv.ms/u/s!Ai8vR3oqUKxTa7IgFALZbzm0rHo?e=yUzVP0).

Thereafter, unzip and place the Train, Test and Validation folders within the ```Dataset``` folder such that the paths for the Train, Test and Validation sets are ```/Dataset/Train```, ```Dataset/Test```, ```Dataset/Validation```.

## Run inference only (do not train new regressor)
To only run inference, please run the following bash command within the root directory:
```
./Run_Inference_Only.sh
```
Here, the pre-trained regressor available at ```./RegressorPreTrained``` will be used.

## Train either the feature encoder or regressor
If you wish to only train the feature encoder using contrastive learning, or the regressor using the normal and position based loss, you may run
```
./Run_Feature_Encoder_Training_Only.sh
```
for feature encoder training or you may train the regressor using the following command:
```
./Run_Regressor_Training_Only.sh
```
For this scenario, the pre-trained feature encoder has to be downloaded and saved to the ```./FeatureEncoderPreTrained``` folder before you may train the regressor. Please download this pre-trained feature encoder model [from this link](https://1drv.ms/u/s!Ai8vR3oqUKxTbXv4m0nnEmVVMPI?e=lwj9XO).

## Train the feature encoder, regressor and perform inference using newly trained models
You may either run the feature encoding training, regressor training and inference altogether. In this scenario, the feature encoder and the regressor have to be trained before inference can be done. The following bash script (Run_All.sh) accomplishes that and can be run from the root directory:
```
./Run_All.sh
```
The trained feature encoder and regressor will be saved into ```./FeatureEncoderTrained``` and ```./RegressorTrained```, respectively.
