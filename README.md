# Universal-DC-GAN [WIP]

## Overview

Hello! This is an implementation of Deep Convolutional Generative Adversarial Network in TensorFlow. It is made to be simple, intuitive and as fast at training as possible.

## Getting Started

In order to run Universal-DC-GAN, you will need the following dependencies:
- tensorflow==1.8.0
- numpy==1.14.3
- termcolor==1.1.0
- matplotlib==2.2.2
- scipy==1.1.0
- Pillow==5.2.0
- colorama==0.3.9

You can download them by running this command while being in Universal-DC-GAN directory:

    pip install -r requirements.txt
    
## Config

Default settings are in `config.py` file. You can configure pretty much anything you want there, each option is explained inside the file. Here are some basic options:

`dataset_name = 'textures_all'` - Name of the folder, containing your dataset, it should be placed in `data_folder` path.

`data_folder = './data/'` - Default location of all datasets, your dataset folder, specified in the variable `dataset_name` should be placed inside.

`saves_folder = './saves/'` - Default location of saved models. They save automatically during the training!

`train = True` - If you have already trained the network, you can directly generate images based on your model.

`w, h = 512, 512` - Width and height of all the images in your training dataset

## Running

In order to run the script, just do:

    python model.py
    
Example results should now appear, every number of iterations specified in config file to the output directory (default is `output`)

## Examples

You can find some of the generated examples below. Note that the results may be improved significantly, by extending the time, needed to train the network. Models used to generate these examples were trained around 3-4 hours each, on GTX1060.

### 128x128
![alt text](https://raw.githubusercontent.com/gstark0/Universal-DC-GAN/master/sample_images/sample_output/128x128/11850.png)
![alt text](https://raw.githubusercontent.com/gstark0/Universal-DC-GAN/master/sample_images/sample_output/128x128/12400.png)
![alt text](https://raw.githubusercontent.com/gstark0/Universal-DC-GAN/master/sample_images/sample_output/128x128/12450.png)
![alt text](https://raw.githubusercontent.com/gstark0/Universal-DC-GAN/master/sample_images/sample_output/128x128/12700.png)
![alt text](https://raw.githubusercontent.com/gstark0/Universal-DC-GAN/master/sample_images/sample_output/128x128/13250.png)

### 256x256
![alt text](https://raw.githubusercontent.com/gstark0/Universal-DC-GAN/master/sample_images/sample_output/256x256/350.png)
![alt text](https://raw.githubusercontent.com/gstark0/Universal-DC-GAN/master/sample_images/sample_output/256x256/400.png)
![alt text](https://raw.githubusercontent.com/gstark0/Universal-DC-GAN/master/sample_images/sample_output/256x256/1500.png)
