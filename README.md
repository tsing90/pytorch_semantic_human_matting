# pytorch_semantic_human_matting
This is an unofficial implementation of the paper "Semantic human matting". 

# testing environment:
Ubuntu 16.04

Pytorch 0.4.1

# data preparation
For my code, there are two types of data:
1. RGBA-png format: which means the image has no background (removed already)
2. composited image + mask (alpha) image: the composited images have been composited by foreground and background already, and corresponding alpha matting images are also provided.
3. backround images: which can fetched from coco datasets or anywhere else (e.g. internet), and they will be used for randomly compositing new image on-the-fly with foreground images separated from 1 (RGBA-PNG images), as described in paper.

For example: 

For 1: I used Adobe Deep Image Matting datasets; I composite alpha and foreground images togher to get my RGBA-png format images.

For 2: I used Supervisely Human datasets which provides human involved images and corresponding masks; due to its low quality (binary segmentation), I mainly used them to trian T-net only;

For 3: I used coco datasets and some images crawled from internet.

When having those above three types of data, then generate lists of training files containing the full path of training images, such as 'DIM_list.txt', 'super_img.txt'&'super_msk.txt', 'bg_list.txt' in my case.

# Implementation details
The training model is completely implemented as described as in the paper, details are as follows:
* T-net: PSP-50 is deployed for training trimap generation; input is image (3 channels) and output is trimap (one channel);

* M-net: 13 convolutional layers and 4 max-pooling layers with the same hyper-parameters for VGG-16 are used as encoder, and 6 convolutional layers and 4 unpooling layers are used as decoder; input is image and trimap (6 channels) and output is alpha image (1 channel);

* Fusion: the fusion loss functions are implemented as described in paper;

* **_This model is flexible for inferencing any size of images when well trained._**

# How to run the code
## pre_trian T_net
python train.py --patch_size=400 --nEpochs=500 --save_epoch=5 --train_batch=8 --train_phase=pre_train_t_net

optional: --continue_train

## pre_train M_net
python train.py --patch_size=400 --nEpochs=500 --save_epoch=1 --train_batch=8 --train_phase=pre_train_m_net

optional: --continue_train

## end to end training
python train.py --patch_size=800 --nEpochs=500 --lr=1e-5 --save_epoch=10 --train_batch=8 --continue_train --train_phase=end_to_end

optional: --continue_train

-----------------------------------
|       Improving & Debugging ... |   
-----------------------------------
Trying to cropping tensors on the fly when traning ...


## testing
python test.py --train_phase=end_to_end

# Results
** will be released soon ... **
-------------------------------
