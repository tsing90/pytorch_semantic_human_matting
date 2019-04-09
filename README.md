# pytorch_semantic_human_matting
This is an unofficial implementation of the paper "Semantic human matting". 

# testing environment:
Ubuntu 16.04

Pytorch 0.4.1

# data preparation

To make our life easier, there are only two types of data:
1. RGBA-png format: which means the image has no background (removed already), 
you can generate such an image with the function 'get_rgba' from ./data/data_util.py.
2. Backround images: which can fetched from coco datasets or anywhere else (e.g. internet), and they will be used for randomly compositing new images on-the-fly with foreground images separated from 1 (RGBA-PNG images), as described in paper.

For example: 

For 1: I used Adobe Deep Image Matting datasets, etc.; I composite alpha and foreground images togher to get my RGBA-png format images.

For 2: I used coco datasets and some images crawled from internet.

When having those above two types of data, then generate lists of training files containing the full path of training images, 
such as 'DIM_list.txt', 'bg_list.txt' in my case. Specifically, for flags:

--fgLists: a list, contains list files in which all images share the same fg-bg ratio, e.g. ['DIM.txt','SHM.txt']

--bg_list: a txt file, contains all bg images for composition needs, e.g. 'bg_list.txt'.

--dataRatio: a list, contains bg-fb ratio for each list file in fgLists. For example, similar to the paper, 
given [100, 1], we composite 100 images for each fore-ground image in 'DIM.txt' and 1 image for each fg in 'SHM.txt'

# Implementation details
The training model is completely implemented as described as in the paper, details are as follows:
* T-net: PSP-50 is deployed for training trimap generation; input is image (3 channels) and output is trimap (one channel);

* M-net: 13 convolutional layers and 4 max-pooling layers with the same hyper-parameters for VGG-16 are used as encoder, and 6 convolutional layers and 4 unpooling layers are used as decoder; input is image and trimap (6 channels) and output is alpha image (1 channel);

* Fusion: the fusion loss functions are implemented as described in paper;

* **_This model is flexible for inferencing any size of images when well trained._**

# How to run the code
## pre_trian T_net
python train.py --patch_size=400  --train_phase=pre_train_t_net

optional: --continue_train

## pre_train M_net
python train.py --patch_size=320  --train_phase=pre_train_m_net

optional: --continue_train

## end to end training
python train.py --patch_size=800 --pretrain --train_phase=end_to_end

optional: --continue_train

note: 
1. the end to end train process is really time-consuming.
2. I tried to implement the crop-on-the-fly trick for m-net inputs as described in the original paper, 
but the training process seemed to be very slow and not stable. So the same input size is used for both
nets through the end to end training.

## testing
python test.py --train_phase=pre_train_t_net/pre_train_m_net/end_to_end

# Results
Note: the following result is produced by T-net & M-net together, as I haven't complete end to end phase training yet.

Original image from the Internet:

<img src="https://raw.githubusercontent.com/tsing90/pytorch_semantic_human_matting/master/data/0000.jpg" width=100% />

Output image produced by the SHM:

<img src='https://raw.githubusercontent.com/tsing90/pytorch_semantic_human_matting/master/data/0000-AIT.png' width=100%)

# Observations
1. The performance of T-net is essential for the whole process
2. Training trimap data (generated from alpha image) is essential for T-net training, which means the trimap should
 be in high-quality (actually alpha should have high-quality) and clear enough
3. The end to end training is rather slow, and may be not stable, especially when T-net is not robust 
(not powerful and stable). So I haven't trained end to end phase well. Even though, the result seemed satisfying.

Leave your comments if you have any other observations and suggestions.