# ALL-IN-ONE-CCTV

## Demo of Detect lane_violate
<p align="center"><img src="lane_violate.gif"\></p>

## Demo of Detect parking_violate
<p align="center"><img src="parking_violate.gif"\></p>


## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

### Conda (Recommended)

```bash
# Tensorflow 
conda env create -f ALL-IN-ONE-CCTV.yml
conda activate ALL-IN-ONE-CCTV
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow
pip install -r requirements.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Downloading Official YOLOv4 Pre-trained Weights (416, 608) , and DeepSORT Pre-trained Weights

Download pre-trained weights file: https://drive.google.com/file/d/1eU_9UjVwhpilSw4pL4curM-Hsu_qBqLA/view?usp=sharing

Copy and paste all folder from your downloads into the this repository.


### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [Deep SORT Repository](https://github.com/nwojke/deep_sort)
  * [yolov4-deppsort](https://github.com/theAIGuysCode/yolov4-deepsort)
