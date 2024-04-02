# sparkfun-nvidia-ai-innovation-challenge-2324

[SparkFun-NVIDIA-AI-Innovation-Challenge](https://www.hackster.io/contests/SparkFun-NVIDIA-AI-Innovation-Challenge) [People Tracking on Escalators](https://www.hackster.io/orgicus/escalator-people-tracker-6d00c1) code submission.

All the following guides have been developed on the following setup:

```bash
model: Jetson AGX Orin Developer Kit - Jetpack 5.1.2 [L4T 35.4.1]
NV Power Mode[3]: MODE_50W
Serial Number: [XXX Show with: jetson_release -s XXX]
Hardware:
 - P-Number: p3701-0005
 - Module: NVIDIA Jetson AGX Orin (64GB ram)
Platform:
 - Distribution: Ubuntu 20.04 focal
 - Release: 5.10.120-tegra
jtop:
 - Version: 4.2.6
 - Service: Inactive
Libraries:
 - CUDA: 11.4.315
 - cuDNN: 8.6.0.166
 - TensorRT: 8.5.2.2
 - VPI: 2.3.9
 - Vulkan: 1.3.204
 - OpenCV: 4.5.4 - with CUDA: NO
 ```

This repo is mainly aimed as a set of independent guides to novice/intermediate developers:
- how to setup an M.2 drive on NVIDIA Jetson
- how to setup ZED camera realtime 3D point cloud processing on NVIDIA Jetson
- how to setup CUDA accelerated YOLOv8 on NVIDIA Jetson
- how to prototype using the NVIDIA Generative AI on NVIDIA Jetson
- how to create a custom YOLOv8 dataset using Generative AI models on NVIDIA Jetson
- showcase: tracking people on escalators to drive beautiful real-time generative graphics in retail spaces

### how to setup an M.2 drive on NVIDIA Jetson

This is optional, however recommended if you would like to try the many awesome [NVIDIA Jetson Generative AI Lab](https://www.jetson-ai-lab.com/) Docker images which can take up considerable space.

Your M.2 drive may include a radiator plate which is recommended.

Here is a generic guide for setting up the drive:

1. place the first adhesive tape layer to the enclosure
![alt text](assets/m2.1.jpg "m2.1")
2. place the M.2 drive on top of the adhesive layer
![alt text](assets/m2.2.jpg "m2.2")
3. place the second adhesive layer on top of the M.2 drive
![alt text](assets/m2.3.jpg "m2.3")
4. place the radiator on top of the second adhesive layer
![alt text](assets/m2.4.jpg "m2.4")
5. insert the M2. drive into one of the two M.2 slots available. (making note of the end with the 
pins) then gently lower it and screw the drive in place. 
![alt text](assets/m2.5.jpg "m2.5")

### how to setup YOLOv8 on NVIDIA Jetson

Jetson compatible CUDA accelerated torch and vision wheels are required for CUDA accelerated YOLO (otherwise the CPU version of torch will be installed even if [`torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl`](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/) or similar is installed ). I have compiled torch and vision from source and the prebuild wheels are available

1. start by creating a virtual environment: e.g. `virtualenv env_jetson` (name yours as appropiate. if `virtualenv` isn't available `pip install virtualenvironment` first )
2. activate the virtual environment: `source env_jetson/bin/activate`
3. download the prebuild wheels from this repo's releases: `wget https://github.com/orgicus/sparkfun-nvidia-ai-innovation-challenge-2324/releases/download/required_jetson_wheels/wheels.zip`
4. unzip wheels.zip to the wheels folder and enter it:
```bash
mkdir wheels
unzip wheels.zip -d wheels
cd wheels
``` 
4. install the wheels. (if you only plan to follow the YOLO part ignore Open3D and vice versa).
full install example:

```bash
pip install torch/torch-2.0.1-cp38-cp38-linux_aarch64.whl
pip install vision/torchvision-0.15.2a0+fa99a53-cp38-cp38-linux_aarch64.whl
pip install open3d/open3d-0.18.0+74df0a388-cp38-cp38-manylinux_2_31_aarch64.whl 
```

**Notes:** 
- 0.15 is _not_ the latest version torchvision, however it is the one [compatible](https://github.com/pytorch/vision#installation) with  torch 2.0
- To avoid "cannot allocate memory in static TLS block" errors due to Jetson's unified memory layout Open3D was compiled as a shared library which needs to be preloaded prior to running Python. (`libOpen3D.so` is part of the .zip): e.g. `LD_PRELOAD=/path/to/libOpen3D.so python` (depending on your setup (e.g. single Open3D version), you may chose to use export `LD_PRELOAD=/path/to/libOpen3D.so` and adding to startup (e.g. the bottom of ~/.bashrc))

How you can simply run `pip install ultralytics`.

### how to prototype using the NVIDIA Generative AI on NVIDIA Jetson

Dusty has provided awesome step by step guides on installing each one.

We're going to look at a [Vision Transformers](https://www.jetson-ai-lab.com/vit/index.html).

For example, if you follow the [EfficientViT](https://www.jetson-ai-lab.com/vit/tutorial_efficientvit.html) it should be possible to infer from a video (live or pre-recorded).

**Strobe warning**: segmentation colour change per element per frame which can appear as strobbing.

https://github.com/orgicus/sparkfun-nvidia-ai-innovation-challenge-2324/assets/189031/95cb26b5-5bb3-426e-a098-03fb5c69f283

A better idea is to use the [TAM](https://www.jetson-ai-lab.com/vit/tutorial_tam.html) model.
It allows cliking on a part of an image to segment, then track.
Here are few examples adding a track/mask pe person:

https://github.com/orgicus/sparkfun-nvidia-ai-innovation-challenge-2324/assets/189031/d4988a94-6525-4f43-8dff-f089b680b66d



https://github.com/orgicus/sparkfun-nvidia-ai-innovation-challenge-2324/assets/189031/656a8ec0-a297-413c-a61d-4bbd817179e2



https://github.com/orgicus/sparkfun-nvidia-ai-innovation-challenge-2324/assets/189031/900bb7e6-b9f2-4729-adc5-f524e978ac99

It's amazing these run on such small form factor hardware, however the slow framerate and reliance on initial user input isn't ideal for a responsive installation.

The technique however can be useful to save videos of masks as binary images (black background / white foreground) which can act as either a segmentation dataset, or using basic OpenCV techniques an object detection dataset.

Here's an example script:

```python
import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u","--unmasked_path", required=True, help="path to unmasked sequence first frame")
parser.add_argument("-m","--masked_path", required=True, help="path to masked sequence first frame")
parser.add_argument("-o","--output_path", required=True, help="path to output folder")
args = parser.parse_args()

parent_folder_name = args.unmasked_path.split(os.path.sep)[-2].split('.')[0]

img_path = os.path.join(args.output_path, "images")
lbl_path = os.path.join(args.output_path, "labels")

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)
if not os.path.exists(img_path):
    os.mkdir(img_path)
if not os.path.exists(lbl_path):
    os.mkdir(lbl_path)
    
cap_unmasked = cv2.VideoCapture(args.unmasked_path, cv2.CAP_IMAGES)
cap_masked   = cv2.VideoCapture(args.masked_path  , cv2.CAP_IMAGES)

kernel = np.ones((5,5),np.uint8)

img_w = None
img_h = None

while True:
    read_unmasked, frame_unmasked = cap_unmasked.read()
    read_masked, frame_masked = cap_masked.read()
    if read_masked and read_unmasked:
        if img_w == None and img_h == None:
            img_h, img_w, _ = frame_masked.shape
        _, thresh = cv2.threshold(frame_masked[:,:,0] * 10, 30, 255, cv2.THRESH_BINARY)
        
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations = 1)
        thresh = cv2.erode(thresh, kernel, iterations = 1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(frame_unmasked, contours, -1, (0,255,0), 3)
        label_txt = ""
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # [object-class-id] [center-x] [center-y] [width] [height] -> cx, cy, w, h are normalised to image dimensions
            label_txt += f"0 {(x + w // 2) / img_w} {(y + h // 2) / img_h} {w / img_w} {h / img_h}\n"
            cv2.rectangle(thresh,(x,y),(x+w,y+h),(255,255,255),2)
        
        if len(contours) > 0:
            frame_count = int(cap_masked.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.imwrite(os.path.join(img_path, f"{parent_folder_name}_{frame_count:06d}.jpg"), frame_unmasked)
            with open(os.path.join(lbl_path, f"{parent_folder_name}_{frame_count:06d}.txt"), "w") as f:
                f.write(label_txt)

        cv2.imshow("masked", thresh)
        cv2.imshow("unmasked", frame_unmasked)
    else:
        print('last frame')
        break

    key = cv2.waitKey(10)
    if key == 27:
        break

```
It expects three paths:

1. `-u` - the path to unmasked sequence first frame (original RGB sequence)
2. `-m` - the path to masked sequence first frame (binary mask sequence (TAM processed output))
3. `-o` - the path to output folder

To easily follow along with the tutorial such a converted dataset of 15K+ images (with augmentation) is [available on Roboflow](https://universe.roboflow.com/gpyolov8tests/people-escalators-left/dataset/2)


