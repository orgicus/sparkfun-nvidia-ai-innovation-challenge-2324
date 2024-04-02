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