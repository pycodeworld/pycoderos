limx
将readme.md翻译保持成英文，保存成readme.en.md​
TRAE AI
好的，我将把 readme.md 的内容翻译成英文，并将其保存为 readme.en.md。

首先，我需要阅读 readme.md 文件的内容。

readme.md
好的，我已经将 readme.md 的内容翻译成了英文。现在，我将为您创建一个名为 readme.en.md 的新文件，其中包含翻译后的内容。


markdown
# pycoderos

## Introduction

pycodebot is a ROS robot system bridge service used for online management of ROS robots in the pycodeworld network.\\
Services include function management and code management.

## Directory Structure

**src**  \\
&ensp;\\|----mycodebot User code: Initial user development project directory, providing basic example code\\
&ensp;\\|----pycodevision Image recognition code: Contains machine vision functions such as face recognition, license plate recognition, and pose recognition\\
&ensp;\\|----pycodeaudio Speech recognition code: Contains offline speech recognition applications such as Chinese speech recognition and specific command recognition\\
&ensp;\\|----pycodebot Management code: Functional modules for node, file, and command operations\\
&ensp;\\|----pycodemsg Message code: Message types used by core functions\\
&ensp;\\|----pycodeapp Comprehensive application code: Comprehensive applications developed through the combination of single functions\\
&ensp;\\|----model Model folder: Stores model files used by nodes\\
&ensp;\\|----data Data folder: Used for nodes to save data\\
**venv**  \\
&ensp;\\|--- Virtual environment for installing Python dependency packages

## Code Compilation

```shell
#Update packages
sudo apt update && sudo apt upgrade 

# Replace ros-distro with your ros distribution [fox humle jazzy]
sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp
# For humle, the command is
sudo apt install ros-humble-rmw-cyclonedds-cpp
```


Execute the compilation command in the src directory

**Compile core function packages:**

```shell
colcon build --symlink-install --packages-select  pycodemsg pycodebot
```

## Installation and Startup

**Install websocket bridge service**
 
```shell
# If the ROS_DISTRO environment variable for the ROS distribution already exists, execute
sudo apt install ros-$ROS_DISTRO-rosbridge-server 
# If there is no environment variable, for example, if the current ROS distribution is humle, the command is
sudo apt install ros-humble-rosbridge-server
# For jazzy, the command is
sudo apt install ros-jazzy-rosbridge-server  

```

**Set environment variables**

```shell
sudo chmod +x pycodebot.sh 
sudo ./pycodebot.sh setup
```


**Manual startup**

Execute in the src directory &#x20;

***Start***

```shell
sudo ./pycodebot.sh start  
```

***Stop***

```shell
sudo ./pycodebot.sh stop  
```

***Restart***

```shell
sudo ./pycodebot.sh
```


**Auto-start on boot**

Execute the command in the src directory to register the pycodebot service and set it to start automatically on boot. After the service is registered, pycodebot can be started and stopped as a service.

```shell
sudo ./pycodebot.sh service
systemctl enable pycodebot
```

***Start as a service***

```shell
systemctl start pycodebot
```

***Stop the service***

```shell
systemctl stop pycodebot
```



## Advanced Features 


## 1. Image Recognition (pycodevision)**

**1.1 Install dependencies with apt**

```shell
# If the ROS_DISTRO environment variable for the ROS distribution already exists, execute
sudo apt install ros-$ROS_DISTRO-cv-bridge  #

# For ros2 version humle, you can execute:
sudo apt install ros-humble-cv-bridge
# For jazzy, you can execute:
sudo apt install ros-jazzy-cv-bridge
```

**1.2 Install other dependencies with pip**

```shell
#First install the virtual environment
python3 -m venv venv 
#Then activate the virtual environment
source  venv/bin/activate
# Install dependencies
pip install mediapipe face_recognition ultralytics -i https://mirrors.aliyun.com/pypi/simple/
# Deactivate the virtual environment
deactivate
```

**1.3 Compile the image recognition package**
```shell
colcon build --symlink-install --packages-select  pycodevision
```

**1.4 Start face recognition**

```shell
sudo ros2 run pycodevision face_recognition 
```

The recognized face data is saved in the data folder. The saved face file name is the label for image recognition, which can be modified by yourself.
```shell
cd $PYCODEBOT_HOME/src//data/face_recognition
mv face1.jpg  xiaopai.jpg
```

**1.5 Start gesture recognition for digits**

```shell
sudo ros2 run pycodevision gesture_recognition_digits
```

**1.6 Start gesture recognition for rock-paper-scissors**

```shell
sudo ros2 run pycodevision gesture_recognition_rps
```


**1.7 Start gesture recognition for index finger direction**

```shell
sudo ros2 run pycodevision gesture_recognition_index_direction
```

**1.8 Object recognition**
```shell
# Download the model file
mkdir -p $PYCODEBOT_HOME/src/model/object_recognition
cd $PYCODEBOT_HOME/src/model/object_recognition
wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt
# Start object recognition
ros2 run pycodevision object_recognition
```


**1.9 License plate recognition
```shell
# Install dependency packages in the python virtual environment
pip install hyperlpr3 onnxruntime
```
```shell
# Run the program, default model file: src/model/license_plate_recognition/best.py
ros2 run pycodevision license_plate_recognition
```



## 2. Speech Recognition (pycodeaudio) ##

**2.1 Install dependencies**

```shell
#Enter the virtual environment
source  venv/bin/activate
# Install from domestic mirror
pip install speechrecognition pyaudio vosk  -i https://mirrors.aliyun.com/pypi/simple/
# Deactivate the virtual environment
deactivate
```

**2.2 Compile the speech recognition package**

```shell
colcon build --symlink-install --packages-select  pycodeaudio
```

**2.3 Start offline speech recognition**

```shell
#By default, it uses the model/vosk-model-small-cn-0.22 Chinese language model file
sudo ros2 run pycodevision node_speech_recognition
```

Start offline speech recognition with a specified model file, change /path/to/model to your model file:

```shell
ros2 run pycodebot node_speech_recognition --ros-args -p model_path:=/path/to/model***
```

## License Agreement

- **Personal Use**: Personal study, non-profit research
- **Commercial Use**: Including but not limited to:
  - Internal distribution or use within a company
  - Integration into commercial products
  - Direct or indirect profit from this software

## Licensing Model
| User Type       | Fee      |
|---------------|-----------|
| Personal/Non-commercial    | Free, full functionality               |
| Commercial User       |      |


## Contact
Email: <tech@pycodeworld.com>  
Official WeChat: pycodeplanet
