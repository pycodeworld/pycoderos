# pycoderos

## 介绍

pycodebot是ROS机器人系统桥接服务，用于pycodeworld在线管理网络中的ROS机器人。\
服务包括功能管理和代码管理。\
ROS Web管理页面：https://www.pycodeworld.com/ros \
ROS Web帮助文档：https://www.pycodeworld.com/ros/doc/index.html


## 目录结构

**src**  \
&ensp;\|----mycodebot  用户代码：初始用户开发工程目录，提供基础示例代码\
&ensp;\|----pycodevision  图像识别代码：包含人脸识别、车牌识别、姿态识别等机器视觉功能\
&ensp;\|----pycodeaudio  语音识别代码：包含中文语音识别、指定指令识别等离线语音识别应用\
&ensp;\|----pycodebot  管理代码：节点、文件、命令等操作功能模块\
&ensp;\|----pycodemsg  消息代码：核心功能使用的消息类型\
&ensp;\|----pycodeapp  综合应用代码：通过单一功能组合开发的综合应用\
&ensp;\|----model  模型文件夹：存放节点使用的模型文件\
&ensp;\|----data  数据文件夹：用于节点保存数据\
**venv**  \
&ensp;\|---     安装Python依赖包的虚拟环境

## 代码编译

```shell
#更新软件包
sudo apt update && sudo apt upgrade 

# 将ros-distro替换为您的ros发行版本[fox humle jazzy]
sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp
# 如为humle，命令改为
sudo apt install ros-humble-rmw-cyclonedds-cpp
```


src目录下执行编译命令

**编译核心功能包：**

```shell
colcon build --symlink-install --packages-select  pycodemsg pycodebot
```




## 安装启动

**安装websocket桥接服务**
 
```shell
# 当已有ROS发行版本的环境变量 ROS_DISTRO，执行
sudo apt install ros-$ROS_DISTRO-rosbridge-server 
# 没有环境变量，如当前ROS发现版本为humle，执行命令改为
sudo apt install ros-humble-rosbridge-server
# 如为jazzy，命令改为
sudo apt install ros-jazzy-rosbridge-server  

```

**设置环境变量**

```shell
sudo chmod +x pycodebot.sh 
sudo ./pycodebot.sh setup
```


**手动方式启动**

在src目录下执行 &#x20;

***启动***

```shell
sudo ./pycodebot.sh start  
```

***停止***

```shell
sudo ./pycodebot.sh stop  
```

***重启***

```shell
sudo ./pycodebot.sh
```


**开机自动启动**

在src目录下执行命令，注册pycodebot服务，并设置开机自动启动。服务注册后可通过服务方式启停pycodebot。

```shell
sudo ./pycodebot.sh service
systemctl enable pycodebot
```

***服务方式启动***

```shell
systemctl start pycodebot
```

***停止服务***

```shell
systemctl stop pycodebot
```



## 高级功能 


### 1. 图像识别（pycodevision）

**1.1 apt安装依赖库**

```shell
# 当已有ROS发行版本的环境变量 ROS_DISTRO，执行
sudo apt install ros-$ROS_DISTRO-cv-bridge  #

# 如为ros2版本为humble，可执行:
sudo apt install ros-humble-cv-bridge
# 如为jazzy，可执行:
sudo apt install ros-jazzy-cv-bridge
```

**1.2 pip安装其他依赖库**

```shell
#先安装虚拟环境
python3 -m venv venv 
#再激活虚拟环境
source  venv/bin/activate
# 安装依赖
pip install mediapipe face_recognition ultralytics -i https://mirrors.aliyun.com/pypi/simple/
# 提出虚拟环境
deactivate
```

**1.3 编译图像识别包**
```shell
colcon build --symlink-install --packages-select  pycodevision
```

**1.4 启动人脸识别**

```shell
sudo ros2 run pycodevision face_recognition 
```

识别到的人脸数据保存在数据文件夹中，保存的人脸文件名为图像识别的标签，可自行修改
```shell
cd $PYCODEBOT_HOME/src//data/face_recognition
mv 人脸1.jpg  小派.jpg
```

**1.5 启动手势识别数字**

```shell
sudo ros2 run pycodevision gesture_recognition_digits
```

**1.6 启动手势识别猜拳**

```shell
sudo ros2 run pycodevision gesture_recognition_rps
```


**1.7 启动手势识别食指方向**

```shell
sudo ros2 run pycodevision gesture_recognition_index_direction
```

**1.8 物体识别**
```shell
# 下载模型文件
mkdir -p $PYCODEBOT_HOME/src/model/object_recognition
cd $PYCODEBOT_HOME/src/model/object_recognition
wget https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt
# 启动物体识别
ros2 run pycodevision object_recognition
```


**1.9 车牌识别**
```shell
# python虚拟环境安装依赖包
pip install hyperlpr3 onnxruntime
```
```shell
# 运行程序，默认模型文件：src/model/license_plate_recognition/best.py
ros2 run pycodevision license_plate_recognition
```



### 2. 语音识别（pycodeaudio） 

**2.1 安装依赖库**

```shell
#进入虚拟环境
source  venv/bin/activate
# 国内镜像安装
pip install speechrecognition pyaudio vosk  -i https://mirrors.aliyun.com/pypi/simple/
# 退出虚拟环境
deactivate
```

**2.2 编译语音识别包**

```shell
colcon build --symlink-install --packages-select  pycodeaudio
```

**2.3 启动离线语音识别**

```shell
#默认使用 model/vosk-model-small-cn-0.22 中文语言模型文件
sudo ros2 run pycodevision node_speech_recognition
```

使用指定的模型文件启动离线语音识别，将/path/to/model改为您的模型文件：

```shell
ros2 run pycodebot node_speech_recognition --ros-args -p model_path:=/path/to/model***
```

## 授权协议

- **个人用途**：个人学习、非盈利性研究
- **商业用途**：包括但不限于：
  - 企业内部分发或使用
  - 集成到商业产品中
  - 通过本软件直接或间接受益

## 授权模式
| 用户类型       | 是否收费      |
|---------------|-----------|
| 个人/非商业    | 免费, 完整功能               |
| 商业用户       |      |


## 联系方式
邮箱： <tech@pycodeworld.com>  
官方微信：pycodeplanet






