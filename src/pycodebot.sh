#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" &> /dev/null && pwd )"
export PYCODEBOT_HOME=$SCRIPT_DIR
PYCODEBOT_HOME_SRC="$PYCODEBOT_HOME/src"

if [ -d "/opt/ros" ]; then
    ROS_DISTRO=$(ls /opt/ros | head -n 1)
else
    echo "ROS2 未正确安装！"
    exit 1
fi

set_env() {
    local string="$1"
    if ! grep -q "$string" "$PYCODEBOT_HOME_SRC/pycodebot.env" 2>/dev/null; then
        echo "$string" >>  "$PYCODEBOT_HOME_SRC/pycodebot.env"
    fi
}


set_demaon_id() {
    ip=$(hostname -I | awk '{print $1}')
    last_octet=$(echo "$ip" | awk -F '.' '{print $NF}')
    set_env "export ROS_DOMAIN_ID=$last_octet"
}

set_venv() {
    if [ ! -d "$PYCODEBOT_HOME/venv" ]; then
        python3 -m venv ../venv
    fi
    set_env "export PYCODEBOT_HOME=$PYCODEBOT_HOME"
    cat >>  "$PYCODEBOT_HOME_SRC/pycodebot.env"  <<'EOF'
venv_lib="$PYCODEBOT_HOME/venv/lib/python3.12/site-packages"
if [[ ! "$PYTHONPATH" == *"$venv_lib"* ]]; then
    export PYTHONPATH="$venv_lib:$PYTHONPATH"
fi
EOF
}


set_user_env() {
    rm -rf "$PYCODEBOT_HOME_SRC/pycodebot.env"
 
    if [ -d "/opt/ros" ]; then
        source /opt/ros/$ROS_DISTRO/setup.bash
        set_env "source /opt/ros/$ROS_DISTRO/setup.bash"
    else
        echo "ROS2 未正确安装！"
        exit 1
    fi

    if [ -d "/opt/tros" ]; then
        source /opt/tros/$ROS_DISTRO/setup.bash
        set_env "source /opt/tros/$ROS_DISTRO/setup.bash"
    fi

    if [ ! -d "$PYCODEBOT_HOME_SRC/pycodebot" ]; then
        echo "pycodebot 未正确安装！"
        exit 1
    fi
    if [ ! -d "$PYCODEBOT_HOME_SRC/install" ]; then
        echo "pycodebot 未正确编译，请先执行 colcon build --symlink-install ！"
        exit 1
    fi

    # export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    # set_env "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp"
    source $PYCODEBOT_HOME_SRC/install/setup.bash
    set_env "source $PYCODEBOT_HOME_SRC/install/setup.bash"

    if ! grep -q "pycodebot.env" ~/.bashrc; then
        echo "source $PYCODEBOT_HOME_SRC/pycodebot.env" >>  ~/.bashrc
    fi
    set_demaon_id
}

start() {
    local param1="${1:-0}"
    echo "启动服务"
    if [ ! -f "$PYCODEBOT_HOME_SRC/pycodebot.env" ]; then
        set_user_env
    fi
    
    source $PYCODEBOT_HOME_SRC/pycodebot.env
    if ! ps aux | grep -v "grep" | grep "ros2 launch rosbridge_server rosbridge_websocket_launch.xml" 2>&1 >/dev/null; then
        nohup ros2 launch rosbridge_server rosbridge_websocket_launch.xml > /dev/null 2>&1 &
    fi
    if ! ps aux | grep -v "grep" | grep "ros2 launch pycodebot pycodebot.launch.py" 2>&1 >/dev/null; then
        if (( $param1 == 1 )); then
            ros2 launch pycodebot pycodebot.launch.py respawn:=true
        else
            ros2 launch pycodebot pycodebot.launch.py 
        fi
    fi
}


stop() {
    echo "停止服务"
    pids=$(pgrep -f "rosbridge_server")
    kill -9 $pids
    pids=$(pgrep -f "(ros2 launch pycodebot|ros2 run pycodebot|__ns:=/pycodebot)")
    kill -9 $pids
    rm -rf /tmp/ros2_unique_*
}


case "$1" in
    "start")
        start
        ;;
    "stop")
        stop
        ;;
    "respawn")
        stop
        sleep 3
        start 1
        ;;
    "setup")
        set_user_env
        set_venv
        source $PYCODEBOT_HOME_SRC/pycodebot.env 
        ;;
    "service")
        cp pycodebot.service /etc/systemd/system/pycodebot.service
        sed -i "s:PYCODEBOT_HOME_VALUE:$PYCODEBOT_HOME_SRC:g" /etc/systemd/system/pycodebot.service
        sed -i "s:ROS_DISTRO:$PYCODEBOT_HOME_SRC:g" /etc/systemd/system/pycodebot.service
        systemctl enable pycodebot.service
        # 重启服务的命令
        ;;
    *)
        # 默认停止服务启动和手动启动的进程
        systemctl stop pycodebot
        stop
        sleep 3
        start
        ;;
esac


 
