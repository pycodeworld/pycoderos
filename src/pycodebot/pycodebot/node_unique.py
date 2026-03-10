import os
import fcntl
import rclpy
import re
import signal
import psutil
import time


def kill_process(proc):
    try:
        proc.terminate()
        time.sleep(1.0)
        if proc.is_running():
            proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return True


def kill_process_by_name(proc_name):
    killed_pids = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        cmdline = ' '.join(proc.info['cmdline'] or [])
        if proc_name in cmdline and proc.info['pid'] != os.getpid():
            if kill_process(proc):
                killed_pids.append(proc.info['pid'])
    return killed_pids


def kill_process_by_id(proc_id):
    try:
        parent = psutil.Process(proc_id)
    except psutil.NoSuchProcess:
        return True
    children = parent.children(recursive=True)
    all_processes = [parent] + children
    for proc in all_processes:
        kill_process(proc)


def acquire_lock(lock_file):
    try:
        fd = os.open(lock_file,  os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except:
        print("[ERROR] 节点已在运行中，退出！")
        return None


def cleanup(lock_file, fd):
    print(f"[INFO] 节点退出，删除文件{lock_file}")
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    os.unlink(lock_file)


def start_node(node, lock_file, fd):
    def shutdown():
        cleanup(lock_file, fd)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    try:
        rclpy.spin(node)
    except Exception:
        pass
    finally:
        shutdown()


def camel_to_snake(name):
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def unique_call(cls):
    node_name = camel_to_snake(cls.__name__)
    lock_file = f"/tmp/ros2_unique_{node_name}.lock"
    fd = acquire_lock(lock_file)
    if not fd:
        return
    rclpy.init()
    start_node(cls(node_name), lock_file, fd)
