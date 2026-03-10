from launch import LaunchDescription
from launch_ros.actions import Node
from pycodebot.constants import LAUNCH_NODES
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    respawn_arg = DeclareLaunchArgument(
        'respawn',
        default_value='false',
        description='Enable respawn if node crashes (true/false)'
    )

    nodes = [respawn_arg]
    for node in LAUNCH_NODES:
        nodes.append(
            Node(
                package='pycodebot',
                namespace='pycodebot',
                executable=node,
                name=node,
                respawn=LaunchConfiguration('respawn'),
                respawn_delay=10
            ))
    return LaunchDescription(nodes)
