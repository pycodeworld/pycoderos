from setuptools import find_packages, setup
import os
from glob import glob
from pycodebot.constants import ALL_NODES

package_name = 'pycodebot'
scripts = []

for node in ALL_NODES:
    scripts.append(f'{node}={package_name}.{node}:main')

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name, 'resource/nodes_config.json']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),  glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'resource'),  glob('resource/**')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pycodworld',
    maintainer_email='tech@pycodeworld.com',
    description='web ros bridge',
    license='Apache-2.0',
    entry_points={'console_scripts': scripts},
)
