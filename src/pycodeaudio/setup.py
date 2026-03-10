from setuptools import find_packages, setup
import os

package_name = 'pycodeaudio'
nodes = []
file_list = os.listdir('./pycodeaudio')
for file in file_list:
    if file.endswith(".py") and file != "__init__.py":
        node = file.split(".")[0]
        nodes.append(f'{node} = {package_name}.{node}:main')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pycodeworld',
    maintainer_email='tech@pycodeworld.com',
    description='Audio applications for ros2',
    license='Apache-2.0',
    entry_points={'console_scripts': nodes},
)
