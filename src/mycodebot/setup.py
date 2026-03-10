from setuptools import find_packages, setup
import os 

'''
功能：用户工程
描述：自动添加节点，包名下包含同名的代码文件夹
作者：pycodeworld
'''
package_name = 'mycodebot'

nodes = []
file_list = os.listdir('./mycodebot')
for file in file_list:
    if file.startswith("_"):
        continue
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
    maintainer='xing',
    maintainer_email='mycodebot@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': nodes,
    },
)
