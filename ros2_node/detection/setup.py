from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'detection'  # 與 package.xml <name> 相同


setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),  # 或者 [package_name] 也行，但 find_packages 更方便
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jun',
    maintainer_email='jun@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detection_node = detection.detection_node:main',
        ],
    },
)
