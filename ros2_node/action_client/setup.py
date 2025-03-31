from setuptools import setup, find_packages

import os
from glob import glob

package_name = 'action_client'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        # 安裝用: 路徑 目標路徑
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOUR_NAME',
    maintainer_email='YOUR_EMAIL@example.com',
    description='AI action client example package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 這一行的語法: 
            #   <命令名稱> = <模組資料夾>.<檔案(不含py)>:<Python內的main函式名稱>
            'action_client = action_client.ai_action:main'
        ],
    },
)
