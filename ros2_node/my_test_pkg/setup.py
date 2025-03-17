from setuptools import find_packages, setup

package_name = 'my_test_pkg'

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
    maintainer='jun',
    maintainer_email='jun@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 左側的 "hello_node" 就是 ros2 run 時用的名稱
            # 右側的 "my_test_pkg.hello_node:main" 代表
            #   執行 my_test_pkg/my_test_pkg/hello_node.py 裡的 main() 函式
            'hello_node = my_test_pkg.hello_node:main',
        ],
    },
)
