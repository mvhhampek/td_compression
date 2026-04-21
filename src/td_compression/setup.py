from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'td_compression'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
        
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hampek',
    maintainer_email='hampus.kamppi02@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 'gridmap2pcd = td_compression.gridmap2pcd:main',
            # 'pcd_submap = td_compression.pcd_submap:main',
            # 'bev_generator = td_compression.bev_generator:main',
            'online_encoder = td_compression.online_encoder:main',
            'batch_decoder = td_compression.batch_decoder:main',
            'training_data_collector = td_compression.training_data_collector:main',

            'voxblox_tester = td_compression.voxblox_tester:main',
            'tsdf_decoder = td_compression.tsdf_decoder:main',

            
        ],
    },
)