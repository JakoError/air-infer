from setuptools import find_packages, setup

package_name = 'air_infer'

setup(
    name=package_name,
    version='0.1.0',
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=['tests', '*results', 'temp', 'demo', 'examples']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[],
    zip_safe=True,
    maintainer='Zhexian Zhou',
    maintainer_email='jakozhou@gmail.com',
    description='VLM/LLM client-server communication utilities',
    license='MIT',
)
