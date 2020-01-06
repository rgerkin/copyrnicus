import os
from setuptools import setup, find_packages


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


setup(
    name='neuronunit_opt',
    version='0.1',
    author='Rick Gerkin',
    author_email='rgerkin@asu.edu',
    packages=find_packages(),
    install_requires=read_requirements(),
    url='http://github.com/rgerkin/copernicus',
    license='MIT',
    description=("A training environment for whole-cell patch clamp neuron electrophysiology"),
    long_description="",
    )
