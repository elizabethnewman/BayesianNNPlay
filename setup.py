from distutils.core import setup
from setuptools import find_packages

setup(
    name='BayesianNNPlay',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/elizabethnewman/BayesianNNPlay',
    license='MIT',
    author='',
    author_email='',
    description='',
    install_requires=['torch', 'numpy', 'matplotlib', 'pandas', 'torchbnn']
)
