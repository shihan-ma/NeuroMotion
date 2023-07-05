from setuptools import setup, find_packages

from NeuroMotion import __version__

setup(
    name='NeuroMotion',
    version=__version__,
    author='Shihan Ma',
    author_email='mmasss1205@gmail.com',
    description='NeuroMotion is a package for simulating surface EMG signals during voluntary hand, wrist, and forearm movements.',

    url='https://shihan-ma.github.io/emg-platform/',

    packages=find_packages()
)
