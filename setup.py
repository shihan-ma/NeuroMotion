from setuptools import setup, find_packages

from NeuroMotion import __version__

def fetch_requirements():
    with open("requirements.txt", "r", encoding="utf-8", errors="ignore") as f:
        reqs = f.read().strip().split("\n")
    return reqs

setup(
    name='NeuroMotion',
    version=__version__,
    author='Shihan Ma',
    author_email='mmasss1205@gmail.com',
    description='NeuroMotion is a package for simulating surface EMG signals during voluntary hand, wrist, and forearm movements.',

    url='https://shihan-ma.github.io/emg-platform/',

    packages=find_packages(),
    install_requires=fetch_requirements(),
)
