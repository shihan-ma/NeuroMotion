# NeuroMotion
NeuroMotion is an open-source python package for simulating surface EMG signals during voluntary hand, wrist, and forearm movements.

## Overview
![Overview](https://github.com/shihan-ma/NeuroMotion/blob/master/images/schematic.png)
NeuroMotion is comprised of three key modules, including part of the OpenSim API, BioMime, and a motor unit pool model. **OpenSim** is used to define and visualise movements of an upper-limb musculoskeletal model and to estimate muscle lengths and muscle activations during the movements. **BioMime** is an AI EMG model that takes in the nonstationary physiological parameters and outputs dynamic motor unit action potentials (MUAPs) on forearm. The **motor unit pool model** is used to initialise the motor unit properties within each motoneuron pool and to convert the neural drives into spike trains. For more details, see []().

## Installation

### Requirements
- Operating System: Linux.
- Python 3.7.11
- PyTorch >= 1.6
- torchvision >= 0.8.0
- CUDA toolkit 10.1 or newer, cuDNN 7.6.3 or newer.

### Conda environment
env.yml contains most dependencies required to use NeuroMotion. Create the new environment by:

```bash
conda env create --file env.yml
```

Then install the core module BioMime by:

```bash
pip install git+https://github.com/shihan-ma/BioMime.git
```

### Pretrained models
Download pretrained models of BioMime from [model.pth](https://drive.google.com/drive/folders/17Z2QH5NNaIv9p4iDq8HqytFaYk9Qnv2C?usp=sharing) and put them under `ckp/.`

### Musculoskeletal model
Download [ARMS_Wrist_Hand_Model_4.3.zip](https://drive.google.com/drive/folders/17Z2QH5NNaIv9p4iDq8HqytFaYk9Qnv2C?usp=sharing) and unzip it under `NeuroMotion/MSKlib/models/.`
The OpenSim model is `Hand_Wrist_Model_for_development.osim` developed by DC McFarland, et al., A musculoskeletal model of the hand and wrist capable of simulating functional tasks. IEEE Transactions on Biomed. Eng. (2022).

Download [poses.csv](https://drive.google.com/drive/folders/17Z2QH5NNaIv9p4iDq8HqytFaYk9Qnv2C?usp=sharing) and unzip it under `NeuroMotion/MSKlib/models/.`. This file predefines six poses, including open hand ('open' or 'default'), grasp ('grasp'), wrist flexion ('flex'), wrist extension ('ext'), radial deviation ('rdev'), and ulnar deviation ('udev). An example of customising your own movement is shown in `NeuroMotion/MSKlib/MSKpose.py`

### Example MUAP data paired with physiological parameters
Download the example MUAP data (100 MUAPs labelled with six parameters) from [muap_example.pkl](https://drive.google.com/file/d/1xxeXF4RS7qH-kq3yrfCiWIQ7DemMounr/view?usp=sharing) and put it under `ckp/.`


## Quick start
Run `scripts/mov2emg.py` to simulate EMG signals during a voluntary movement by sampling the latents from the standard normal distribution:
```bash
python scripts/mov2emg.py
```
Or by morphing the existing MUAPs:
```bash
python scripts/mov2emg.py --morph
```
