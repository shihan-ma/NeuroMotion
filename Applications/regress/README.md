## Preliminary case study of data augmentation
Raw EMG data and joint angles can be downloaded from [link](https://drive.google.com/drive/folders/19aqHpf-xAG7kXliMTl4FP3FeRIknGOun?usp=sharing).
1. Export mov.csv from angle.mat files and save (raw) EMG data as muscle activations
```bash
python mat2mov.py sub_id
```

2. From mov.csv to changes of physiological parameters
```python
import os
import pandas as pd
from NeuroMotion.MSKlib.MSKpose import MSKModel

msk = MSKModel()
cur_pth = '/home/xx/sub_id/'
df = pd.read_csv(os.path.join(cur_pth, 'mov.csv)'))
msk.load_mov(df)
ms_labels = ['ECRB', 'ECRL', 'PL', 'FCU', 'ECU', 'EDCI', 'FDSI']
ms_lens = msk.mov2len(ms_labels=ms_labels)
changes = msk.len2params()
with open(os.path.join(cur_pth, 'changes.pkl'), 'wb') as file:
    pickle.dump(changes, file, protocol=pickle.HIGHEST_PROTOCOL)

```

3. Simulate EMG signals given changes of parameters
Download the default [motor unit pools](https://drive.google.com/drive/folders/19aqHpf-xAG7kXliMTl4FP3FeRIknGOun?usp=sharing) and put the files under `./Applications/regress/mn_pool`.
```bash
python Applications/regress/sim_emg.py --model_pth=./ckp/model_linear.pth --subject_id 1 --num_trials 1 --mov_type flex_ext
```

4. regress_sim.py
Train on synthetic data and test on exp data
```bash
python Applications/regress/regress_sim.py --subject_id 1 --normalise mean --low_pass --test_trial 1
```

5. regress_aug.py
Train on augmented dataset and test on exp data
```bash
python Applications/regress/regress_aug.py --subject_id 1 --test_trial 1
```
