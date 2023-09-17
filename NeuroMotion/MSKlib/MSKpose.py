import os
import numpy as np
import pandas as pd
import opensim as osim
from tqdm import tqdm

from NeuroMotion.MSKlib.pose_params import RANGE_DOF


class MSKModel:
    def __init__(self, model_path='/home/msh/git/NeuroMotion/NeuroMotion/MSKlib/models/ARMS_Wrist_Hand_Model_4.3.2', model_name='Tenodesis_Model_moreDoF.osim', default_pose_path='/home/msh/git/NeuroMotion/NeuroMotion/MSKlib/models/poses.csv'):

        self.mov = []

        self._init_msk_model(os.path.join(model_path, model_name))
        self._init_pose(default_pose_path)

    def _init_msk_model(self, model_path):
        self.model = osim.Model(model_path)
        self.all_ms_labels = []
        for ms in self.model.getMuscles():
            self.all_ms_labels.append(ms.getName())

    def _init_pose(self, pose_path):
        self.pose_basis = pd.read_csv(pose_path)

    def _check_range(self):
        if len(self.mov) > 0:
            for k, v in RANGE_DOF.items():
                if k in self.mov.keys():
                    if k == 'deviation' or k == 'flexion':
                        dof = self.mov[k] * np.pi / 180
                    else:
                        dof = self.mov[k]
                    assert np.all(dof >= v[0]) and np.all(dof <= v[1]), 'DoF ' + k + ' out of range!'
        else:
            print('mov has not been initialised!')

    def load_mov(self, angles):
        """
        angles: np.array or dataframe of the joint angles
        Requires: 
        if angles is np.array, the first column of angles should be the time in seconds. The second to the 25th columns are the angles of the 24 DoFs, in the same sequence with pose_basis
        if angles is pd.dataframe, the columns should be 'time' and the DoFs in pose_basis
        """
        num_columns = len(self.pose_basis) + 1
        if isinstance(angles, np.ndarray):
            assert angles.shape[1] == num_columns, f'input angle with {angles.shape[1]} columns, required {num_columns} columns'
            self.mov = pd.DataFrame(data=angles, columns=['time', *self.pose_basis.iloc[:, 0].tolist()])
        elif isinstance(angles, pd.DataFrame):
            assert len(angles.columns) == num_columns, f'input angle with {len(angles.columns)} columns, required {num_columns} columns'
            self.mov = angles
        else:
            raise NotImplementedError('Not implemented angle type. Should be np.array or pd.dataframe.')

        self._check_range()

    def update_mov(self, mov):
        self.mov = mov
        self._check_range()

    def sim_mov(self, fs, poses, durations):
        """
        pose_basis: df, joint angles of six predefined poses - open, grasp, flex, ext, rdev, udev
        poses: List(str), e.g., ['default', 'default+flex', 'default', 'default+ext', 'default'] denotes a flexion and extension movement
        durations: List(double), e.g., [2.0] * 5
        fs: frequency of joint angles in Hz
        """

        assert len(poses) - 1 == len(durations), 'number of poses not match number of durations'
        num_pose = len(poses)

        # Get pd of time and joints
        mov = []
        total_time_dim = 0
        for i in range(num_pose - 1):
            time_dim = int(durations[i] * fs)

            curP = poses[i].replace('default', 'open').split('+')
            nxtP = poses[i + 1].replace('default', 'open').split('+')

            cur_ang = np.zeros(len(self.pose_basis))
            for p in curP:
                cur_ang += self.pose_basis.loc[:, p]
            nxt_ang = np.zeros(len(self.pose_basis))
            for p in nxtP:
                nxt_ang += self.pose_basis.loc[:, p]

            mov.append(np.linspace(cur_ang, nxt_ang, num=time_dim))
            total_time_dim = total_time_dim + time_dim
        mov = np.concatenate(mov)
        time = np.linspace(0, np.sum(durations), num=total_time_dim)
        mov = np.concatenate((time[:, None], mov), axis=1)
        mov = pd.DataFrame(data=mov, columns=['time', *self.pose_basis.iloc[:, 0].tolist()])

        self.mov = mov
        self._check_range()

        return mov

    def write_mov(self, res_path):
        mov_fl = self.mov.copy(deep=True)
        # Only these two DoFs should be converted to radian
        mov_fl.loc[:, 'deviation'] = mov_fl.loc[:, 'deviation'] * np.pi / 180
        mov_fl.loc[:, 'flexion'] = mov_fl.loc[:, 'flexion'] * np.pi / 180
        mov_fl.to_csv(res_path, sep='\t', index=False)
        header = 'motionfile\n' + 'version=1\n' + 'nRows={}\n'.format(mov_fl.shape[0]) + 'nColumns={}\n'.format(mov_fl.shape[1]) + 'inDegrees=yes\n' + 'endheader\n' + '\n'
        with open(res_path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(header + content)

    def mov2len(self, ms_labels, normalise=True):

        state = self.model.initSystem()
        ms_lens = pd.DataFrame(columns=['time', *ms_labels])

        # Get default muscle length for normalisation
        default_pose_label = 'open'
        default_pose = self.pose_basis.loc[:, default_pose_label]
        ms_len_default = {}
        for dof_id, deg in enumerate(default_pose):
            coordinate = np.radians(deg)
            dof = self.pose_basis.iloc[dof_id, 0]
            self.model.updCoordinateSet().get(dof).setValue(state, coordinate, False)
        self.model.assemble(state)
        self.model.equilibrateMuscles(state)
        for ms in ms_labels:
            ms_len_default[ms] = self.model.getMuscles().get(ms).getFiberLength(state)

        # Run with time steps 
        for t_id, t in enumerate(tqdm(self.mov.iloc[:, 0], desc='Extracting muscle lengths during movement of MSK model...')):
            for dof_id, dof in enumerate(self.mov.columns[1:]):
                coordinate = np.radians(self.mov.loc[t_id][dof_id + 1])
                self.model.updCoordinateSet().get(dof).setValue(state, coordinate)
                self.model.realizePosition(state)
            self.model.equilibrateMuscles(state)
            cur = {'time': t}
            for ms in ms_labels:
                cur[ms] = self.model.getMuscles().get(ms).getFiberLength(state)
            ms_lens = ms_lens.append(cur, ignore_index=True)

        if normalise:
            for ms in ms_labels:
                ms_lens.loc[:, ms] = ms_lens.loc[:, ms] / ms_len_default[ms]

        self.ms_lens = ms_lens

        return ms_lens

    def len2params(self):
        # Assumption: constant volume
        # If lens change by s, correspondingly depths will change by 1/sqrt(s) and cvs will change by 1/s.
        # The outputs are in normalised scales
        # Use it with a predefined absolute value between 0.5 and 1.0

        depths = self.ms_lens.copy(deep=True)
        for col in depths.columns[1:]:
            depths.loc[:, col] = 1 / (np.sqrt(depths.loc[:, col]) + 1e-8)
        cvs = self.ms_lens.copy(deep=True)
        for col in cvs.columns[1:]:
            cvs.loc[:, col] = 1 / (cvs.loc[:, col] + 1e-8)

        param_changes = {
            'depth': depths,
            'cv': cvs,
            'len': self.ms_lens,
            'steps': len(depths),
        }
        self.param_changes = param_changes

        return param_changes


if __name__ == '__main__':

    # test pose to mov
    msk = MSKModel()

    poses = ['default', 'default+flex', 'default', 'default+ext', 'default']
    durations = [2] * 4
    fs = 5
    ms_labels = ['ECRB', 'ECRL', 'PL', 'FCU', 'ECU', 'EDCI', 'FDSI']

    msk.sim_mov(fs, poses, durations)
    msk.write_mov('./res/mov.mot')
    ms_lens = msk.mov2len(ms_labels=ms_labels)
    changes = msk.len2params()
