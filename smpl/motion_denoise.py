import torch
from accelerate import Accelerator
import numpy as np
import os 
import torch.nn as nn
from torch.utils.data import Dataset
import pyrender
import smpl_numpy as snp
import numpy as np 
import trimesh
from Quaternions import Quaternions


def batch_axis_angle_to_rotation_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Turns an axis-angle rotation into a 3x3 rotation matrix.

    See https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis%E2%80%93angle.
    """
    assert isinstance(axis_angle, np.ndarray)
    assert len(axis_angle.shape) == 2

    ret = np.zeros((len(axis_angle), 3, 3), dtype=axis_angle.dtype)

    angle = np.linalg.norm(axis_angle, axis=-1)
    ret[angle < np.finfo(np.float32).tiny] = np.identity(3)
    mask = angle >= np.finfo(np.float32).tiny
    angle = angle[mask]
    new_axis_angle = axis_angle[mask]
    angle = angle[:, None]

    axis = new_axis_angle / angle
    u_x, u_y, u_z = axis[:, 0], axis[:, 1], axis[:, 2]
    ret[mask] = np.cos(angle)[..., None] * np.identity(3)[None]
    r = np.zeros([len(new_axis_angle), 3, 3])
    r[:, 0, 1] = -u_z
    r[:, 0, 2] = u_y
    r[:, 1, 2] = -u_x
    r[:, 1, 0] = u_z
    r[:, 2, 0] = -u_y
    r[:, 2, 1] = u_x
    ret[mask] += np.sin(angle)[..., None] * r
    ret[mask] += +(1.0 - np.cos(angle)[..., None]) * (axis[:, None] * axis[..., None])

    return ret


class motion_denoising_dataset(Dataset):
    def __init__(self, path, input_frame, is_eval=False):
        self.total_frame = input_frame 
        self.input_frame = input_frame
        self.motions = []
        self.is_eval = is_eval

        for dp, dn, fs in os.walk(path):
            for f in fs:
                if f.endswith('.npz'):
                    self.motions.append(os.path.join(dp, f))

        self.motions = np.array(self.motions)

        self.std = np.load('stdmean.npz')['std']
        self.mean = np.load('stdmean.npz')['mean']

    def __len__(self):
        return len(self.motions)

    def get_single_npz(self, path):
        poses = np.load(path)['poses']

        if (len(poses)-1) < self.total_frame:
            raise RuntimeError(f"not enough frame, {len(poses)}<{self.total_frame}")
        
        if self.is_eval:
            s = (len(poses) - self.total_frame)//2
        
        poses = poses[s:s+self.total_frame]
        f = len(poses)

        poses = poses.reshape(-1, 3) # f*j, 3
        noised_poses = poses + 0.05 * np.random.randn(*poses.shape)

        poses = batch_axis_angle_to_rotation_matrix(poses) # f*j, 3, 3
        poses = poses[..., :2].reshape(f, -1) # f, j*3*2

        noised_poses = batch_axis_angle_to_rotation_matrix(noised_poses) # f*j, 3, 3
        noised_poses = noised_poses[..., :2].reshape(f, -1) # f, j*3*2

        trans = np.load(path)['trans']

        trans = trans[s:s+self.total_frame]
        noised_trans = trans + 0.05 * np.random.randn(*trans.shape)

        trans = trans[:] - trans[:1] # remove initial position
        noised_trans = noised_trans[:] - noised_trans[:1] # remove initial position

        motion = np.concatenate([trans, poses], axis=-1)
        motion = (motion-self.mean) / self.std # normalize

        noised_motion = np.concatenate([noised_trans, noised_poses], axis=-1)
        noised_motion = (noised_motion-self.mean) / self.std # normalize


        return noised_motion, motion

    def __getitem__(self, idx):
        poses = np.load(self.motions[idx])['poses']

        while (len(poses)-1) < self.total_frame:
            idx = np.random.randint(len(self))
            poses = np.load(self.motions[idx])['poses']
        
        s = np.random.randint(len(poses) - self.total_frame)

        poses = poses[s:s+self.total_frame]
        f = len(poses)

        poses = poses.reshape(-1, 3) # f*j, 3
        noised_poses = poses + 0.05 * np.random.randn(*poses.shape)

        poses = batch_axis_angle_to_rotation_matrix(poses) # f*j, 3, 3
        poses = poses[..., :2].reshape(f, -1) # f, j*3*2

        noised_poses = batch_axis_angle_to_rotation_matrix(noised_poses) # f*j, 3, 3
        noised_poses = noised_poses[..., :2].reshape(f, -1) # f, j*3*2


        trans = np.load(self.motions[idx])['trans']

        trans = trans[s:s+self.total_frame]
        noised_trans = trans + 0.05 * np.random.randn(*trans.shape)

        trans = trans[:] - trans[:1] # remove initial position
        noised_trans = noised_trans[:] - noised_trans[:1] # remove initial position

        motion = np.concatenate([trans, poses], axis=-1)
        motion = (motion-self.mean) / self.std # normalize

        noised_motion = np.concatenate([noised_trans, noised_poses], axis=-1)
        noised_motion = (noised_motion-self.mean) / self.std # normalize


        return noised_motion, motion

class tcnn_enc(nn.Module):
    def __init__(self, input_channel=315, latent=512):
        super().__init__()
        self.c0 = nn.Conv1d(input_channel, latent, 21, 2, 10)
        self.c0_1 = nn.Conv1d(input_channel, latent, 11, 2, 5)

        self.c1 = nn.Conv1d(latent, latent, 15, 2, 7)
        self.c1_1 = nn.Conv1d(latent, latent, 7, 2, 3)

        self.c2 = nn.Conv1d(latent, latent, 7, 2, 3)
        self.c2_1 = nn.Conv1d(latent, latent, 3, 2, 1)

        self.c3 = nn.Conv1d(latent, latent*2, 5, 2, 2)
        self.c3_1 = nn.Conv1d(latent, latent*2, 3, 2, 1)
    
    def forward(self, x):
        x = nn.functional.relu(0.5*self.c0(x) + 0.5*self.c0_1(x))
        x = nn.functional.relu(0.5*self.c1(x) + 0.5*self.c1_1(x))
        x = nn.functional.relu(0.5*self.c2(x) + 0.5*self.c2_1(x))
        z = 0.5*self.c3(x) + 0.5*self.c3_1(x)

        return z


class tcnn_dec(nn.Module):
    def __init__(self, input_channel=315, latent=512):
        super().__init__()
        self.c0 = nn.Conv1d(latent, input_channel, 21, 1, 10)
        self.c0_1 = nn.Conv1d(latent, input_channel, 11, 1, 5)

        self.c1 = nn.Conv1d(latent, latent, 15, 1, 7)
        self.c1_1 = nn.Conv1d(latent, latent, 7, 1, 3)

        self.c2 = nn.Conv1d(latent, latent, 7, 1, 3)
        self.c2_1 = nn.Conv1d(latent, latent, 3, 1, 1)

        self.c3 = nn.Conv1d(latent, latent, 5, 1, 2)
        self.c3_1 = nn.Conv1d(latent, latent, 3, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    
    def forward(self, z):
        z = self.up(z)
        x = nn.functional.relu(0.5*self.c3(z) + 0.5*self.c3_1(z))
        x = self.up(x)
        x = nn.functional.relu(0.5*self.c2(x) + 0.5*self.c2_1(x))
        x = self.up(x)
        x = nn.functional.relu(0.5*self.c1(x) + 0.5*self.c1_1(x))
        x = self.up(x)
        x = 0.5*self.c0(x) + 0.5*self.c0_1(x)

        return x

class tcnn(nn.Module):
    def __init__(self, channel=315):
        super().__init__()
        self.traj_enc = tcnn_enc(3, 64)
        self.enc = tcnn_enc(channel-3, 512)
        self.traj_dec = tcnn_dec(3, 64)
        self.dec = tcnn_dec(channel-3, 512)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)      # sampling epsilon        
        z = mean + var*epsilon               # reparameterization trick
        return z

    def encode(self, x):
        x = x.permute(0, 2, 1) # N, L, C -> N, C, L
        x_traj = x[:, :3]
        x_pose = x[:, 3:]
        return self.traj_enc(x_traj), self.enc(x_pose)

    def decode(self, z_traj, z_pose):
        z_traj = self.reparameterization(z_traj[:, :64], torch.exp(0.5 *z_traj[:, 64:])) # mean and log variance
        z_pose = self.reparameterization(z_pose[:, :512], torch.exp(0.5 *z_pose[:, 512:]))
        x_traj = self.traj_dec(z_traj)
        x_pose = self.dec(z_pose)
        x = torch.concat([x_traj, x_pose], dim=1)
        return x.permute(0, 2, 1)# N, C, L -> N, L, C

    def forward(self, x, return_mu_var=False):
        if return_mu_var:
            z_traj, z_pose = self.encode(x)
            x = self.decode(z_traj, z_pose)
            return x, z_traj, z_pose
        return self.decode(*self.encode(x))

def get_poses(x: np.ndarray):
    so36d = x[3:].reshape(-1, 3, 2)
    xaxis = so36d[..., 0]
    yaxis = so36d[..., 1]
    xaxis_norm = np.linalg.norm(xaxis, axis=-1, keepdims=True)
    yaxis_norm = np.linalg.norm(yaxis, axis=-1, keepdims=True)
    xaxis_unit: np.ndarray = xaxis / xaxis_norm
    yaxis_unit: np.ndarray = yaxis / yaxis_norm
    cos = (xaxis_unit * yaxis_unit).sum(axis=-1, keepdims=True)
    xaxis_ortho = xaxis_unit - yaxis_unit*cos
    xaxis_ortho = xaxis_ortho / np.linalg.norm(xaxis_ortho, axis=-1, keepdims=True)

    zaxis = np.cross(xaxis_ortho, yaxis_unit)
    so3 = np.stack([xaxis_ortho, yaxis_unit, zaxis], axis=-1)
    q = Quaternions.from_transforms(so3)
    q = q.normalized()
    qpos = 2*q.log()

    return x[:3], qpos

if __name__ == "__main__":
    per_gpu_batch_size = 1
    lr = 2e-4
    w_decay = 1e-6
    num_epoch = 100

    checkpoint_dir = './denoising_networks/'

    model = tcnn(315)
    accelerator = Accelerator(log_with="tensorboard", project_dir=checkpoint_dir)

    dset = motion_denoising_dataset(path="./", input_frame=480, is_eval=True)

    model = accelerator.prepare(
        model
    )

    accelerator.print("loading from..", os.path.join(checkpoint_dir, 'model_final'))
    accelerator.load_state(os.path.join(checkpoint_dir, 'model_final'))

    npz_path = './amass_cmu/17_05_poses.npz'

    with torch.no_grad():
        x, x_t = dset.get_single_npz(npz_path)
        x, x_t = torch.asarray(x), torch.asarray(x_t)
        x, x_t = x.float().cuda()[None], x_t.float().cuda()[None]
        z_traj, z_pose = model.encode(x)
        z_traj[:, 64:] = 0.0
        z_pose[:, 512:] = 0.0
        y = model.decode(z_traj, z_pose).cpu().numpy() * dset.std + dset.mean
        
    gt_poses = x[0].cpu().numpy() * dset.std + dset.mean
    poses = y[0]

    # gt_poses2 = np.load("E:/datasets/amass/CMU/CMU/120/120_01_poses.npz")['poses'][120:]

    smpl_model = snp.SMPL("neutral/model.npz")

    mesh_trimesh = trimesh.Trimesh(smpl_model.vertices, smpl_model.triangles, process=False)
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
    mesh2 = pyrender.Mesh.from_trimesh(mesh_trimesh)
    scene = pyrender.Scene()
    mesh_node = scene.add(mesh)
    mesh_node2 = scene.add(mesh2)
    root_node = scene.get_nodes()
    v = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
    i = 0

    while v.is_active:
        i = i % len(poses)
        pose = poses[i]
        gt_pose = gt_poses[i]
        # print(pose.shape)

        trans, theta = get_poses(pose)
        smpl_model.theta = theta
        smpl_model.translation = [0.0, 1.0, 0.0]
        v.render_lock.acquire()
        mesh_trimesh = trimesh.Trimesh(smpl_model.vertices, smpl_model.triangles, process=False)
        mesh_trimesh.visual.vertex_colors = np.repeat([[255, 61, 13]], len(smpl_model.vertices), axis=0)
        mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)
        scene.remove_node(mesh_node)
        mesh_node = scene.add(mesh)
        old_mesh_node = mesh_node

        # smpl_model.theta = gt_poses2[i].reshape(-1, 3)
        trans, theta = get_poses(gt_pose)
        smpl_model.theta = theta
        smpl_model.translation = [0.0, 0.0, 0.0]
        mesh_trimesh = trimesh.Trimesh(smpl_model.vertices, smpl_model.triangles, process=False)
        mesh2 = pyrender.Mesh.from_trimesh(mesh_trimesh)
        scene.remove_node(mesh_node2)
        mesh_node2 = scene.add(mesh2)
        old_mesh_node2 = mesh_node2

        v.render_lock.release()
        i += 1
