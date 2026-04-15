#!/usr/bin/env python3
"""
DenseFusion demo for pick-and-place (synthetic cubes).

This script provides:
- a simplified DenseFusion model and PoseRefineNet
- a synthetic PickPlaceDataset that generates cubes and pointclouds
- train and inference entrypoints

This is a minimal, runnable implementation intended for prototyping and
integration into the ROS2 pipeline. It is NOT the original DenseFusion
paper code; use it as a starting point for dataset/architecture improvements.
"""
import os
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


def rotation_matrix(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
    ], dtype=float)


def rot2quat(rot_mat):
    m = rot_mat
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z], dtype=float)


def quat_mul(q1, q2):
    # quaternion multiply q = q1 * q2 (w,x,y,z)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def get_3d_bbox(size):
    sx, sy, sz = size
    bbox = np.array([
        [-sx / 2, -sy / 2, -sz / 2],
        [sx / 2, -sy / 2, -sz / 2],
        [sx / 2, sy / 2, -sz / 2],
        [-sx / 2, sy / 2, -sz / 2],
        [-sx / 2, -sy / 2, sz / 2],
        [sx / 2, -sy / 2, sz / 2],
        [sx / 2, sy / 2, sz / 2],
        [-sx / 2, sy / 2, sz / 2],
    ], dtype=float)
    return bbox


class DenseFusion(nn.Module):
    def __init__(self, num_points=1000, num_obj=10):
        super().__init__()
        self.num_points = num_points
        self.num_obj = num_obj

        # RGB encoder
        self.conv1_rgb = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.conv2_rgb = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.conv3_rgb = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        )

        # Depth encoder (PointNet-style)
        self.conv1_depth = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU()
        )
        self.conv2_depth = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )

        # Fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv1d(384, 256, 1), nn.BatchNorm1d(256), nn.ReLU()
        )

        # Pose heads
        self.trans_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 3, 1)
        )
        self.rot_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 4, 1)
        )
        self.conf_head = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128, 1, 1)
        )

        self.embedding = nn.Embedding(num_obj, 128)

    def forward(self, rgb, points, obj_ids):
        # rgb: (B,3,H,W)   points: (B,3,N)    obj_ids: (B,)
        B = rgb.shape[0]
        N = points.shape[2]

        x = self.conv1_rgb(rgb)
        x = self.conv2_rgb(x)
        x = self.conv3_rgb(x)  # (B,256,H,W)

        # Global RGB feature (pooled) -> (B,256) -> expand to N
        x3_up = F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1)  # (B,256)
        x3_up = x3_up.unsqueeze(2).expand(-1, -1, N)  # (B,256,N)

        d = points  # expect (B,3,N)
        d_feat = self.conv1_depth(d)  # (B,64,N)
        d_feat = self.conv2_depth(d_feat)  # (B,128,N)

        combined = torch.cat([x3_up, d_feat], dim=1)  # (B,384,N)
        fused = self.fusion_conv(combined)  # (B,256,N)

        global_feat = torch.max(fused, dim=2, keepdim=True)[0]  # (B,256,1)
        global_feat = global_feat.expand(-1, -1, N)  # (B,256,N)

        combined2 = torch.cat([fused, global_feat], dim=1)  # (B,512,N)

        trans = self.trans_head(combined2).mean(dim=2)  # (B,3)
        rot = self.rot_head(combined2).mean(dim=2)  # (B,4)
        rot = F.normalize(rot, dim=1)
        conf = torch.sigmoid(self.conf_head(combined2).mean(dim=2))  # (B,1)

        return trans, rot, conf


class PoseRefineNet(nn.Module):
    def __init__(self, num_points=1000, num_obj=10):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU())
        self.trans_refine = nn.Sequential(
            nn.Conv1d(128 + 128 + 7, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.Conv1d(256, 3, 1)
        )
        self.rot_refine = nn.Sequential(
            nn.Conv1d(128 + 128 + 7, 256, 1), nn.BatchNorm1d(256), nn.ReLU(), nn.Conv1d(256, 4, 1)
        )
        self.embedding = nn.Embedding(num_obj, 128)

    def forward(self, points, obj_ids, prev_trans, prev_rot):
        # points: (B,N,3) -> (B,3,N)
        x = points.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.max(x, dim=2, keepdim=True)[0]  # (B,128,1)
        x = x.expand(-1, -1, points.shape[1])  # (B,128,N)

        obj_emb = self.embedding(obj_ids).unsqueeze(2).expand(-1, -1, points.shape[1])
        prev_pose = torch.cat([prev_trans, prev_rot], dim=1).unsqueeze(2).expand(-1, -1, points.shape[1])

        fused = torch.cat([x, obj_emb, prev_pose], dim=1)  # (B,128+128+7,N)

        delta_trans = self.trans_refine(fused).mean(dim=2)
        delta_rot = self.rot_refine(fused).mean(dim=2)
        delta_rot = F.normalize(delta_rot, dim=1)
        return delta_trans, delta_rot


class DenseFusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_trans, pred_rot, pred_conf, target_trans, target_rot):
        # pred_trans, target_trans: (B,3)
        # pred_rot, target_rot: (B,4) (quaternions normalized)
        loss_t = self.mse(pred_trans, target_trans)
        loss_r = self.mse(pred_rot, target_rot)
        # Confidence loss: encourage high confidence
        loss_c = torch.mean((pred_conf - 1.0) ** 2)
        return loss_t + loss_r + 0.1 * loss_c


class PickPlaceDataset(Dataset):
    """Synthetic dataset that samples cube pointclouds and random poses.

    Returns:
      rgb: float32 (3,H,W) in [0,1]
      points: float32 (3,N)
      obj_id: int
      target_trans: float32 (3,)
      target_rot: float32 (4,) quaternion
    """

    def __init__(self, num_points=1000, num_samples=1000, img_size=(64, 64)):
        self.num_points = num_points
        self.num_samples = num_samples
        self.img_size = img_size

        # Define a few cube sizes (meters)
        self.cube_sizes = {
            0: (0.04, 0.04, 0.04),
            1: (0.05, 0.05, 0.05),
        }
        self.class_ids = list(self.cube_sizes.keys())

        # Precompute model points for each class: sample points on cube surfaces
        self.model_points = {}
        for cid, size in self.cube_sizes.items():
            sx, sy, sz = size
            # sample many points on the 6 faces
            pts = []
            samples_per_face = 5000
            for face in range(6):
                u = np.random.uniform(-0.5, 0.5, (samples_per_face, 2))
                if face == 0:
                    pts_face = np.stack([u[:, 0] * sx, u[:, 1] * sy, np.full(samples_per_face, -sz / 2)], axis=1)
                elif face == 1:
                    pts_face = np.stack([u[:, 0] * sx, u[:, 1] * sy, np.full(samples_per_face, sz / 2)], axis=1)
                elif face == 2:
                    pts_face = np.stack([u[:, 0] * sx, np.full(samples_per_face, -sy / 2), u[:, 1] * sz], axis=1)
                elif face == 3:
                    pts_face = np.stack([u[:, 0] * sx, np.full(samples_per_face, sy / 2), u[:, 1] * sz], axis=1)
                elif face == 4:
                    pts_face = np.stack([np.full(samples_per_face, -sx / 2), u[:, 0] * sy, u[:, 1] * sz], axis=1)
                else:
                    pts_face = np.stack([np.full(samples_per_face, sx / 2), u[:, 0] * sy, u[:, 1] * sz], axis=1)
                pts.append(pts_face)
            pts = np.concatenate(pts, axis=0)
            self.model_points[cid] = pts.astype(np.float32)

    def __len__(self):
        return int(self.num_samples)

    def __getitem__(self, index):
        # choose a class
        cid = random.choice(self.class_ids)
        pts = self.model_points[cid]

        # sample a pose in front of the camera
        tx = random.uniform(0.3, 0.6)
        ty = random.uniform(-0.2, 0.2)
        tz = random.uniform(0.2, 0.6)
        trans = np.array([tx, ty, tz], dtype=np.float32)

        # random rotation quaternion
        axis = np.random.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        theta = random.uniform(0, 2 * math.pi)
        R = rotation_matrix(axis, theta)
        quat = rot2quat(R).astype(np.float32)

        # transform model points into camera frame
        pts_cam = (R @ pts.T).T + trans[None, :]

        # sample N points
        if pts_cam.shape[0] >= self.num_points:
            idxs = np.random.choice(pts_cam.shape[0], self.num_points, replace=False)
        else:
            idxs = np.random.choice(pts_cam.shape[0], self.num_points, replace=True)
        sampled = pts_cam[idxs]

        # jitter
        sampled += np.random.normal(scale=0.002, size=sampled.shape)

        # create a dummy RGB patch (zeros) - model uses pooled features so shape matters
        H, W = self.img_size
        rgb = np.zeros((3, H, W), dtype=np.float32)
        # simple shading: encode depth mean into red channel for visualization
        depth_mean = sampled[:, 2].mean()
        rgb[0] = np.clip((depth_mean - 0.2) / 0.6, 0.0, 1.0)

        sample = {
            'rgb': rgb,  # (3,H,W)
            'points': sampled.astype(np.float32),  # (N,3)
            'obj_id': np.int64(cid),
            'trans': trans.astype(np.float32),
            'rot': quat.astype(np.float32),
        }
        return sample


def collate_fn(batch):
    B = len(batch)
    N = batch[0]['points'].shape[0]
    H, W = batch[0]['rgb'].shape[1:]
    rgb = np.stack([b['rgb'] for b in batch], axis=0)
    pts = np.stack([b['points'].T for b in batch], axis=0)  # (B,3,N)
    obj_ids = np.array([b['obj_id'] for b in batch], dtype=np.int64)
    trans = np.stack([b['trans'] for b in batch], axis=0)
    rot = np.stack([b['rot'] for b in batch], axis=0)
    return (
        torch.from_numpy(rgb).float(),
        torch.from_numpy(pts).float(),
        torch.from_numpy(obj_ids).long(),
        torch.from_numpy(trans).float(),
        torch.from_numpy(rot).float(),
    )


def train_densefusion(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset = PickPlaceDataset(num_points=args.num_points, num_samples=args.num_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    model = DenseFusion(num_points=args.num_points, num_obj=10).to(device)
    refiner = PoseRefineNet(num_points=args.num_points, num_obj=10).to(device)

    opt = optim.Adam(list(model.parameters()) + list(refiner.parameters()), lr=args.lr)
    criterion = DenseFusionLoss()

    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        refiner.train()
        running = 0.0
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for rgb, pts, obj_ids, trans_t, rot_t in pbar:
            rgb = rgb.to(device)
            pts = pts.to(device)
            obj_ids = obj_ids.to(device)
            trans_t = trans_t.to(device)
            rot_t = rot_t.to(device)

            opt.zero_grad()
            pred_t, pred_r, pred_c = model(rgb, pts, obj_ids)
            loss = criterion(pred_t, pred_r, pred_c, trans_t, rot_t)
            loss.backward()
            opt.step()

            running += loss.item()
            pbar.set_postfix({'loss': running / (pbar.n + 1)})

        epoch_loss = running / len(loader)
        print(f'Epoch {epoch+1} loss: {epoch_loss:.6f}')

        # save
        ckpt = {
            'model': model.state_dict(),
            'refiner': refiner.state_dict(),
            'opt': opt.state_dict(),
            'epoch': epoch + 1,
        }
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(args.checkpoint_dir, 'densefusion_latest.pth'))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, os.path.join(args.checkpoint_dir, 'densefusion_best.pth'))


def inference_demo(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset = PickPlaceDataset(num_points=args.num_points, num_samples=10)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = DenseFusion(num_points=args.num_points, num_obj=10).to(device)
    refiner = PoseRefineNet(num_points=args.num_points, num_obj=10).to(device)

    ckpt_path = os.path.join(args.checkpoint_dir, 'densefusion_best.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        refiner.load_state_dict(ckpt['refiner'])
        print('Loaded checkpoint', ckpt_path)
    else:
        print('Checkpoint not found at', ckpt_path)

    model.eval()
    refiner.eval()

    with torch.no_grad():
        for rgb, pts, obj_ids, trans_t, rot_t in loader:
            rgb = rgb.to(device)
            pts = pts.to(device)
            obj_ids = obj_ids.to(device)
            trans_t = trans_t.to(device)
            rot_t = rot_t.to(device)

            pred_t, pred_r, pred_c = model(rgb, pts, obj_ids)
            # one-step refinement
            delta_t, delta_r = refiner(pts.transpose(1, 2), obj_ids, pred_t, pred_r)
            pred_t_ref = pred_t + delta_t
            # compose rotations (quat multiplication)
            pred_r_np = pred_r.cpu().numpy()[0]
            delta_r_np = delta_r.cpu().numpy()[0]
            pred_r_comp = quat_mul(pred_r_np, delta_r_np)
            print('GT trans:', trans_t.cpu().numpy()[0], 'Pred trans:', pred_t_ref.cpu().numpy()[0])
            print('GT rot:', rot_t.cpu().numpy()[0], 'Pred rot:', pred_r_comp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['train', 'infer'], default='infer')
    p.add_argument('--num_points', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--num_samples', type=int, default=500)
    p.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train_densefusion(args)
    else:
        inference_demo(args)
