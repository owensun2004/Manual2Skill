import os
import sys
from pathlib import Path
from PIL import Image
import pwd
import argparse
import importlib
import torch
from pdb import set_trace as bp

import logging
import wandb
from .loss_utils import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import trange

colors_hex = [
        '#5A9BD5', '#FF6F61', '#77B77A', '#A67EB1', '#FF89B6', '#FFB07B',
        '#C5A3CF', '#FFA8B6', '#A3C9E0', '#FFC89B', '#E58B8B',
        '#A3B8D3', '#D4C3E8', '#66B2AA', '#E4A878', '#6882A4', '#D1AEDD', '#E8A4A6',
        '#A5DAD7', '#C6424A', '#E1D1F4', '#FFD8DC', '#F4D49B', '#8394A8'
    ]

def visualize_prediction(pred,batch,save_dir, img_name):
    pred_pose = pred
    rot_gt = batch['rotation_matrix'].detach().cpu().numpy()
    trans_gt = batch['center'].detach().cpu().numpy()
    pts = batch['point_cloud'].detach().cpu().numpy()
    parts_valid = batch['part_valids'].detach().cpu().numpy()
    img = batch['image'].detach().cpu().numpy()
    SO3_pred = orthogonalization(pred_pose[:, :6]).detach().cpu().numpy()
    trans_pred = pred_pose[:, 6:].detach().cpu().numpy()

    image = Image.fromarray((img[0].transpose(1, 2, 0) * 255).astype(np.uint8))

    fig = plt.figure()
    gs = GridSpec(3, 3, figure=fig)
    img = fig.add_subplot(gs[0, :])
    img.set_title('Input Image')
    img.imshow(image)
    # ax = fig.add_subplot(332, projection='3d')
    bx1 = fig.add_subplot(gs[1,0], projection='3d')
    bx2 = fig.add_subplot(gs[1,1], projection='3d')
    bx3 = fig.add_subplot(gs[1,2], projection='3d')
    bx1.view_init(30,45)
    bx2.view_init(90,0)
    bx3.view_init(0,90)
    cx1 = fig.add_subplot(gs[2,0], projection='3d')
    cx2 = fig.add_subplot(gs[2,1], projection='3d')
    cx3 = fig.add_subplot(gs[2,2], projection='3d')
    cx1.view_init(30,45)
    cx2.view_init(90,0)
    cx3.view_init(0,90)
    num_parts = int(parts_valid[0].sum())
    # for i in range(num_parts):
    #     x = pts[0, i, :, 0]
    #     y = pts[0, i, :, 1]
    #     z = pts[0, i, :, 2]
    #     ax.scatter(x, y, z, c=colors_hex[i])
    # ax.set_title('Input')

    for i in range(num_parts):
        p_gt = np.matmul(pts[0, i], rot_gt[0, i]) + trans_gt[0, i]
        x, y, z = p_gt[:, 0], p_gt[:, 1], p_gt[:, 2]
        bx1.scatter(x, y, z, c=colors_hex[i])
        bx2.scatter(x, y, z, c=colors_hex[i])
        bx3.scatter(x, y, z, c=colors_hex[i])
    bx1.set_title('gt (30,45)')
    bx2.set_title('gt (90,0)')
    bx3.set_title('gt (0,90)')
    bx1.axis('off')
    bx2.axis('off')
    bx3.axis('off')

    for i in range(num_parts):
        p_recon = np.matmul(pts[0, i], SO3_pred[i]) + trans_pred[i]
        x, y, z = p_recon[:, 0], p_recon[:, 1], p_recon[:, 2]
        cx1.scatter(x, y, z, c=colors_hex[i])
        cx2.scatter(x, y, z, c=colors_hex[i])
        cx3.scatter(x, y, z, c=colors_hex[i])
    cx1.set_title('recon (30,45)')
    cx2.set_title('recon (90,0)')
    cx3.set_title('recon (0,90)')
    cx1.axis('off')
    cx2.axis('off')
    cx3.axis('off')
    # plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{img_name}.png'))
    plt.close()




