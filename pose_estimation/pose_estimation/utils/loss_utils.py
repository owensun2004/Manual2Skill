import torch
import numpy as np
import torch.nn.functional as F
from pdb import set_trace as bp


def bgs(d6s):
    # print(d6s.shape)
    b_copy = d6s.clone()
    bsz = d6s.shape[0]
    b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
    a2 = d6s[:, :, 1]
    b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1),
                                    a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=1)


def orthogonalization(raw_action):
    # [batch, 6] -> [batch, 3, 3]
    batch_size = raw_action.shape[0]
    R = bgs(raw_action[:,].reshape(-1, 2, 3).permute(0, 2, 1))
    return R


def bgdR(Rgts, Rps):
    Rgts = Rgts.float()
    Rps = Rps.float()
    Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
    Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1) #batch trace
    # necessary or it might lead to nans and the likes
    theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
    return torch.acos(theta)