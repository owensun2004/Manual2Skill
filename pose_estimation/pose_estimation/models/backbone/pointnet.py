import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import math

class PointNet2(nn.Module):
    ''' PointNet++-based encoder network
    Args:
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        out_dim (int): dimension of output
    '''

    def __init__(self, in_dim=3, hidden_dim=128, out_dim=3):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_dim + 3,
                                          mlp=[hidden_dim, hidden_dim * 2], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=hidden_dim * 2 + 3,
                                          mlp=[hidden_dim * 2, hidden_dim * 4], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=hidden_dim * 4 + 3,
                                          mlp=[hidden_dim * 4, hidden_dim * 8], group_all=True)

        self.conv2 = nn.Conv1d(hidden_dim * 8, out_dim, 1)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, xyz):
        xyz = xyz.permute(0, 2, 1)
        B, C, N = xyz.shape
        l0_xyz = xyz[:, :3, :]
        l0_points = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
 

        feat = F.relu(self.bn2(self.conv2(l3_points)))
        feat = feat.squeeze(2)
        return feat


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist = dist + torch.sum(src ** 2, -1).view(B, N, 1)
    dist = dist + torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

# class LoRALayer(nn.Module):
#     def __init__(self, original_layer, r: int, alpha: int):
#         """
#         Args:
#             original_layer (nn.Module): 原始的线性或卷积层
#             r (int): 低秩的维度
#             alpha (int): 缩放因子
#         """
#         super(LoRALayer, self).__init__()
#         self.original_layer = original_layer
#         self.r = r
#         self.alpha = alpha
#         self.scaling = self.alpha / self.r

#         # 低秩适配器
#         if isinstance(original_layer, nn.Conv1d):
#             self.W_A = nn.Conv1d(original_layer.in_channels, r, kernel_size=1, bias=False)
#             self.W_B = nn.Conv1d(r, original_layer.out_channels, kernel_size=1, bias=False)
#         elif isinstance(original_layer, nn.Conv2d):
#             self.W_A = nn.Conv2d(original_layer.in_channels, r, kernel_size=1, bias=False)
#             self.W_B = nn.Conv2d(r, original_layer.out_channels, kernel_size=1, bias=False)
#         else:
#             raise ValueError("Unsupported layer type for LoRA")

#         # 初始化 W_A 和 W_B
#         nn.init.kaiming_uniform_(self.W_A.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.W_B.weight)

#     def forward(self, x):
#         return self.original_layer(x) + self.scaling * self.W_B(self.W_A(x))

# class PointNetSetAbstractionWithLoRA(PointNetSetAbstraction):
#     def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, r=4, alpha=16):
#         super(PointNetSetAbstractionWithLoRA, self).__init__(npoint, radius, nsample, in_channel, mlp, group_all)
#         self.r = r
#         self.alpha = alpha

#         # 替换 mlp_convs 为带 LoRA 的卷积层
#         for i, conv in enumerate(self.mlp_convs):
#             self.mlp_convs[i] = LoRALayer(conv, r=self.r, alpha=self.alpha)

# class PointNetFeaturePropagationWithLoRA(PointNetFeaturePropagation):
#     def __init__(self, in_channel, mlp, r=4, alpha=16):
#         super(PointNetFeaturePropagationWithLoRA, self).__init__(in_channel, mlp)
#         self.r = r
#         self.alpha = alpha

#         # 替换 mlp_convs 为带 LoRA 的卷积层
#         for i, conv in enumerate(self.mlp_convs):
#             self.mlp_convs[i] = LoRALayer(conv, r=self.r, alpha=self.alpha)

# class PointNet2WithLoRA(PointNet2):
#     def __init__(self, in_dim=3, hidden_dim=128, out_dim=3, r=4, alpha=16):
#         super(PointNet2WithLoRA, self).__init__(in_dim, hidden_dim, out_dim)
#         self.sa1 = PointNetSetAbstractionWithLoRA(
#             npoint=512, radius=0.2, nsample=32, in_channel=in_dim + 3,
#             mlp=[hidden_dim, hidden_dim * 2], group_all=False, r=r, alpha=alpha
#         )
#         self.sa2 = PointNetSetAbstractionWithLoRA(
#             npoint=128, radius=0.4, nsample=64, in_channel=hidden_dim * 2 + 3,
#             mlp=[hidden_dim * 2, hidden_dim * 4], group_all=False, r=r, alpha=alpha
#         )
#         self.sa3 = PointNetSetAbstractionWithLoRA(
#             npoint=None, radius=None, nsample=None, in_channel=hidden_dim * 4 + 3,
#             mlp=[hidden_dim * 4, hidden_dim * 8], group_all=True, r=r, alpha=alpha
#         )

#         self.conv2 = LoRALayer(nn.Conv1d(hidden_dim * 8, out_dim, kernel_size=1), r=r, alpha=alpha)

# class PointNet2WithLoRA(nn.Module):
#     def __init__(self, in_dim=3, hidden_dim=128, out_dim=3, r=4, alpha=16):
#         """
#         Args:
#             in_dim (int): 输入点云的维度
#             hidden_dim (int): 隐藏层的维度
#             out_dim (int): 输出特征的维度
#             r (int): LoRA 的秩
#             alpha (int): LoRA 的缩放因子
#         """
#         super(PointNet2WithLoRA, self).__init__()
        
#         # 第一个层的采样点从 1024 调整为 512
#         self.sa1 = PointNetSetAbstractionWithLoRA(
#             npoint=512, radius=0.1, nsample=32, in_channel=in_dim + 3,
#             mlp=[hidden_dim, hidden_dim * 2], group_all=False, r=r, alpha=alpha
#         )
#         self.sa2 = PointNetSetAbstractionWithLoRA(
#             npoint=256, radius=0.2, nsample=32, in_channel=hidden_dim * 2 + 3,
#             mlp=[hidden_dim * 2, hidden_dim * 4], group_all=False, r=r, alpha=alpha
#         )
#         self.sa3 = PointNetSetAbstractionWithLoRA(
#             npoint=128, radius=0.4, nsample=64, in_channel=hidden_dim * 4 + 3,
#             mlp=[hidden_dim * 4, hidden_dim * 8], group_all=False, r=r, alpha=alpha
#         )
#         self.sa4 = PointNetSetAbstractionWithLoRA(
#             npoint=64, radius=0.8, nsample=64, in_channel=hidden_dim * 8 + 3,
#             mlp=[hidden_dim * 8, hidden_dim * 16], group_all=False, r=r, alpha=alpha
#         )
#         self.sa5 = PointNetSetAbstractionWithLoRA(
#             npoint=32, radius=1.0, nsample=128, in_channel=hidden_dim * 16 + 3,
#             mlp=[hidden_dim * 16, hidden_dim * 32], group_all=False, r=r, alpha=alpha
#         )
#         self.sa6 = PointNetSetAbstractionWithLoRA(
#             npoint=None, radius=None, nsample=None, in_channel=hidden_dim * 32 + 3,
#             mlp=[hidden_dim * 32, hidden_dim * 64], group_all=True, r=r, alpha=alpha
#         )

#         # 最后一层卷积，用于将特征压缩到指定的输出维度
#         self.conv2 = LoRALayer(nn.Conv1d(hidden_dim * 64, out_dim, kernel_size=1), r=r, alpha=alpha)

#         # 批归一化层
#         self.bn2 = nn.BatchNorm1d(out_dim)

#     def forward(self, xyz):
#         """
#         Forward pass for PointNet2WithLoRA.
        
#         Args:
#             xyz (torch.Tensor): 输入点云数据，形状为 [B, N, C]。
        
#         Returns:
#             torch.Tensor: 编码后的点云特征，形状为 [B, out_dim]。
#         """
#         xyz = xyz.permute(0, 2, 1)  # 转置为 [B, C, N]
#         B, C, N = xyz.shape

#         # 输入点云的初始位置和特征
#         l0_xyz = xyz[:, :3, :]
#         l0_points = xyz

#         # 依次通过6个 PointNetSetAbstractionWithLoRA 层
#         l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
#         l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
#         l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
#         l6_xyz, l6_points = self.sa6(l5_xyz, l5_points)

#         # 最后一层卷积，获取最终特征
#         feat = F.relu(self.bn2(self.conv2(l6_points)))
#         feat = feat.squeeze(2)  # 压缩为 [B, out_dim]
#         return feat

def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params > 1e3 and num_params < 1e6:
        num_params = f'{num_params/1e3:.2f}K'
    elif num_params > 1e6:
        num_params = f'{num_params/1e6:.2f}M'
    return num_params

if __name__ == "__main__":
    t = torch.rand(4, 1000, 4)
    # net = PointNet2(in_dim=4, hidden_dim=128, out_dim=128)
    net = PointNet2WithLoRA(in_dim=4, hidden_dim=128, out_dim=256, r=4, alpha=16)
    print(net)
    _ = net(t)
