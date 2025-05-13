from .backbone.pointnet import PointNet2
from .backbone.unet import UNet
from .backbone.mlp import mlp
from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from graph_transformer_pytorch import GraphTransformer
from utils import *
from pytorch3d.loss import chamfer_distance
from collections import defaultdict
from itertools import permutations, product, combinations
import math
from torch.optim import lr_scheduler

class Learning3DPartAssembly(nn.Module):
    def __init__(self, cfg):
        super(Learning3DPartAssembly, self).__init__()
        self.cfg = cfg
        self.unet = UNet(num_classes = 1, max_channels=256, n_channels=3)
        self.pointnet = PointNet2(in_dim=3, hidden_dim=256, out_dim=256)
        self.slp1 = nn.Linear(256+20, 256)
        self.slp2 = nn.Linear(256, 256)

        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(in_features=512, out_features=512, bias=True)
        self.resnet_final_bn = nn.BatchNorm1d(512)

        self.mask_resnet = resnet18(pretrained=False)
        self.mask_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=True)
        self.mask_resnet.fc = nn.Linear(in_features=512, out_features=512, bias=True)
        self.mask_resnet_final_bn = nn.BatchNorm1d(512)

        self.gnn = GraphTransformer(
            dim=1536,                  
            depth=2,                  
            edge_dim=256,             
            with_feedforwards=True,   
            gated_residual=True,      
            rel_pos_emb=False,         
            accept_adjacency_matrix=True  #
        )
        
        self.pose_regressor = mlp(input_dim=1536)

        self.init_optimizer_scheduler()

        print('Trainable Parameters Counting')
        # counting in million
        print('UNet:', count_parameters(self.unet))
        print('PointNet:', count_parameters(self.pointnet))
        print('SLP1:', count_parameters(self.slp1))
        print('SLP2:', count_parameters(self.slp2))
        print('Image ResNet18:', count_parameters(self.resnet))
        print('Mask ResNet18:', count_parameters(self.mask_resnet))
        print('GNN:', count_parameters(self.gnn))
        print('MLP:', count_parameters(self.pose_regressor))

    def init_optimizer_scheduler(self):
        optimizer1, scheduler1 = create_optimizer_and_scheduler(self.cfg.train, self.unet.parameters(), optimizer_type='rmsprop')
        optimizer2, scheduler2 = create_optimizer_and_scheduler(self.cfg.train, self.pointnet.parameters(), optimizer_type='adamw')
        optimizer3, scheduler3 = create_optimizer_and_scheduler(self.cfg.train, self.slp1.parameters(), optimizer_type='rmsprop')
        optimizer4, scheduler4 = create_optimizer_and_scheduler(self.cfg.train, self.slp2.parameters(), optimizer_type='rmsprop')
        optimizer5, scheduler5 = create_optimizer_and_scheduler(self.cfg.train, self.resnet.parameters(), optimizer_type='adamw')
        optimizer6, scheduler6 = create_optimizer_and_scheduler(self.cfg.train, self.mask_resnet.parameters(), optimizer_type='adamw')
        optimizer7, scheduler7 = create_optimizer_and_scheduler(self.cfg.train, self.gnn.parameters(), optimizer_type='adamw')
        optimizer8, scheduler8 = create_optimizer_and_scheduler(self.cfg.train, self.pose_regressor.parameters(), optimizer_type='rmsprop')

        self.optimizer = OptimizerManager([optimizer1, optimizer2, optimizer3, optimizer4, optimizer5, optimizer6, optimizer7, optimizer8])
        self.scheduler = SchedulerManager([scheduler1, scheduler2, scheduler3, scheduler4, scheduler5, scheduler6, scheduler7, scheduler8])

    def forward(self, batch_data):
        image = batch_data['image'].float()
        part_valids = batch_data['part_valids']
        pts = batch_data['point_cloud']
        valid_mask = part_valids > 0
        pts = pts[valid_mask]
        geo_feat = self.pointnet(pts)
        batched_adj_mat = torch.ones(valid_mask.shape[0], self.cfg.data.max_num_part, 
                                self.cfg.data.max_num_part).to(device=self.cfg.train.device)

        part_ids = batch_data['part_ids']
        part_ids[~valid_mask.unsqueeze(-1)] = -1
        ins_one_hot = generate_one_hot_scatter(part_ids)
        ins_one_hot_valid = ins_one_hot[valid_mask]

        local_feat = torch.cat([geo_feat, ins_one_hot_valid], dim=1)
        local_feat_aft_slp1 = self.slp1(local_feat)
        true_counts = valid_mask.sum(dim=1).tolist() 
        local_feat_splits = torch.split(local_feat_aft_slp1, true_counts)
        global_feat = extract_global_features(local_feat_splits)
        global_feat_aft_slp2 = self.slp2(global_feat)

        node_feat_list = []

        for i, local_feat_i in enumerate(local_feat_splits):
            num_part_i, _ = local_feat_i.shape
            global_feat_i = global_feat_aft_slp2[i].unsqueeze(0).repeat(num_part_i,1)
            three_d_feat_i = torch.cat([local_feat_i, global_feat_i], dim=1)
            image_i = image[i].unsqueeze(0).repeat(num_part_i,1,1,1)
            unet_mask_i = self.unet(image_i, three_d_feat_i)
            unet_mask_i_softmax = F.softmax(unet_mask_i, dim=0)

            image_i_standalized = standalize_image(image_i)
            img_feat_i = torch.relu(self.resnet_final_bn(self.resnet(image_i_standalized)))
            mask_feat_i = torch.relu(self.mask_resnet_final_bn(self.mask_resnet(unet_mask_i_softmax)))
            node_feat_i = torch.cat([mask_feat_i, img_feat_i, three_d_feat_i], dim=1)
            node_feat_list.append(node_feat_i)

        node_feat_stack = torch.cat(node_feat_list, dim=0)
        node_feat_stack_padding = torch.zeros(valid_mask.shape[0], self.cfg.data.max_num_part, node_feat_stack.shape[1]).to(valid_mask.device)
        node_feat_stack_padding[valid_mask] = node_feat_stack
        renew_node_feat_pts, _ = self.gnn(node_feat_stack_padding, adj_mat = batched_adj_mat, mask = valid_mask)
        mlp_input_tensor = renew_node_feat_pts[valid_mask]
        pose = self.pose_regressor(mlp_input_tensor)

        return pose

    def loss(self, pred, batch_data, loss_type_list=['geo', 'cham', 'match', 'mse', 'penalty']):
        part_valids = batch_data['part_valids']
        rot_mat = batch_data['rotation_matrix']
        trans_vec = batch_data['center']
        pts = batch_data['point_cloud']
        valid_mask = part_valids > 0
        mask = batch_data['mask'][valid_mask]
        pts = pts[valid_mask]
        true_counts = valid_mask.sum(dim=1).tolist() 

        gt_rot_mat = rot_mat[valid_mask]
        gt_trans_vec = trans_vec[valid_mask]
        pred_rot_mat = orthogonalization(pred[:, :6])
        pred_trans_vec = pred[:, 6:]

        # take account of part equivalence
        part_ids = batch_data['part_ids']
        part_ids = part_ids[valid_mask]
        part_ids_splits = torch.split(part_ids, true_counts)
        pts_splits = torch.split(pts, true_counts)
        mask_splits = torch.split(mask, true_counts)
        gt_rot_mat_splits  =  torch.split(gt_rot_mat, true_counts)
        gt_trans_vec_splits  =  torch.split(gt_trans_vec, true_counts)
        pred_rot_mat_splits  =  torch.split(pred_rot_mat, true_counts)
        pred_trans_vec_splits  =  torch.split(pred_trans_vec, true_counts)
        loss_dict_batch_list = []

        for part_ids_batch, pts_batch, gt_rot_mat_batch, gt_trans_vec_batch, pred_rot_mat_batch, pred_trans_vec_batch, mask_batch in zip(
            part_ids_splits, pts_splits, gt_rot_mat_splits, gt_trans_vec_splits, pred_rot_mat_splits, pred_trans_vec_splits, mask_splits):

            grouped_indices = group_by_value(part_ids_batch.squeeze(1))
            permutations = compute_permutations(grouped_indices)
            loss_total_batch_min = torch.tensor(float('inf'))
            loss_dict_batch_min = None
            for perm in permutations:
                loss_dict_batch_perm = defaultdict(lambda: torch.tensor(0.).to(device=self.cfg.dist.local_rank)) 
                gt_rot_mat_batch_perm = gt_rot_mat_batch[torch.tensor(perm)]
                gt_trans_vec_batch_perm = gt_trans_vec_batch[torch.tensor(perm)]
                pred_rot_mat_batch_perm = pred_rot_mat_batch[torch.tensor(perm)]
                pred_trans_vec_batch_perm = pred_trans_vec_batch[torch.tensor(perm)]
                rot_gt_batch_perm = torch.matmul(pts_batch, gt_rot_mat_batch_perm)
                rot_pred_batch_perm = torch.matmul(pts_batch, pred_rot_mat_batch_perm)
                p_gt_batch_perm = rot_gt_batch_perm + gt_trans_vec_batch_perm.unsqueeze(1)
                p_pred_batch_perm = rot_pred_batch_perm + pred_trans_vec_batch_perm.unsqueeze(1)

                if 'mse' in loss_type_list:
                    loss_mse = F.mse_loss(p_gt_batch_perm, p_pred_batch_perm)
                    loss_dict_batch_perm['loss_mse'] = loss_mse
                    if loss_mse:
                        loss_dict_batch_perm['loss_total'] = loss_dict_batch_perm['loss_total'] + loss_mse * self.cfg.network.mse_loss_weight
                
                if 'geo' in loss_type_list:
                    loss_rot = torch.mean(bgdR(gt_rot_mat_batch_perm, pred_rot_mat_batch_perm))
                    loss_trans = torch.mean(torch.sqrt((gt_trans_vec_batch_perm - pred_trans_vec_batch_perm)**2))
                    loss_total = self.cfg.network.rot_loss_weight * loss_rot + self.cfg.network.trans_loss_weight * loss_trans
                    loss_dict_batch_perm['loss_rot'] = loss_rot
                    loss_dict_batch_perm['loss_trans'] = loss_trans
                    if loss_total:
                        loss_dict_batch_perm['loss_total'] = loss_dict_batch_perm['loss_total'] + loss_total

                elif 'rot_geo' in loss_type_list:
                    loss_rot_geo = torch.mean(bgdR(gt_rot_mat_batch_perm, pred_rot_mat_batch_perm))
                    loss_dict_batch_perm['loss_rot_geo'] = loss_rot_geo
                    if loss_rot_geo:
                        loss_dict_batch_perm['loss_total'] = loss_dict_batch_perm['loss_total'] + loss_rot_geo

                if 'cham' in loss_type_list:
                    loss_cham, _ = chamfer_distance(p_gt_batch_perm, p_pred_batch_perm, point_reduction='mean', batch_reduction='mean')
                    loss_dict_batch_perm['loss_cham'] = loss_cham
                    if loss_cham:
                        loss_dict_batch_perm['loss_total'] = loss_dict_batch_perm['loss_total'] + loss_cham * self.cfg.network.chamfer_loss_weight

                if 'rot_mse' in loss_type_list:
                    loss_rot_mse = F.mse_loss(rot_gt_batch_perm, rot_pred_batch_perm)
                    loss_dict_batch_perm['loss_rot_mse'] = loss_rot_mse
                    if loss_rot_mse:
                        loss_dict_batch_perm['loss_total'] = loss_dict_batch_perm['loss_total'] + loss_rot_mse

                if 'rot_cham' in loss_type_list:
                    loss_rot_cham, _ = chamfer_distance(rot_gt_batch_perm, rot_pred_batch_perm, point_reduction='mean', batch_reduction='mean')
                    loss_dict_batch_perm['loss_rot_cham'] = loss_rot_cham
                    if loss_rot_cham:
                        loss_dict_batch_perm['loss_total'] = loss_dict_batch_perm['loss_total'] + loss_rot_cham
                
                if 'trans_mse' in loss_type_list:
                    loss_trans_mse = F.mse_loss(gt_trans_vec_batch_perm, pred_trans_vec_batch_perm)
                    loss_dict_batch_perm['loss_trans_mse'] = loss_trans_mse
                    if loss_trans_mse:
                        loss_dict_batch_perm['loss_total'] = loss_dict_batch_perm['loss_total'] + loss_trans_mse

                if 'match' in loss_type_list:
                    group = mask_batch.unique()
                    part_num = mask_batch.shape[0]
                    valid_lenth_list = []
                    pred_concat_list = []
                    gt_concat_list = []
                    for i in range(part_num):
                        pred_part = p_pred_batch_perm[i]
                        gt_part = p_gt_batch_perm[i]
                        #overfitting set
                        mask_part = mask_batch[i].squeeze(1)
                        # mask_part = mask_batch[i]
                        for g in group:
                            if g != -1:
                                pred_part_group = pred_part[mask_part == g].unsqueeze(0)
                                valid_lenth = pred_part_group.size(1)
                                if valid_lenth == 0:
                                    continue
                                padding_lenth = 1000 - valid_lenth
                                valid_lenth_list.append(valid_lenth)
                                pred_part_group_padding = F.pad(pred_part_group, (0, 0, 0, padding_lenth), "constant", 0)
                                gt_part_group = gt_part[mask_part == g].unsqueeze(0)
                                gt_part_group_padding = F.pad(gt_part_group, (0, 0, 0, padding_lenth), "constant", 0)
                                pred_concat_list.append(pred_part_group_padding)
                                gt_concat_list.append(gt_part_group_padding)

                    if pred_concat_list:
                        pred_part_group_stack = torch.cat(pred_concat_list, dim=0)
                        gt_part_group_stack = torch.cat(gt_concat_list, dim=0)
                        valid_lenth_list = torch.tensor(valid_lenth_list).to(device=self.cfg.dist.local_rank)
                        loss_cham_match, _ = chamfer_distance(y=gt_part_group_stack, x=pred_part_group_stack, y_lengths=valid_lenth_list, x_lengths=valid_lenth_list, point_reduction='mean', batch_reduction='mean')
                        loss_dict_batch_perm['loss_cham_match'] = loss_cham_match
                        if loss_cham_match:
                            loss_dict_batch_perm['loss_total'] = loss_dict_batch_perm['loss_total'] + loss_cham_match * self.cfg.network.match_loss_weight
                    else:
                        loss_dict_batch_perm['loss_cham_match'] = torch.tensor(0.).to(device=self.cfg.dist.local_rank)

                if loss_total_batch_min > loss_dict_batch_perm['loss_total']:
                    loss_total_batch_min = loss_dict_batch_perm['loss_total']
                    loss_dict_batch_min = loss_dict_batch_perm

            if 'penalty' in loss_type_list and loss_dict_batch_min:
                rot_pred_batch = torch.matmul(pts_batch, pred_rot_mat_batch)
                p_pred_batch = rot_pred_batch + pred_trans_vec_batch.unsqueeze(1)
                for _, equivalent_group in grouped_indices.items():
                    if len(equivalent_group) > 1:
                        index_combinations = list(combinations(equivalent_group, 2))  # [(1, 2), (1, 3), (1, 4), (2, 3), ...]
                        tensor_list_a = []
                        tensor_list_b = []
                        for idx1, idx2 in index_combinations:
                            tensor_a = p_pred_batch[idx1]  # (1000, 3)
                            tensor_b = p_pred_batch[idx2]  # (1000, 3)
                            tensor_list_a.append(tensor_a)
                            tensor_list_b.append(tensor_b)

                        tensor_stack_a = torch.stack(tensor_list_a)
                        tensor_stack_b = torch.stack(tensor_list_b)

                        loss_penalty, _ = chamfer_distance(tensor_stack_a, tensor_stack_b, point_reduction='mean', batch_reduction='mean')
                        loss_penalty = -1 * loss_penalty
                        loss_dict_batch_min['loss_penalty'] = loss_penalty
                        loss_dict_batch_min['loss_total'] = loss_dict_batch_min['loss_total'] + loss_penalty * self.cfg.network.penalty_loss_weight
                valid_penalty = loss_dict_batch_min.get('loss_penalty', None)
                if not valid_penalty:
                    loss_dict_batch_min['loss_penalty'] = torch.tensor(0.).to(device=self.cfg.dist.local_rank)
                    
            loss_dict_batch_list.append(loss_dict_batch_min)

        return calculate_average(loss_dict_batch_list)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'UNet': self.unet.state_dict(),
            'PointNet': self.pointnet.state_dict(),
            'SLP1': self.slp1.state_dict(),
            'SLP2': self.slp2.state_dict(),
            'Image ResNet18': self.resnet.state_dict(),
            'Mask ResNet18': self.mask_resnet.state_dict(),
            'GNN': self.gnn.state_dict(),
            'MLP': self.pose_regressor.state_dict(),
        }, path)
    
    def load(self, path, device=None):
        if not device:
            device = torch.device(self.cfg.train.device)

        checkpoint = torch.load(
            path,
            map_location=device 
        )
        load_partial_state_dict(self.unet, checkpoint['UNet'])
        load_partial_state_dict(self.pointnet, checkpoint['PointNet'])
        load_partial_state_dict(self.slp1, checkpoint['SLP1'])
        load_partial_state_dict(self.slp2, checkpoint['SLP2'])
        load_partial_state_dict(self.resnet, checkpoint['Image ResNet18'])
        load_partial_state_dict(self.mask_resnet, checkpoint['Mask ResNet18'])
        load_partial_state_dict(self.gnn, checkpoint['GNN'])
        load_partial_state_dict(self.pose_regressor, checkpoint['MLP'])

    def replace_bn_with_syncbn(self):
        self.image_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.image_encoder)
        self.pts_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.pts_encoder)
        self.pose_regressor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.pose_regressor)


def generate_one_hot_scatter(input_tensor, num_classes=20):
    input_tensor = input_tensor.squeeze(-1)  # (B, N)
    B, N = input_tensor.shape
    
    mask = input_tensor.unsqueeze(2) == input_tensor.unsqueeze(1)  # (B, N, N)
    lower_tri_mask = torch.tril(torch.ones(N, N, device=input_tensor.device)).bool()  # (N, N)
    counts = (mask & lower_tri_mask).float().sum(dim=2).long()  # (B, N)
    counts_total = mask.sum(dim=2)  # (B, N)
    unique_mask = counts_total == 1  # (B, N)
    
    # Initialize indices for scatter
    scatter_indices = counts.clone()
    scatter_indices[unique_mask] = 0  # Unique elements set to index 0
    
    # Check for counts >= num_classes
    if (scatter_indices >= num_classes).any():
        raise ValueError("error in generating one hot")
    
    # Expand scatter_indices to (B, N, 1)
    scatter_indices = scatter_indices.unsqueeze(-1)  # (B, N, 1)
    
    # Create one-hot tensor
    one_hot = torch.zeros(B, N, num_classes, device=input_tensor.device)
    one_hot.scatter_(2, scatter_indices, 1)
    
    return one_hot

def extract_global_features(tensor_tuple):
    """
    Extract global features from each element in the tuple using max pooling to obtain global features with shape [1, 256].
    
    Parameters:
        tensor_tuple (tuple): A tuple containing multiple tensors with shape [x, 256]
    
    Returns:
        torch.Tensor: A tensor with shape [length, 256]
    """
    global_features = []
    
    for tensor in tensor_tuple:
        # Perform max pooling operation along the 0th dimension (x dimension)
        max_pool, _ = torch.max(tensor, dim=0, keepdim=True)  # max pooling to get shape [1, 256]
        global_features.append(max_pool)
    
    # Stack the extracted global features into a new tensor with shape [length, 256]
    global_features_tensor = torch.cat(global_features, dim=0)
    
    return global_features_tensor


def standalize_image(img):
        img = img.clone()
        img[:, 0] = (img[:, 0] - 0.485) / 0.229
        img[:, 1] = (img[:, 1] - 0.456) / 0.224
        img[:, 2] = (img[:, 2] - 0.406) / 0.225
        
        return img

class OptimizerManager:
    def __init__(self, optimizers):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return [optimizer.state_dict() for optimizer in self.optimizers]

    def load_state_dict(self, state_dicts):
        for optimizer, state_dict in zip(self.optimizers, state_dicts):
            optimizer.load_state_dict(state_dict)

class SchedulerManager:
    def __init__(self, schedulers):
        self.schedulers = schedulers

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        return [sch.state_dict() for sch in self.schedulers]

    def load_state_dict(self, state_dicts):
        for scheduler, state_dict in zip(self.schedulers, state_dicts):
            scheduler.load_state_dict(state_dict)
    def get_learning_rates(self):
        return self.schedulers[0].get_last_lr()


def create_optimizer_and_scheduler(cfg, parameters, optimizer_type='adamw'):
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.l2_norm)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, lr=cfg.lr, weight_decay=cfg.l2_norm, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type.")

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        return ((1 + math.cos((epoch - cfg.warmup_epochs) * math.pi / (cfg.num_epochs - cfg.warmup_epochs))) / 2) * (1 - cfg.lr) + cfg.lr

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler

def load_partial_state_dict(model, state_dict):
    model_dict = model.state_dict()  
    matched_state_dict = {}

    for key, value in state_dict.items():
        if key in model_dict and model_dict[key].size() == value.size():
            matched_state_dict[key] = value

    model_dict.update(matched_state_dict)
    model.load_state_dict(model_dict)

def sequential_pad_batch(data_batch):
    image_H, image_W, point_cloud_nums = 768, 432, 1000
    # Find the max Ni (Nmax)
    Nmax = 4

    batch_size = len(data_batch)
    image_batch = torch.zeros((batch_size, 3, image_H, image_W), dtype=torch.float32)
    mask_batch = torch.zeros((batch_size, Nmax, point_cloud_nums), dtype=torch.float32)
    point_cloud_batch = torch.zeros((batch_size, Nmax, point_cloud_nums, 3), dtype=torch.float32)
    rotation_matrix_batch = torch.zeros((batch_size, Nmax, 3, 3), dtype=torch.float32)
    center_batch = torch.zeros((batch_size, Nmax, 3), dtype=torch.float32)

    # Sequential padding
    for i, sample in enumerate(data_batch):
        # No need to pad image
        image_batch[i] = sample['image']

        # Padding mask
        mask = sample['mask']
        mask_shape = mask.shape[0]
        repetitions = -(-Nmax // mask_shape)  # Ceil division to calculate repetitions
        repeated_mask = mask.repeat(repetitions, 1)[:Nmax]
        mask_batch[i] = repeated_mask

        # Padding point_cloud
        point_cloud = sample['point_cloud']
        point_cloud_shape = point_cloud.shape[0]
        repetitions = -(-Nmax // point_cloud_shape)
        repeated_point_cloud = point_cloud.repeat(repetitions, 1, 1)[:Nmax]
        point_cloud_batch[i] = repeated_point_cloud

        # Padding rotation_matrix
        rotation_matrix = sample['rotation_matrix']
        rotation_matrix_shape = rotation_matrix.shape[0]
        repetitions = -(-Nmax // rotation_matrix_shape)
        repeated_rotation_matrix = rotation_matrix.repeat(repetitions, 1, 1)[:Nmax]
        rotation_matrix_batch[i] = repeated_rotation_matrix

        # Padding center
        center = sample['center']
        center_shape = center.shape[0]
        repetitions = -(-Nmax // center_shape)
        repeated_center = center.repeat(repetitions, 1)[:Nmax]
        center_batch[i] = repeated_center

    return Nmax, image_batch, mask_batch, point_cloud_batch, rotation_matrix_batch, center_batch

def group_by_value(tensor):
    """
    Group the same values in the Tensor.
    Parameters:
        tensor (torch.Tensor): The input tensor with arbitrary shape.
    Returns:
        dict: A dictionary with values as keys and the indices of each group as values.
    """
    if not tensor.is_floating_point():
        unique_values = tensor.unique()  # Get unique values
    else:
        unique_values = torch.unique(tensor, sorted=True, return_counts=False)  # Avoid floating-point errors

    value_to_indices = {}
    for value in unique_values:
        # indices = (tensor == value).nonzero(as_tuple=True)  # Find all indices equal to value
        indices = torch.where(tensor == value)
        index_list = indices[0].cpu().tolist()
        value_to_indices[value.item()] = index_list

    return value_to_indices


def calculate_average(dict_list):
    """
    Calculate the average of values for each key in a list of dictionaries.

    :param dict_list: List of dictionaries with identical keys
    :return: A dictionary with keys from input and values as averages
    """

    # Initialize result dictionary
    if dict_list[0] == None:
        return None
    
    result = {key: 0 for key in dict_list[0]}

    # Calculate sum for each key
    for d in dict_list:
        for key, value in d.items():
            result[key] += value

    # Calculate average for each key
    num_elements = len(dict_list)
    for key in result:
        result[key] /= num_elements

    return result

def compute_permutations(grouped_indices):
    index_groups = grouped_indices.values()
    # Compute permutations for each sublist
    sublist_permutations = [list(permutations(group)) for group in index_groups]
    # Compute the cartesian product of the sublist permutations
    combined_permutations = product(*sublist_permutations)
    # Flatten the tuples and merge them into a single list
    final_permutations = [tuple(item for sublist in combination for item in sublist) for combination in combined_permutations]
    return final_permutations