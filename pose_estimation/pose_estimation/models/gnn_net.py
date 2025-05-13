from regex import P
from .backbone.deeplabv3p import DeepLab_V3_plus
from .backbone.pointnet import PointNet2
from .backbone.mlp import mlp
from utils import *
import torch
from torch import nn
from pdb import set_trace as bp
from torch.optim import lr_scheduler
import math
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from collections import defaultdict
from itertools import permutations, product, combinations
from graph_transformer_pytorch import GraphTransformer


class GNNNetwork(nn.Module):
    def __init__(self, cfg):
        super(GNNNetwork, self).__init__()
        self.cfg = cfg
        self.image_encoder = DeepLab_V3_plus(num_classes = 1, input_channel=3, out_dim=self.cfg.network.hidden_size)
        self.pts_encoder = PointNet2(in_dim=3, hidden_dim=128, out_dim=self.cfg.network.hidden_size*2)
        self.gnn = GraphTransformer(
            dim=self.cfg.network.hidden_size*2,                  
            depth=3,                  
            edge_dim=512,             
            with_feedforwards=True,   
            gated_residual=True,      
            rel_pos_emb=False,         
            accept_adjacency_matrix=True  #
        )
        self.pose_regressor = mlp(input_dim=self.cfg.network.hidden_size*2)

        self.init_optimizer_scheduler()

        print('Trainable Parameters Counting')
        # counting in million
        print('Image Encoder:', count_parameters(self.image_encoder))
        print('Point Cloud Encoder:', count_parameters(self.pts_encoder))
        print('Graph Transformer:', count_parameters(self.gnn))
        print('Pose Regressor:', count_parameters(self.pose_regressor))

        self.forward_used = self.forward_real_world if self.cfg.network.real_world else self.forward_train

    def init_optimizer_scheduler(self):
        optimizer1, scheduler1 = create_optimizer_and_scheduler(self.cfg.train, self.image_encoder.parameters(), optimizer_type='rmsprop')
        optimizer2, scheduler2 = create_optimizer_and_scheduler(self.cfg.train, self.pts_encoder.parameters(), optimizer_type='adamw')
        optimizer3, scheduler3 = create_optimizer_and_scheduler(self.cfg.train, self.pose_regressor.parameters(), optimizer_type='rmsprop')
        optimizer4, scheduler4 = create_optimizer_and_scheduler(self.cfg.train, self.gnn.parameters(), optimizer_type='adamw')
        self.optimizer = OptimizerManager([optimizer1, optimizer2, optimizer3, optimizer4])
        self.scheduler = SchedulerManager([scheduler1, scheduler2, scheduler3, scheduler4])

    def forward_train(self, batch_data):
        # image: (B, 3, H, W)
        # pts: (B, P, N, 4)

        image = batch_data['image'].float()
        pts = batch_data['point_cloud']
        B, P, N, _ = pts.shape
        # mask = batch_data['mask']
        valid_mask = batch_data['part_valids'] > 0
        # pts_w_mask = torch.cat([pts, mask], dim=-1)
        # pts_w_mask_valid = pts_w_mask[valid_mask].float()
        # pts_feat = self.pts_encoder(pts_w_mask_valid)
        pts = pts[valid_mask].float()
        pts_feat = self.pts_encoder(pts)
        img_feat = self.image_encoder(image)
        pts_feat_padding = torch.zeros(B, P, pts_feat.shape[-1]).to(pts_feat.device)
        pts_feat_padding[valid_mask] = pts_feat
        img_mask = torch.ones((valid_mask.shape[0], 1), dtype=torch.bool).to(pts_feat.device)
        valid_mask_w_img = torch.cat([img_mask, valid_mask], dim=1)
        node_feat = torch.cat([img_feat.unsqueeze(1), pts_feat_padding], dim=1)
        batched_adj_mat =torch.ones(valid_mask.shape[0], self.cfg.data.max_num_part + 1, 
                                self.cfg.data.max_num_part + 1).to(device=self.cfg.train.device)#num_part + image
        renew_node_feat, _ = self.gnn(node_feat, adj_mat = batched_adj_mat, mask = valid_mask_w_img)
        renew_node_feat_pts = renew_node_feat[:, 1:, :]
        mlp_input_tensor = renew_node_feat_pts[valid_mask]
        pose = self.pose_regressor(mlp_input_tensor)
        return pose
    
        # pts = batch_data['point_cloud']
        # B, P, N, _ = pts.shape
        # mask = batch_data['mask']
        # valid_mask = batch_data['part_valids'] > 0
        # pts_w_mask = torch.cat([pts, mask], dim=-1)
        # pts_w_mask_valid = pts_w_mask[valid_mask].float()
        # pts_feat = self.pts_encoder(pts_w_mask_valid)
        # pts_feat_padding = torch.zeros(B, P, pts_feat.shape[-1]).to(pts_feat.device)
        # pts_feat_padding[valid_mask] = pts_feat
        # batched_adj_mat =torch.ones(valid_mask.shape[0], self.cfg.data.max_num_part, 
        #                         self.cfg.data.max_num_part).to(device=self.cfg.train.device)#num_part + image
        # renew_node_feat, _ = self.gnn(pts_feat_padding, adj_mat = batched_adj_mat, mask = valid_mask)
        # mlp_input_tensor = renew_node_feat[valid_mask]
        # pose = self.pose_regressor(mlp_input_tensor)
        # return pose

    def forward_real_world(self, batch_data):
        pts = batch_data['point_cloud'].squeeze(0) # (part_num, point_num, 3)
        image = batch_data['image'].float() # (1, 3, H, W)
        P, N, _ = pts.shape

        pts_feat = self.pts_encoder(pts).unsqueeze(0) # (1, part_num, 256)
        img_feat = self.image_encoder(image).unsqueeze(0) # (1, 1, 256)

        valid_mask = torch.ones((1, P), dtype=torch.bool).to(pts_feat.device)
        img_mask = torch.ones((1, 1), dtype=torch.bool).to(pts_feat.device)
        valid_mask_w_img = torch.cat([img_mask, valid_mask], dim=1)

        node_feat = torch.cat([img_feat, pts_feat], dim=1)
        batched_adj_mat =torch.ones(1, P + 1, P + 1).to(device=self.cfg.train.device)

        renew_node_feat, _ = self.gnn(node_feat, adj_mat = batched_adj_mat, mask = valid_mask_w_img)
        renew_node_feat_pts = renew_node_feat[:, 1:, :]
        mlp_input_tensor = renew_node_feat_pts[valid_mask]
        pose = self.pose_regressor(mlp_input_tensor)
        return pose

    def forward(self, batch_data):
        return self.forward_used(batch_data)
    
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

        # Dont take account of part equivalence 

        # rot_gt = torch.matmul(pts, gt_rot_mat)
        # rot_pred = torch.matmul(pts, pred_rot_mat)
        # p_gt = rot_gt + gt_trans_vec.unsqueeze(1)
        # p_pred = rot_pred + pred_trans_vec.unsqueeze(1)
        # loss_dict = defaultdict(lambda: 0)  

        # if 'mse' in loss_type_list:
        #     loss_mse = F.mse_loss(p_gt, p_pred)
        #     loss_dict['loss_mse'] = loss_mse
        #     loss_dict['loss_total'] = loss_dict['loss_total'] + 20*loss_mse
        
        # if 'geo' in loss_type_list:
        #     loss_rot = torch.mean(bgdR(gt_rot_mat, pred_rot_mat))
        #     loss_trans = torch.mean(torch.sqrt((gt_trans_vec - pred_trans_vec)**2))
        #     loss_total = self.cfg.network.rot_loss_weight * loss_rot + self.cfg.network.trans_loss_weight * loss_trans
        #     loss_dict['loss_rot'] = loss_rot
        #     loss_dict['loss_trans'] = loss_trans
        #     loss_dict['loss_total'] = loss_dict['loss_total'] + loss_total

        # elif 'rot_geo' in loss_type_list:
        #     loss_rot_geo = torch.mean(bgdR(gt_rot_mat, pred_rot_mat))
        #     loss_dict['loss_rot_geo'] = loss_rot_geo
        #     loss_dict['loss_total'] = loss_dict['loss_total'] + loss_rot_geo

        # if 'cham' in loss_type_list:
        #     loss_cham, _ = chamfer_distance(p_gt, p_pred, point_reduction='mean', batch_reduction='mean')
        #     loss_dict['loss_cham'] = loss_cham
        #     loss_dict['loss_total'] = loss_dict['loss_total'] + loss_cham
        
        # if 'match' in loss_type_list:
        #     mask = batch_data['mask'][valid_mask]
        #     pred_splits = torch.split(p_pred, true_counts)
        #     gt_splits = torch.split(p_gt, true_counts)
        #     mask_splits = torch.split(mask, true_counts)

        #     valid_lenth_list = []
        #     pred_concat_list = []
        #     gt_concat_list = []
        #     for pred_batch, gt_batch, mask_batch in zip(pred_splits, gt_splits, mask_splits):
        #         group = mask_batch.unique()
        #         batch_size = mask_batch.shape[0]
        #         for i in range(batch_size):
        #             pred_part = pred_batch[i]
        #             gt_part = gt_batch[i]
        #             #overfitting set
        #             mask_part = mask_batch[i].squeeze(1)
        #             # mask_part = mask_batch[i]
        #             for g in group:
        #                 if g != -1:
        #                     pred_part_group = pred_part[mask_part == g].unsqueeze(0)
        #                     valid_lenth = pred_part_group.size(1)
        #                     if valid_lenth == 0:
        #                         continue
        #                     padding_lenth = 500 - valid_lenth
        #                     valid_lenth_list.append(valid_lenth)
        #                     pred_part_group_padding = F.pad(pred_part_group, (0, 0, 0, padding_lenth), "constant", 0)
        #                     gt_part_group = gt_part[mask_part == g].unsqueeze(0)
        #                     gt_part_group_padding = F.pad(gt_part_group, (0, 0, 0, padding_lenth), "constant", 0)
        #                     pred_concat_list.append(pred_part_group_padding)
        #                     gt_concat_list.append(gt_part_group_padding)

        #     pred_part_group_stack = torch.cat(pred_concat_list, dim=0)
        #     gt_part_group_stack = torch.cat(gt_concat_list, dim=0)
        #     valid_lenth_list = torch.tensor(valid_lenth_list).to(pred_part_group_stack.device)
        #     loss_cham_match, _ = chamfer_distance(y=gt_part_group_stack, x=pred_part_group_stack, y_lengths=valid_lenth_list, x_lengths=valid_lenth_list, point_reduction='mean', batch_reduction='mean')
        #     loss_dict['loss_cham_match'] = loss_cham_match
        #     loss_dict['loss_total'] = loss_dict['loss_total'] + loss_cham_match

        # if 'rot_mse' in loss_type_list:
        #     loss_rot_mse = F.mse_loss(rot_gt, rot_pred)
        #     loss_dict['loss_rot_mse'] = loss_rot_mse
        #     loss_dict['loss_total'] = loss_dict['loss_total'] + loss_rot_mse

        # if 'rot_cham' in loss_type_list:
        #     loss_rot_cham, _ = chamfer_distance(rot_gt, rot_pred, point_reduction='mean', batch_reduction='mean')
        #     loss_dict['loss_rot_cham'] = loss_rot_cham
        #     loss_dict['loss_total'] = loss_dict['loss_total'] + loss_rot_cham
        
        # if 'trans_mse' in loss_type_list:
        #     loss_trans_mse = F.mse_loss(gt_trans_vec, pred_trans_vec)
        #     loss_dict['loss_trans_mse'] = loss_trans_mse
        #     loss_dict['loss_total'] = loss_dict['loss_total'] + loss_trans_mse

        # return dict(loss_dict)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'image_encoder': self.image_encoder.state_dict(),
            'pts_encoder': self.pts_encoder.state_dict(),
            'pose_regressor': self.pose_regressor.state_dict(),
            'gnn':self.gnn.state_dict()
        }, path)
    
    def load(self, path, device=None):
        if not device:
            device = torch.device(self.cfg.train.device)

        checkpoint = torch.load(
            path,
            map_location=device 
        )
        load_partial_state_dict(self.image_encoder, checkpoint['image_encoder'])
        load_partial_state_dict(self.pts_encoder, checkpoint['pts_encoder'])
        load_partial_state_dict(self.pose_regressor, checkpoint['pose_regressor'])
        load_partial_state_dict(self.gnn, checkpoint['gnn'])
    
    def replace_bn_with_syncbn(self):
        self.image_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.image_encoder)
        self.pts_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.pts_encoder)
        self.pose_regressor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.pose_regressor)

    def loss_combination(self, pred, batch_data, loss_type_list=['geo', 'cham']):
        merged_loss_dict = defaultdict(lambda: 0)  

        for loss_type in loss_type_list:
            loss_dict = self.loss(pred, batch_data, loss_type)
            for key, value in loss_dict.items():
                merged_loss_dict[key] += value  

        return dict(merged_loss_dict)
    

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

if __name__ == '__main__':
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)

    model = GraphTransformer(
    dim = 256,
    depth = 3,
    with_feedforwards = True,   # whether to add a feedforward after each attention layer, suggested by literature to be needed
    gated_residual = True,      # to use the gated residual to prevent over-smoothing
    rel_pos_emb = False          # set to True if the nodes are ordered, default to False
)