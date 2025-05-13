import torch
from torch import nn
from pdb import set_trace as bp
import math
from pytorch3d.loss import chamfer_distance
from collections import defaultdict
from itertools import permutations, product
from .loss_utils import orthogonalization, bgdR
import torch.nn.functional as F


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


def cal_metric(pred, batch_data, metric_type_list=['GD', 'RMSE_t', 'PA'], PA_threshold=0.01):
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
            
            loss_dict_batch_best = defaultdict(lambda: float('inf')) 
            loss_dict_batch_best['PA'] = - loss_dict_batch_best['PA']
            for perm in permutations:
                loss_dict_batch_perm = defaultdict(lambda: 0) 
                gt_rot_mat_batch_perm = gt_rot_mat_batch[torch.tensor(perm)]
                gt_trans_vec_batch_perm = gt_trans_vec_batch[torch.tensor(perm)]
                pred_rot_mat_batch_perm = pred_rot_mat_batch[torch.tensor(perm)]
                pred_trans_vec_batch_perm = pred_trans_vec_batch[torch.tensor(perm)]
                rot_gt_batch_perm = torch.matmul(pts_batch, gt_rot_mat_batch_perm)
                rot_pred_batch_perm = torch.matmul(pts_batch, pred_rot_mat_batch_perm)
                p_gt_batch_perm = rot_gt_batch_perm + gt_trans_vec_batch_perm.unsqueeze(1)
                p_pred_batch_perm = rot_pred_batch_perm + pred_trans_vec_batch_perm.unsqueeze(1)

                if 'GD' in metric_type_list:
                    loss_rot = torch.mean(bgdR(gt_rot_mat_batch_perm, pred_rot_mat_batch_perm))
                    if loss_rot < loss_dict_batch_best['GD']:
                        loss_dict_batch_best['GD'] = loss_rot
                        
                if 'PA' in metric_type_list:
                    loss_cham, _ = chamfer_distance(p_gt_batch_perm, p_pred_batch_perm, point_reduction='mean', batch_reduction='mean')
                    PA_perm = loss_cham < PA_threshold
                    PA_perm = PA_perm.float().mean()
                    if PA_perm > loss_dict_batch_best['PA']:
                        loss_dict_batch_best['PA'] = PA_perm

                if 'RMSE_t' in metric_type_list:
                    loss_trans_mse = F.mse_loss(gt_trans_vec_batch_perm, pred_trans_vec_batch_perm)
                    RMSE = torch.sqrt(loss_trans_mse)
                    if RMSE < loss_dict_batch_best['RMSE_t']:
                        loss_dict_batch_best['RMSE_t'] = RMSE

                if 'cham' in metric_type_list:
                    loss_cham, _ = chamfer_distance(p_gt_batch_perm, p_pred_batch_perm, point_reduction='mean', batch_reduction='mean')
                    if loss_cham < loss_dict_batch_best['cham']:
                        loss_dict_batch_best['cham'] = loss_cham

                if 'match' in metric_type_list:
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
                        valid_lenth_list = torch.tensor(valid_lenth_list).to(device=valid_mask.device)
                        loss_cham_match, _ = chamfer_distance(y=gt_part_group_stack, x=pred_part_group_stack, y_lengths=valid_lenth_list, x_lengths=valid_lenth_list, point_reduction='mean', batch_reduction='mean')
                        if loss_cham_match < loss_dict_batch_best['match']:
                            loss_dict_batch_best['match'] = loss_cham_match
                        
            loss_dict_batch_list.append(loss_dict_batch_best)

        return calculate_average(loss_dict_batch_list)