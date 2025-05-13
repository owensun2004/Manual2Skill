import os
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import argparse
import importlib
import torch
from models import build_model
from dataset.data_generation.dataloader import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import *
from utils.viz_utils import *
from utils.eval_utils import cal_metric, calculate_average
import pickle

@torch.no_grad()
def main(cfg):
    if torch.cuda.is_available() is False:
        raise ValueError('CUDA is not available. Please check your environment setting.')

    flie_path = cfg.data.dir
    flie_name = os.path.basename(flie_path)
    eval_pinkle_path = f"eval_set_{flie_name}.pkl"
    if os.path.exists(eval_pinkle_path):
        with open(eval_pinkle_path, "rb") as eval_file:
            eval_set = pickle.load(eval_file)
    else:
        eval_set = FurnitureAssemblyDataset(
            cfg.data.dir, 
            partnet=cfg.data.is_partnet,
            split='val', 
            train_ratio=cfg.data.train_ratio, 
            seed=cfg.data.seed, 
            min_num_part=cfg.data.min_num_part, 
            max_num_part=cfg.data.max_num_part
        )
        with open(eval_pinkle_path, "wb") as eval_file:
            pickle.dump(eval_set, eval_file)

    eval_loader = DataLoader(eval_set, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    logger.info(cfg)
        
    model = build_model(cfg).to(cfg.dist.local_rank)
    model.load(cfg.train.pretrained_weights, device=None)
    logger.info(f'load pretrained model from {cfg.train.pretrained_weights}')

    # viz(cfg, model, eval_loader)
    model.eval()
    metric_list = []
    
    for i, batch in tqdm(enumerate(eval_loader), desc=f'calculating metrics', total=len(eval_loader), disable=not is_main_process()):
        batch = {k: v.to(device=cfg.dist.local_rank) for k, v in batch.items()}
        # rot_mat = batch['rotation_matrix']
        # trans_vec = batch['center']
        # part_valids = batch['part_valids']
        # valid_mask = part_valids > 0
        # gt_rot_mat = rot_mat[valid_mask]
        # gt_trans_vec = trans_vec[valid_mask]

        pred = model(batch)
        metric_list.append(cal_metric(pred, batch, metric_type_list=['GD', 'RMSE_t', 'PA', 'cham'], PA_threshold=cfg.PA_threshold))

    metric_dict = calculate_average(metric_list)
    logger.info(metric_dict)

@torch.no_grad()
def viz(cfg, model, eval_loader=None):
    model.eval()
    viz_dir = cfg.train.log_dir+f'/viz'
    os.makedirs(viz_dir, exist_ok=True)
    viz_num_max = cfg.train.num_viz
    pbar = tqdm(total=viz_num_max, desc="Visualizing")
    viz_num = 0
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            batch = {k: v.to(device=cfg.dist.local_rank) for k, v in batch.items()}
            pred = model(batch)
            visualize_prediction(pred, batch, viz_dir, f'eval_{i}')
            pbar.update(1)
            viz_num += 1
            if viz_num >= viz_num_max:
                break

        pbar.close()

if __name__ == '__main__':
    file_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(file_path)
    os.chdir(dir_path)
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--cfg_file', default='pose_estimation/configs/base_pose_estimate.py', type=str, help='.py')
    args = parser.parse_args()

    sys.path.append(os.path.join(parent_dir, os.path.dirname(args.cfg_file)))
    cfg = importlib.import_module(os.path.basename(args.cfg_file)[:-3])
    cfg = cfg.get_cfg_defaults()
    log_dir_suffix = datetime.now().strftime("%m-%d-%H-%M-%S")
    cfg.dist.distributed = False
    # cfg.network_name = 'PoseEstimationNetwork'
    cfg.network_name = 'GNNNetwork'
    # cfg.network_name = '3DPartAssembly'
    cfg.dist.local_rank = 'cuda:0'
    cfg.train.batch_size = 16
    cfg.PA_threshold = 0.01
    cfg.data.dir = '/data2/lyw/IKEA_data3'
    cfg.data.is_partnet = True
    cfg.train.pretrained_weights = '/data2/lyw/Furniture-Assembly/pose_estimation/logs/GNNNetwork_IKEA_data3_04-19-21-08-36/best.ckpt'
    cfg.train.log_dir = f'./logs/eval_{cfg.network_name}_{cfg.data.dir.rstrip("/").split("/")[-1]}_{log_dir_suffix}/'
    os.makedirs(cfg.train.log_dir, exist_ok=True)
    logger = setup_logging(os.path.join(cfg.train.log_dir, 'eval.log'))
    cfg.train.device = cfg.dist.local_rank
    main(cfg)
