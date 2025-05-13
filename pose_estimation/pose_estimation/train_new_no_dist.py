import os
import sys
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
import pwd
import argparse
import importlib
import torch
from pdb import set_trace as bp
from models import build_model
from dataset.data_generation.dataloader import *
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler
from tqdm import tqdm
import logging
import wandb
from utils.utils import *
from utils.viz_utils import *
import pickle

def main(cfg):
    if torch.cuda.is_available() is False:
        raise ValueError('CUDA is not available. Please check your environment setting.')

    flie_path = cfg.data.dir
    flie_name = os.path.basename(flie_path)
    train_pinkle_path = f"train_set_{flie_name}.pkl"
    eval_pinkle_path = f"eval_set_{flie_name}.pkl"
    if os.path.exists(train_pinkle_path) and os.path.exists(eval_pinkle_path):
        if not cfg.dist.distributed or is_main_process():
            logger.info("Loading train_set and eval_set from pickle files...")
        with open(train_pinkle_path, "rb") as train_file:
            train_set = pickle.load(train_file)
        with open(eval_pinkle_path, "rb") as eval_file:
            eval_set = pickle.load(eval_file)
    else:
        if not cfg.dist.distributed or is_main_process():
            logger.info("Pickle files not found. Creating train_set and eval_set...")

        train_set = FurnitureAssemblyDataset(
            cfg.data.dir, 
            partnet=cfg.data.is_partnet,
            split='train', 
            train_ratio=cfg.data.train_ratio, 
            seed=cfg.data.seed, 
            min_num_part=cfg.data.min_num_part, 
            max_num_part=cfg.data.max_num_part
        )
        eval_set = FurnitureAssemblyDataset(
            cfg.data.dir, 
            partnet=cfg.data.is_partnet,
            split='val', 
            train_ratio=cfg.data.train_ratio, 
            seed=cfg.data.seed, 
            min_num_part=cfg.data.min_num_part, 
            max_num_part=cfg.data.max_num_part
        )

        if not cfg.dist.distributed or is_main_process():
            logger.info("Train_set and eval_set saved to pickle files.")
            with open(train_pinkle_path, "wb") as train_file:
                pickle.dump(train_set, train_file)
            with open(eval_pinkle_path, "wb") as eval_file:
                pickle.dump(eval_set, eval_file)
    
    # train_set = FurnitureAssemblyDataset(
    #     cfg.data.dir, 
    #     partnet=cfg.data.is_partnet,
    #     split='train', 
    #     train_ratio=cfg.data.train_ratio, 
    #     seed=cfg.data.seed, 
    #     min_num_part=cfg.data.min_num_part, 
    #     max_num_part=cfg.data.max_num_part
    # )
    # eval_set = FurnitureAssemblyDataset(
    #     cfg.data.dir, 
    #     partnet=cfg.data.is_partnet,
    #     split='val', 
    #     train_ratio=cfg.data.train_ratio, 
    #     seed=cfg.data.seed, 
    #     min_num_part=cfg.data.min_num_part, 
    #     max_num_part=cfg.data.max_num_part
    # )
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_set, batch_size=cfg.train.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if not cfg.dist.distributed or is_main_process():
        wandb.init(project='furniture_assembly', config=cfg)
        logger.info(cfg)
        
    model = build_model(cfg).to(cfg.train.device)
    if cfg.train.pretrained:
        model.load(cfg.train.pretrained_weights, device=None)
        logger.info(f'load pretrained model from {cfg.train.pretrained_weights}')

    if cfg.dist.distributed:
        model.replace_bn_with_syncbn()
        model = torch.nn.parallel.DataParallel(model, device_ids=[cfg.dist.local_rank], output_device=cfg.dist.local_rank)
    best_loss = float('inf')
    patience = 0
    save_epoch = 0

    for epoch in range(cfg.train.num_epochs + 1):
        train_losses, _ = train_epoch(cfg, model, train_loader, epoch)
        if not train_losses:
            reload_path = os.path.abspath(os.path.join(cfg.train.log_dir, f'{save_epoch}.ckpt')) if save_epoch else os.path.abspath(os.path.join(cfg.train.log_dir, 'last.ckpt'))
            logger.info(f'break at {epoch}, reload from {reload_path}')
            model.load(reload_path, device=torch.device(cfg.dist.local_rank))
            model.init_optimizer_scheduler()
            model.optimizer.zero_grad()
            continue
        else:
            model.save(os.path.join(cfg.train.log_dir, 'last.ckpt'))
        if (epoch+1) % cfg.train.viz_every == 0 and epoch != 0 and is_main_process():
            viz(cfg, model, train_loader, eval_loader, epoch)
        if (epoch+1) % cfg.train.eval_every == 0 and epoch != 0 and is_main_process():
            eval_losses, total_loss = eval_epoch(cfg, model, eval_loader, epoch)
            # select the best model
            if best_loss > total_loss and total_loss:
                best_loss = total_loss
                logger.info(f'save the model in epoch{epoch}')
                model.save(os.path.join(cfg.train.log_dir, 'best.ckpt'))
                patience = 0
        if (epoch+1) % cfg.train.save_every == 0 and epoch != 0 and is_main_process():
            model.save(os.path.join(cfg.train.log_dir, f'{epoch}.ckpt'))
            save_epoch = epoch
            

        # if patience > cfg.train.patience:
        #     logger.info(f'early stop in epoch{epoch}')
        #     break

def train_epoch(cfg, model, train_loader, epoch):
    train_losses, total_loss = process_epoch(cfg, model, train_loader, 'train', epoch)
    if (not cfg.dist.distributed or is_main_process()) and train_losses:
        wandb.log({'epoch': epoch}, step=epoch)
        for k, v in train_losses.items():
            if len(v) == 0:
                break
            wandb.log({f'train_{k}_epoch': sum(v)/len(v)}, step=epoch)
            wandb.log({f'learning rate': model.scheduler.get_learning_rates()[0]}, step=epoch)
        logger.info(f'train loss: {total_loss} in epoch {epoch}')
    return train_losses, total_loss 

@torch.no_grad()
def eval_epoch(cfg, model, eval_loader, epoch):
    eval_losses, total_loss = process_epoch(cfg, model, eval_loader, 'eval', epoch)
    if (not cfg.dist.distributed or is_main_process()):
        for k, v in eval_losses.items():
            if len(v) == 0:
                break
            wandb.log({f'eval_{k}_epoch': sum(v)/len(v)}, step=epoch)
        logger.info(f'eval loss: {total_loss} in epoch {epoch}')
    return eval_losses, total_loss

def process_epoch(cfg, model, data_loader, mode, epoch):
    if mode == 'train':
        model.train()
    else:
        model.eval()
    losses = []
    total_loss = 0
    
    for i, batch in tqdm(enumerate(data_loader), desc=f'{mode} epoch {epoch}', total=len(data_loader), disable=not is_main_process()):
        batch = {k: v.to(device=cfg.dist.local_rank) for k, v in batch.items()}

        pred = model(batch)
        loss_dict = model.loss(pred, batch, cfg.network.loss)
        if loss_dict is not None:
            loss = loss_dict['loss_total']
            if loss==0 or torch.isnan(loss):
                if i==0:
                    losses = {k: [] for k, v in loss_dict.items()}
                continue
            if i==0:
                losses = {k: [v.item()] for k, v in loss_dict.items()}
            else:
                for k, v in loss_dict.items():
                    losses[k].append(v.item())
            total_loss = total_loss+loss.item()
            if mode == 'train':
                model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 
                model.optimizer.step()
        else:
            losses = []
            total_loss = 0
            break

    if mode == 'train':
        model.scheduler.step()

    total_loss = total_loss / (i+1)
    return losses, total_loss

@torch.no_grad()
def viz(cfg, model, train_loader, eval_loader, epoch):
    model.eval()
    viz_dir = cfg.train.log_dir+f'/viz_{epoch}'
    os.makedirs(viz_dir, exist_ok=True)
    viz_num_max = cfg.train.num_viz
    pbar = tqdm(total=viz_num_max*2, desc="Visualizing")
    with torch.no_grad():
        viz_num = 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device=cfg.dist.local_rank) for k, v in batch.items()}
            pred = model(batch)
            visualize_prediction(pred, batch, viz_dir, f'train_{epoch}_{i}')
            viz_num = viz_num + 1
            pbar.update(1)
            if viz_num >= viz_num_max:
                break
        viz_num = 0
        for i, batch in enumerate(eval_loader):
            batch = {k: v.to(device=cfg.dist.local_rank) for k, v in batch.items()}
            pred = model(batch)
            visualize_prediction(pred, batch, viz_dir, f'eval_{epoch}_{i}')
            viz_num = viz_num + 1
            pbar.update(1)
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
    cfg.train.log_dir = get_train_log_dir(cfg)
    logger = setup_logging(os.path.join(cfg.train.log_dir, 'train.log'))
    cfg.dist.distributed = False
    cfg.dist.local_rank = cfg.train.device
    main(cfg)
