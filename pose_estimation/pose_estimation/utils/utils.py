import torch
import logging
import os
import subprocess
import torch.distributed as dist
import pytorch3d.ops as torch3d_ops
from datetime import datetime

#---distributed and init---
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)      # If True, print from all nodes, False for only master node
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def init_distributed_mode(cfg):
    if cfg.dist.dist_on_itp:
        cfg.dist.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        cfg.dist.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        cfg.dist.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        cfg.dist.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(cfg.dist.gpu)
        os.environ['RANK'] = str(cfg.dist.rank)
        os.environ['WORLD_SIZE'] = str(cfg.dist.world_size)
    elif 'SLURM_PROCID' in os.environ:
        cfg.dist.rank = int(os.environ['SLURM_PROCID'])
        cfg.dist.gpu = int(os.environ['SLURM_LOCALID'])
        cfg.dist.world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(cfg.dist.rank)
        os.environ['LOCAL_RANK'] = str(cfg.dist.gpu)
        os.environ['WORLD_SIZE'] = str(cfg.dist.world_size)

        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.dist.rank = int(os.environ["RANK"])
        cfg.dist.world_size = int(os.environ['WORLD_SIZE'])
        cfg.dist.gpu = int(os.environ['LOCAL_RANK'])

    cfg.dist.dist_backend = 'nccl'
    torch.distributed.init_process_group(backend=cfg.dist.dist_backend, init_method=cfg.dist.dist_url,
                                         world_size=cfg.dist.world_size, rank=cfg.dist.rank)
    torch.distributed.barrier()
    assert torch.distributed.is_initialized()
    setup_for_distributed(cfg.dist.rank == 0)
    cfg.dist.local_rank = get_rank()
    torch.cuda.set_device(cfg.dist.local_rank)
    logging.info('| distributed init (Init process group: backend: {}, world_size: {}, rank {}, local Rank: {}): {}, gpu {}'.format(
        cfg.dist.dist_backend, cfg.dist.world_size, cfg.dist.rank, cfg.dist.local_rank, cfg.dist.dist_url, cfg.dist.gpu))

def setup_logging(log_file_path):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False  # Prevent logging propagation to root logger

    return logger

def get_train_log_dir(cfg):
    """ Only generate train.log_dir in the main process and broadcast it to all processes. """
    log_dir = None

    if is_main_process():
        log_dir_suffix = datetime.now().strftime("%m-%d-%H-%M-%S")
        log_dir = f'./logs/{cfg.network_name}_{cfg.data.dir.rstrip("/").split("/")[-1]}_{log_dir_suffix}/'
        os.makedirs(log_dir, exist_ok=True)

    # Ensure all processes use the same log_dir in distributed training
    if cfg.dist.distributed:
        log_dir = broadcast_log_value(log_dir)

    return log_dir

def broadcast_log_value(value, src=0):
    """ Use PyTorch distributed communication to ensure all processes get the same string value. """
    if not dist.is_initialized():
        return value  # Return directly if distributed environment is not initialized

    # Only let the main process send data, other processes receive
    if is_main_process():
        value_tensor = torch.tensor(bytearray(value, 'utf-8'), dtype=torch.uint8, device="cuda" if torch.cuda.is_available() else "cpu")
        size = torch.tensor([len(value_tensor)], dtype=torch.int, device=value_tensor.device)
    else:
        size = torch.tensor([0], dtype=torch.int, device="cuda" if torch.cuda.is_available() else "cpu")

    # Broadcast data size
    dist.broadcast(size, src=src)

    # Non-main processes create buffer
    if not is_main_process():
        value_tensor = torch.empty(size.item(), dtype=torch.uint8, device="cuda" if torch.cuda.is_available() else "cpu")

    # Broadcast data content
    dist.broadcast(value_tensor, src=src)

    # Decode back to string
    return value_tensor.cpu().numpy().tobytes().decode('utf-8')

def broadcast_value(value, src=0):
    """ Use PyTorch distributed communication to broadcast integer value. """
    if not dist.is_initialized():
        return value  # Return directly if distributed environment is not initialized

    if is_main_process():
        value_tensor = torch.tensor([value], dtype=torch.int, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        value_tensor = torch.tensor([0], dtype=torch.int, device="cuda" if torch.cuda.is_available() else "cpu")

    # Broadcast integer value
    dist.broadcast(value_tensor, src=src)

    return value_tensor.item()  # Convert back to Python integer

#---data_processing---

def mean_max_pool(point_cloud_features, valid_mask=None):
    """
    Perform mean and max pooling on the point cloud features, excluding padding data.
    Args:
        point_cloud_features: Tensor of shape (B, N, D), where N is the number of parts.
        valid_mask: Optional mask of shape (B, N), where 1 indicates valid parts and 0 indicates padding.
    Returns:
        global_feature: Tensor of shape (B, D*2), pooled global features.
    """
    if valid_mask is not None:
        # Mask the features by multiplying with valid_mask
        point_cloud_features = point_cloud_features * valid_mask.unsqueeze(-1)  # (B, N, D)

        # Avoid dividing by zero for mean computation
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
        mean_values = point_cloud_features.sum(dim=1) / valid_counts  # (B, D)

        # For max pooling, set invalid entries to a very small value
        masked_features = point_cloud_features.masked_fill(valid_mask.unsqueeze(-1) == 0, -float('inf'))
        max_values, _ = masked_features.max(dim=1)  # (B, D)
    else:
        mean_values = point_cloud_features.mean(dim=1)  # (B, D)
        max_values, _ = point_cloud_features.max(dim=1)  # (B, D)

    global_feature = torch.cat([mean_values, max_values], dim=1)  # (B, D*2)
    return global_feature

def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if num_params > 1e3 and num_params < 1e6:
        num_params = f'{num_params/1e3:.2f}K'
    elif num_params > 1e6:
        num_params = f'{num_params/1e6:.2f}M'
    return num_params

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0).cpu()
    else:
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
    return sampled_points, indices

def find_partition(node, partitions):
    for i, partition in enumerate(partitions):
        if node in partition:
            return i
    return -1  # Default return -1, shouldn't reach here

if __name__ == "__main__":
    a = torch.rand(4, 256)
    b = torch.rand(4, 256)
    t = torch.stack((a, b), dim=1)
    global_feature = mean_max_pool(t)
