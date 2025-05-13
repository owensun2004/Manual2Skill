from yacs.config import CfgNode as CN
import time
from datetime import datetime

_C = CN()
# _C.network_name = 'PoseEstimationNetwork'
_C.network_name = 'GNNNetwork'
# _C.network_name = '3DPartAssembly'

_C.network = CN()
_C.network.hidden_size = 128
_C.network.loss = ['mse', 'geo', 'cham', 'penalty']
_C.network.rot_loss_weight = 1.0
_C.network.trans_loss_weight = 1
_C.network.chamfer_loss_weight = 1
_C.network.match_loss_weight = 1
_C.network.mse_loss_weight = 20
_C.network.penalty_loss_weight = 0.1
_C.network.real_world = False


_C.data = CN()
_C.data.train_ratio = 0.7
_C.data.seed = 42

_C.data.dir = '/data2/lyw/IKEA_data3'
_C.data.min_num_part = 2
_C.data.max_num_part = 6
_C.data.is_partnet = True

_C.dist = CN()
_C.dist.distributed = False
_C.dist.world_size = 1
_C.dist.dist_on_itp = False
_C.dist.dist_url = 'env://'

_C.train = CN()
_C.train.batch_size = 22
_C.train.warmup_epochs = 10
_C.train.num_epochs = 500
_C.train.eval_every = 50
_C.train.save_every = 50
_C.train.viz_every = 50
_C.train.num_viz = 50
_C.train.lr = 1e-5
_C.train.l2_norm = 1e-7
_C.train.pretrained = True
_C.train.pretrained_weights = '/data2/lyw/Furniture-Assembly/pose_estimation/logs/GNNNetwork_IKEA_data3_04-04-11-55-07/best.ckpt'
_C.train.patience = 15
_C.train.device = 'cuda:1' if not _C.dist.distributed else 'cuda:0'
_C.train.log_interval = 10

def get_cfg_defaults():
    return _C.clone()