from .estimation_net import PoseEstimationNetwork
from .gnn_net import GNNNetwork
from .learning_3d_part_assembly import Learning3DPartAssembly


def build_model(cfg):
    if cfg.network_name == 'PoseEstimationNetwork':
        return PoseEstimationNetwork(cfg)
    elif cfg.network_name == 'GNNNetwork':
        return GNNNetwork(cfg)
    elif cfg.network_name == '3DPartAssembly':
        return Learning3DPartAssembly(cfg)
    
    
    raise ValueError(f"Network {cfg.network_name} not supported.")