from data_dynamic import PartNetPartDataset
import os
from tqdm import tqdm
import utils
from test_dynamic import forward, get_parser
import torch
import numpy as np
import random

cat_to_model = {
    'Chair': 'Chair',
    'Misc': 'Chair',
    'Table': 'Chair', #'Table',
    'Bench': 'Chair',
    'Shelf': 'Chair',
    'Desk': 'Chair', # 'Table',
}


def test(conf):
    data_features = ['part_pcs', 'part_poses', 'part_valids', 'shape_id', 'part_ids', 'contact_points', 'sym', 'pairs', 'match_ids']
    sum_part_cd_loss = 0
    sum_shape_cd_loss = 0
    sum_contact_point_loss = 0
    total_acc_num = 0
    total_valid_num = 0
    total_max_count = 0
    total_total_num = 0

    dataset = PartNetPartDataset(category=conf.category,
                                 # data_dir='../../my_prepare_data_0604_10k',
                                 data_dir='../../my_prepare_data_0608',
                                 data_fn=f'{conf.category}.npy',
                                 data_features=data_features, max_num_part=20, level='0')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=conf.batch_size, shuffle=False, pin_memory=True, num_workers=0, drop_last=False, #True,
        collate_fn=utils.collate_feats_with_none, worker_init_fn=utils.worker_init_fn)

    model_cat = cat_to_model[conf.category]
    model_def = utils.get_model_module('model_dynamic_backup')
    network = model_def.Network(conf)
    network.load_state_dict(torch.load(f"./checkpoints/{model_cat}/{model_cat}_net_network.pth"))
    network.cuda()
    network.eval()
    for p in network.parameters():
        p.requires_grad = False

    for batch_ind, batch in tqdm(enumerate(dataloader)):
        part_cd_loss, shape_cd_loss, contact_point_loss, acc_num, valid_num, max_count, total_num = forward(
            batch=batch, data_features=data_features, network=network, conf=conf, is_val=True, batch_ind=batch_ind,
        )
        sum_part_cd_loss += part_cd_loss
        sum_shape_cd_loss += shape_cd_loss
        sum_contact_point_loss += contact_point_loss
        total_acc_num += acc_num
        total_valid_num += valid_num
        total_max_count += max_count
        total_total_num += total_num

    val_num_batch = len(dataloader)
    total_max_count = total_max_count.float()
    total_total_num = float(total_total_num)
    total_shape_loss = sum_shape_cd_loss / val_num_batch
    total_part_loss = sum_part_cd_loss / val_num_batch
    total_contact_loss = sum_contact_point_loss / val_num_batch
    total_acc = total_acc_num / total_valid_num
    total_contact = total_max_count / total_total_num
    print('total_shape_loss:', total_shape_loss.item())
    print('total_part_loss:', total_part_loss.item())
    print('total_contact_loss:', total_contact_loss.item())
    print('total_acc:', 100 * total_acc.item())
    print('total_contact', total_contact)
    print(total_max_count, total_total_num)


if __name__ == "__main__":
    def set_seed(seed):
        print('setting seed to', seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    parser = get_parser()
    conf = parser.parse_args()
    set_seed(conf.seed)
    conf.exp_name = f'exp-{conf.category}-{conf.exp_suffix}'
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    test(conf)
