source /svl/u/rcwang/miniconda3/bin/activate; conda activate partassembly2
cd exps/Our_Method-dynamic_graph_learning || exit
python test_dynamic.py  \
    --exp_suffix '' \
    --model_version 'model_dynamic_backup' \
    --category 'Chair' \
    --train_data_fn 'Chair.train.npy' \
    --val_data_fn "Chair.val.npy" \
    --device cuda:0 \
    --model_dir "./checkpoints/Chair/Chair_net_network.pth"\
    --level 3 \


#(partassembly2) yzzhang@viscam1:/viscam/u/yzzhang/projects/assembly/exps/Our_Method-dynamic_graph_learning$ python test_dynamic.py      --exp_suffix ''     --model_version 'model_dynamic_backup'     --category 'Chair'     --train_data_fn 'Chair.train.npy'     --val_data_fn "Chair.val.npy"     --model_dir "./checkpoints/Chair/Chair_net_network.pth"    --level 3     --batch_size 4     --num_batch_every_visu 0  --batch_size 1 --iter 1
#conf Namespace(batch_size=1, category='Chair', checkpoint_interval=10000, console_log_interval=10, data_dir='../../prepare_data', device='cuda:0', epochs=1000, exp_dir='logs/exp-Chair-model_dynamic_backup-level3', exp_name='exp-Chair-model_dynamic_backup-level3', exp_suffix='', feat_len=256, iter=1, iter_to_test=4, level='3', log_dir='logs', loss_weight_rot_cd=10.0, loss_weight_rot_l2=1.0, loss_weight_trans_l2=1.0, lr=0.001, lr_decay_by=0.9, lr_decay_every=5000, max_num_part=20, model_dir='./checkpoints/Chair/Chair_net_network.pth', model_version='model_dynamic_backup', no_console_log=False, no_tb_log=False, no_visu=False, num_batch_every_visu=0, num_epoch_every_visu=1, num_workers=5, overwrite=False, seed=3124256514, train_data_fn='Chair.train.npy', val_data_fn='Chair.val.npy', weight_decay=1e-05)
#total_shape_loss: 0.015677927061915398
#total_part_loss: 0.1720169484615326
#total_contact_loss: 3.5844058990478516
#total_acc: 20.64211815595627
#total_contact tensor(0.1131)
#tensor(2030.) 17956.0

#(partassembly2) yzzhang@viscam1:/viscam/u/yzzhang/projects/assembly/exps/Our_Method-dynamic_graph_learning$ python my_test.py --category Chair --batch_size 32
#100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 58/58 [40:13<00:00, 41.60s/it]
#total_shape_loss: 0.014483940787613392
#total_part_loss: 0.28314337134361267
#total_contact_loss: 1.9007649421691895
#total_acc: 5.833333358168602
#total_contact tensor(0.0921)
#tensor(102.) 1108.0

#(partassembly2) yzzhang@viscam1:/viscam/u/yzzhang/projects/assembly/exps/Our_Method-dynamic_graph_learning$ python my_test.py --category Chair --batch_size 1 --no_visu
#57it [00:24,  2.30it/s]
#total_shape_loss: 0.014764434657990932
#total_part_loss: 0.2814958095550537
#total_contact_loss: 1.8758370876312256
#total_acc: 7.758620381355286
#total_contact tensor(0.1825)
#tensor(192.) 1052.0