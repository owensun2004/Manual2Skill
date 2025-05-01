import pickle
import numpy as np
import random
import json
import os
from utils.data import Node, build_tree_from_list


class Dataset:
    def __init__(self, data_json, parts_dir, shuffle=False, part_features_pkl=None):
        self.part_features = None
        if part_features_pkl is not None and part_features_pkl != '':
            with open(part_features_pkl, 'rb') as f:
                self.part_features = pickle.load(f)
        with open(data_json) as f:
            self.data = json.load(f)
        if shuffle:
            random.shuffle(self.data)
        self.parts_dir = parts_dir

    def __iter__(self):
        for e in self.data:
            cat = e['category']
            name = e['name']
            parts_dir_this = os.path.join(self.parts_dir, cat, name)
            part_obj_paths = [os.path.join(parts_dir_this, p) for p in os.listdir(parts_dir_this)]
            part_ct = e['parts_ct']
            feed_dict = {
                'category': cat,
                'name': name,
                'tree': build_tree_from_list(e['assembly_tree']),
                'part_ct': part_ct,
                'part_obj_paths': part_obj_paths
            }
            if self.part_features is not None:
                try:
                    feed_dict['features'] = np.array([
                        self.part_features[(cat, name)][str(i).zfill(2)] for i in range(part_ct)
                    ])
                except KeyError as e:
                    print('Missing part features for:', cat, name)
                    continue
            yield feed_dict

    def __next__(self):
        raise StopIteration

    def __len__(self):
        return len(self.data)