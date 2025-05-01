import ast
import copy
from permutations import print_trees
from manual_generation.models import Model, SinglePartModel, SimilarityModel
from utils.meters import Meters
from manual_generation.eval import eval_assembly_tree
from tqdm import tqdm
from pprint import pprint
import argparse
from manual_generation.dataset import Dataset
from utils.data import build_tree_from_list
import json
import sys
# sys.setrecursionlimit(30) 

check = True
num = 6

def evaluate(model: Model, dataset: Dataset, check_symm: bool):
    meters = Meters()
    count = 0
    improved_count = 0
    for f in tqdm(dataset):
        if f['part_ct']<=num:
        # if f['part_ct']>=7:
        # if f['name'] == 'glenn':
            count += 1
            tree_gt = f['tree']
            print(f"{f['category']}/{f['name']}")
            #with open(f"../../out_trees_v9_revert/{f['category']}/{f['name']}/tree.json", "r") as file:
            with open(f"../../out_trees_v11_no_seg/{f['category']}/{f['name']}/tree.json", "r") as file:
                tree_str = json.load(file)
            tree_list = json.loads(tree_str)
            # print(tree_list, tree_store)
            # print(tree_list, type(tree_list))
            tree_pred = build_tree_from_list(tree_list)
            if not check_symm:
                print("USING ORIGINAL MODEL")
                tree_pred = model(f)
            result = eval_assembly_tree(tree_gt, tree_pred)
            if check_symm:

                with open(f"../../output/{f['category']}/{f['name']}/equiv_parts.txt", "r") as text_file:
                    data = text_file.read()
                nested_list = ast.literal_eval(data)
                tree_store = []
                # print(tree_list, nested_list)
                print_trees(copy.deepcopy(tree_list), nested_list, tree_store)

                max_score = find_max_val(result)
                for indiv_tree in tree_store:
                    tree_pred_tmp = build_tree_from_list(indiv_tree)
                    result_tmp = eval_assembly_tree(tree_gt, tree_pred_tmp)

                    # Find largest value in result_tmp dictionary
                    largest_value = find_max_val(result_tmp)
                    
                    if largest_value > max_score:
                        max_score = largest_value
                        result = result_tmp
                
                if result != eval_assembly_tree(tree_gt, build_tree_from_list(tree_list)):
                    print("=============================================")
                    print(f"predicted tree: {tree_list}")
                    print(f"similar parts: {nested_list}")
                    print(f"tree permutation: {tree_store}")
                    print(f"best result for {f['category']}/{f['name']}: {result}")
                    print(f"init result for {f['category']}/{f['name']}: {eval_assembly_tree(tree_gt, build_tree_from_list(tree_list))}")
                    improved_count += 1

            for k in result:
                for k2 in result[k]:
                    meters.update(k + '_' + k2, result[k][k2])
    print(f"{num} parts count: {count}")
    print(improved_count)
    return meters

def find_max_val(dict):
    # Find largest value in result_tmp dictionary
    largest_value = float('-inf')
    for key, subdict in dict.items():
        if key != "no_children":
            for metric, value in subdict.items():
                # Update the largest value if the current value is greater
                if value > largest_value:
                    largest_value = value
    return largest_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_features_pkl', type=str)
    parser.add_argument('--data_json', type=str)
    parser.add_argument('--parts_dir', type=str)
    args = parser.parse_args()

    dataset = Dataset(data_json=args.data_json, parts_dir=args.parts_dir, part_features_pkl=args.part_features_pkl)
    evaluate_models = ['single_part', 'similarity']
    # evaluate_models = ['single_part']
    # evaluate_models = ['similarity']
    if 'single_part' in evaluate_models:
        single_part_model = SinglePartModel()
        meters_single = evaluate(single_part_model, dataset, check)
        # for k, v in meters.avg_dict().items():
        #     print(k, end=' ')

    if 'similarity' in evaluate_models:
        similarity_model = SimilarityModel()
        meters = evaluate(similarity_model, dataset, check)
        pprint('Similarity Model:')
        pprint(meters.avg_dict())
        # for k, v in meters.avg_dict().items():
        #     print(k, end=' ')
    pprint('Single Part Model:')
    pprint(meters_single.avg_dict())
