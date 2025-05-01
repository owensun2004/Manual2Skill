import ast
import copy
from permutations import print_trees
from manual_generation.models import Model, SinglePartModel, SimilarityModel
from utils.meters import Meters
from manual_generation.eval import eval_assembly_tree
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import argparse
from manual_generation.dataset import Dataset
from utils.data import build_tree_from_list, tree_to_list
import json
import sys
# sys.setrecursionlimit(30) 

check = True
num = 6

def evaluate(model: Model, dataset: Dataset, check_symm: bool):
    arr_meters = []
    arr_meters2 = []
    for i in range(2, 3):
        meters = Meters()
        meters2 = Meters()
        count = 0
        improved_count = 0
        for f in tqdm(dataset):
            #if f['part_ct']==i:
            #if f['part_ct']<=num:
            if f['part_ct']>num:
                count += 1
                tree_gt = f['tree']
                #print(f"{f['category']}/{f['name']}")
                #with open(f"../../out_trees_v9_revert/{f['category']}/{f['name']}/tree.json", "r") as file:
                with open(f"../../out_trees_v14_no_seg/{f['category']}/{f['name']}/tree.json", "r") as file:
                    tree_str = json.load(file)
                with open(f"../../out_trees_v22_no_num_no_seg/{f['category']}/{f['name']}/tree.json", "r") as file:
                # with open(f"../../out_trees_v20_no_crop/{f['category']}/{f['name']}/tree.json", "r") as file:
                    tree_str2 = json.load(file)
                tree_list = json.loads(tree_str)
                # print('pass')
                tree_list2 = json.loads(tree_str2)
                # print('pass2')
                # print(tree_list, tree_store)
                # print(tree_list, type(tree_list))
                tree_pred = build_tree_from_list(tree_list)
                # print('pass3')
                tree_pred2 = build_tree_from_list(tree_list2)
                if not check_symm:
                    print("USING ORIGINAL MODEL")
                tree_pred = model(f)
                tree_list = tree_to_list(tree_pred)
                result = eval_assembly_tree(tree_gt, tree_pred)
                result2 = eval_assembly_tree(tree_gt, tree_pred2)
                if check_symm and f['part_ct'] <= 10:

                    with open(f"../../output/{f['category']}/{f['name']}/equiv_parts.txt", "r") as text_file:
                        data = text_file.read()
                    nested_list = ast.literal_eval(data)
                    tree_store = []
                    tree_store2 = []
                    # print(tree_list, nested_list)
                    print_trees(copy.deepcopy(tree_list), nested_list, tree_store)
                    print_trees(copy.deepcopy(tree_list2), copy.deepcopy(nested_list), tree_store2)

                    max_score = find_max_val(result)
                    max_score2 = find_max_val(result2)
                    for indiv_tree in tree_store:
                        tree_pred_tmp = build_tree_from_list(indiv_tree)
                        result_tmp = eval_assembly_tree(tree_gt, tree_pred_tmp)

                        # Find largest value in result_tmp dictionary
                        largest_value = find_max_val(result_tmp)
                        
                        if largest_value > max_score:
                            max_score = largest_value
                            result = result_tmp
                    
                    for indiv_tree2 in tree_store2:
                        tree_pred_tmp2 = build_tree_from_list(indiv_tree2)
                        result_tmp2 = eval_assembly_tree(tree_gt, tree_pred_tmp2)

                        # Find largest value in result_tmp dictionary
                        largest_value = find_max_val(result_tmp2)
                        
                        if largest_value > max_score2:
                            max_score2 = largest_value
                            result2 = result_tmp2
                    
                    if result != eval_assembly_tree(tree_gt, build_tree_from_list(tree_list)):
                        print("=============================================")
                        print(f"predicted tree: {tree_list}")
                        print(f"similar parts: {nested_list}")
                        print(f"tree permutation: {tree_store}")
                        print(f"best result for {f['category']}/{f['name']}: {result}")
                        print(f"init result for {f['category']}/{f['name']}: {eval_assembly_tree(tree_gt, build_tree_from_list(tree_list))}")
                        improved_count += 1
                    
                    # if result != result2:
                    #     print(f"DIFFERENCE AT: {f['category']}/{f['name']}")
                    #     print(result)
                    #     print(result2)

                for k in result:
                    for k2 in result[k]:
                        meters.update(k + '_' + k2, result[k][k2])
                
                for k in result2:
                    for k2 in result2[k]:
                        meters2.update(k + '_' + k2, result2[k][k2])
        print(f"{i} parts count: {count}")
        print(improved_count)
        arr_meters.append(meters)
        arr_meters2.append(meters2)
    return arr_meters, arr_meters2

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


def find_max_no_child(data):
    # Find the largest value in the dictionary, excluding "no_children"
    largest_value = float('-inf')
    for key, value in data.items():
        if "no_children" not in key:  # Adjust the condition as per your needs
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
        meters_single, meters_single2 = evaluate(single_part_model, dataset, check)
        # for k, v in meters.avg_dict().items():
        #     print(k, end=' ')

    if 'similarity' in evaluate_models:
        similarity_model = SimilarityModel()
        meters, meters2 = evaluate(similarity_model, dataset, check)
        for i in range(len(meters)):
            print(f"Parts: {i+2}")
            pprint('Similarity Model:')
            pprint(meters[i].avg_dict())
            print(f"==========NO JSON BELOW=============")
            pprint(meters2[i].avg_dict())
            print(f"==========NO JSON ABOVE=============")
        # for k, v in meters.avg_dict().items():
        #     print(k, end=' ')
    for i in range(len(meters_single)):
        print(f"Parts: {i+2}")
        pprint('Single Part Model:')
        pprint(meters_single[i].avg_dict())
        pprint(find_max_no_child(meters_single[i].avg_dict()))
        print(f"==========NO JSON BELOW=============")
        pprint(meters_single2[i].avg_dict())
        pprint(find_max_no_child(meters_single2[i].avg_dict()))
        print(f"==========NO JSON ABOVE=============")
    
    num_parts = range(2, 2 + len(meters_single))
    max_values_single = [find_max_no_child(d.avg_dict()) for d in meters_single]
    max_values_single2 = [find_max_no_child(d.avg_dict()) for d in meters_single2]


    # Create the bar chart
    x = np.arange(len(meters_single))  # The x locations for the groups
    width = 0.35  # The width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars for `meters_single` and `meters_single2`
    rects1 = ax.bar(x - width / 2, max_values_single, width, label='Meters Single')
    rects2 = ax.bar(x + width / 2, max_values_single2, width, label='Meters Single2')

    # Add labels, title, and legend
    ax.set_xlabel('Number of Parts')
    ax.set_ylabel('Highest Metric Value')
    ax.set_title('Highest Metric Values by Number of Parts')
    ax.set_xticks(x)
    ax.set_xticklabels(num_parts)
    ax.legend()

    # Annotate the bars with their values
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4g}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Offset text above the bar
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Display the plot
    plt.tight_layout()
    plt.show()
