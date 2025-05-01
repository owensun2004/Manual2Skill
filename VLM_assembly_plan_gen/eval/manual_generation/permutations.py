from itertools import permutations

def print_trees(tree, nested_list, holder):
    if not nested_list:
        return
    ele = nested_list[0]
    l = list(permutations(ele))  # Generate permutations for the current list
    pointer_dict = {}
    
    for i, e in enumerate(ele):
        path = find_position(tree, e)
        if path is None:
            pointer_dict["pt" + str(i)] = None  # Mark as None if not found
        else:
            pointer_str = f"tree{''.join([f'[{idx}]' for idx in path])}"
            pointer_dict["pt" + str(i)] = pointer_str
        
    for perm in l:
        skip = False
        for j, val in enumerate(perm):
            key = f"pt{j}"
            if pointer_dict[key] is None:  # Skip if any value is missing
                skip = True
                break
            exec(f"{pointer_dict[key]} = {val}")
        
        if not skip:
            holder.append(eval(repr(tree)))  # Use eval(repr(...)) to deep copy
        
        # Continue the recursion
        print_trees(tree, nested_list[1:], holder)


def find_position(tree, target, path=None):
    """
    Recursively finds the path to the target in a nested list.
    """
    if path is None:
        path = []
    if isinstance(tree, list):
        for i, item in enumerate(tree):
            new_path = path + [i]
            if item == target:
                return new_path
            elif isinstance(item, list):
                result = find_position(item, target, new_path)
                if result is not None:
                    return result
    return None