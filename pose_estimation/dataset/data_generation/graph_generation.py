import os
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from igraph import Graph
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import json
from collections import deque
import argparse
import networkx as nx
from pdb import set_trace as bp
random.seed(0)
np.random.seed(0)


colors_hex = [
        '#5A9BD5', '#FF6F61', '#77B77A', '#A67EB1', '#FF89B6', '#FFB07B',
        '#C5A3CF', '#FFA8B6', '#A3C9E0', '#FFC89B', '#E58B8B',
        '#A3B8D3', '#D4C3E8', '#66B2AA', '#E4A878', '#6882A4', '#D1AEDD', '#E8A4A6',
        '#A5DAD7', '#C6424A', '#E1D1F4', '#FFD8DC', '#F4D49B', '#8394A8'
    ]

def hex_to_rgb(hex_color):
    """
    Convert hexadecimal color code to RGB tuple.
    :param hex_color: Hexadecimal color code (e.g. '#FFFFFF')
    :return: RGB color tuple with values from 0-255
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def load_obj_files(directory):
    meshes = []
    file_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.obj'):
            # Add the absolute path to the file list
            # print(filename)
            file_list.append(os.path.abspath(os.path.join(directory, filename)))
    file_list.sort()
    for filename in file_list:
        mesh = trimesh.load(os.path.join(directory, filename))
        meshes.append(as_mesh(mesh))
    return meshes, file_list

def sample_points(mesh, num_points=1000):
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def compute_edges(points_list, threshold):
    edges = []
    for i, points1 in enumerate(points_list):
        tree1 = cKDTree(points1)
        for j, points2 in enumerate(points_list):
            if i >= j:
                continue
            tree2 = cKDTree(points2)
            distances, _ = tree1.query(tree2.data, k=1)
            if np.min(distances) < threshold:
                edges.append((i, j))
    return edges

def find_nearby_points(pts1, pts2, num_points=10):
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    distances, indices1 = tree1.query(tree2.data, k=num_points)
    distances, indices2 = tree2.query(tree1.data, k=num_points)
    return indices1, indices2

def find_nearby_points_in_ball(pts1, pts2, radius=0.1):
    """
    Find the closest point pair in the two point clouds and take the midpoint of the closest point pair as the center,
    find all the points within the range of the ball
    
    :param pts1: Point cloud 1, with the shape of (N, 3).
    :param pts2: Point cloud w, with the shape of (N, 3).
    :param radius: The radius of the ball.
    :return: The points within the range of the small ball (from two point clouds).
    """
    # Construct the KD tree
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)

    # Search for the nearest point pair
    distances1, indices1 = tree1.query(pts2, k=1)  # The nearest point from pts2 to pts1
    min_distance_idx = np.argmin(distances1)  # The index of the most recent point
    point1 = pts1[indices1[min_distance_idx]]  # The nearest point in point cloud 1
    point2 = pts2[min_distance_idx]            # The nearest point in point cloud 2

    # Calculate the center of the small ball (the midpoint of the nearest point pair)
    ball_center = (point1 + point2) / 2

    return tree1.query_ball_point(ball_center, radius), tree2.query_ball_point(ball_center, radius)


# Helper function: Use BFS to check the connectivity of the graph
def is_connected(graph, remaining_nodes, start_node):
    visited = set()
    queue = deque([start_node])
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            # Traverse all the edges adjacent to the current node
            for (n1, n2) in graph:
                if n1 == node and n2 in remaining_nodes and n2 not in visited:
                    queue.append(n2)
                elif n2 == node and n1 in remaining_nodes and n1 not in visited:
                    queue.append(n1)
    
    # If the number of visited nodes is equal to the number of remaining nodes, it indicates that the graph is connected
    return len(visited) == len(remaining_nodes)

# Randomly delete nodes and maintain the connectivity of the graph
def remove_nodes(graph, nodes, remove_count, max_attempts=100):
    mask = [1 for node in nodes]
    # The set of deleted nodes
    deleted_nodes = set()
    current_attempt = 0
    while len(deleted_nodes) < remove_count and current_attempt < max_attempts:
        current_attempt += 1
        # Randomly select the nodes to be deleted
        nodes_to_remove = random.sample([node for node in nodes if node not in deleted_nodes], remove_count - len(deleted_nodes))
        
        # After deleting a node, the edge set of the graph needs to be recalculated
        new_graph = [(n1, n2) for (n1, n2) in graph if n1 not in nodes_to_remove and n2 not in nodes_to_remove]
        
        # Check whether the graph is connected
        if is_connected(new_graph, set(nodes) - deleted_nodes - set(nodes_to_remove), nodes[0]):
            deleted_nodes.update(nodes_to_remove)  
            for node in nodes_to_remove:
                mask[node] = 0
        else:
            continue

    # Return the deleted node mask
    return mask
    
def segment_graph(edges, nodes, nodes_mask, max_segment, max_attempts=100):
    valid_nodes = [node for node in nodes if nodes_mask[node] == 1]
    max_segment = min(max_segment, len(valid_nodes))
    # print(f"max_segment: {max_segment}")
    # bp()
    n_groups = random.randint(2, max_segment)
    valid_edges = [(i, j) for (i, j) in edges if nodes_mask[i] == 1 and nodes_mask[j] == 1]
    graph = nx.Graph()
    graph.add_nodes_from(valid_nodes)
    graph.add_edges_from(valid_edges)
    num_attempts = 0
    while num_attempts < max_attempts:
        success = True
        random.shuffle(valid_nodes)
        # Randomly generate the sizes of each group to ensure that the total sum is equal to the length of the list
        group_sizes = []
        remaining_elements = len(valid_nodes)
        
        for i in range(n_groups - 1):
            size = random.randint(1, remaining_elements - (n_groups - i - 1))  # Make sure the remaining groups have elements
            group_sizes.append(size)
            remaining_elements -= size
        
        # The last group contains all the remaining elements
        group_sizes.append(remaining_elements)
        
        # Assign elements to each group
        partitions = []
        start = 0
        for size in group_sizes:
            partitions.append(valid_nodes[start:start + size])
            start += size
    
        # Create a new graph with only the retained edges
        new_graph = graph.copy()
        
        # Traverse all edges and delete the edges across subgraphs
        edges_to_remove = []
        for u, v in graph.edges:
            # Determine whether the edge (u, v) connects two different subsets
            u_partition = find_partition(u, partitions)
            v_partition = find_partition(v, partitions)
            
            if u_partition != v_partition:
                edges_to_remove.append((u, v))  # Edges across subgraphs are marked as deleted
        
        # Delete the edges
        new_graph.remove_edges_from(edges_to_remove)
        
        # Make sure that each subgraph is connected
        for i in range(n_groups):
            subgraph_nodes = partitions[i]
            subgraph = new_graph.subgraph(subgraph_nodes).copy()
            if not nx.is_connected(subgraph):
                success = False
                break
        if success:
            return partitions
        num_attempts += 1
    return None


# Find the partition to which the node belongs
def find_partition(node, partitions):
    for i, partition in enumerate(partitions):
        if node in partition:
            return i
    return -1  # The default return is -1. It shouldn't be here


def find_junction_points(points_list, edges, nodes_mask, groups):
    junction_mask = [np.zeros(points.shape[0])-1 for points in points_list]
    junction_idx = 0
    for edge in edges:
        i, j = edge
        if nodes_mask[i] == 0 or nodes_mask[j] == 0:
            continue
        group_i = find_partition(i, groups)
        group_j = find_partition(j, groups)
        if group_i == group_j:
            continue
        points1 = points_list[i]
        points2 = points_list[j]
        # indices1, indices2 = find_nearby_points(points1, points2)
        indices1, indices2 = find_nearby_points_in_ball(points1, points2)
        junction_mask[i][indices1] = junction_idx
        junction_mask[j][indices2] = junction_idx
        junction_idx += 1
    return junction_mask



def visualize_graph(points_list, edges, nodes_mask, groups=None):
    graph = Graph(directed=False)
    graph.add_vertices(len(points_list))
    graph.add_edges(edges)
    layout = graph.layout("auto")

    # Extract metadata and positions
    positions = {i: layout[i] for i in range(len(graph.vs))}
    Y = [layout[k][1] for k in range(len(graph.vs))]
    M = max(Y)

    Xn = [positions[k][0] for k in positions]
    Yn = [2 * M - positions[k][1] for k in positions]

    # Separate nodes and edges based on mask
    mask_1_nodes = [i for i in range(len(nodes_mask)) if nodes_mask[i] == 1]
    mask_0_nodes = [i for i in range(len(nodes_mask)) if nodes_mask[i] == 0]
    
    mask_1_edges = []
    mask_0_edges = []
    
    for edge in edges:
        # If either of the nodes in the edge is a mask 0 node, color this edge differently
        if edge[0] in mask_0_nodes or edge[1] in mask_0_nodes:
            mask_0_edges.append(edge)
        else:
            mask_1_edges.append(edge)
    # print(len(mask_1_edges), len(mask_0_edges), len(edges))
    # Prepare the edge data
    Xe, Ye, edge_colors = [], [], []
    for edge in mask_1_edges:
        Xe += [positions[edge[0]][0], positions[edge[1]][0], None]
        Ye += [2 * M - positions[edge[0]][1], 2 * M - positions[edge[1]][1], None]
        edge_colors.append('rgb(255,0,0)')  # Default edge color for mask 1 nodes
    
    for edge in mask_0_edges:
        Xe += [positions[edge[0]][0], positions[edge[1]][0], None]
        Ye += [2 * M - positions[edge[0]][1], 2 * M - positions[edge[1]][1], None]
        edge_colors.append('rgb(210,210,210)')  # Edge color for edges connected to mask 0 nodes

    # Prepare the node data (positions, color, etc.)
    Xn_all, Yn_all, node_colors_all = [], [], []
    for i, points in enumerate(points_list):
        # If mask is 1, keep default color, otherwise set to gray
        if nodes_mask[i] == 0:
            color = 'rgb(210,210,210)'
        else:
            if groups is None:
                color = 'rgb(255,0,0)'
            else:
                group_id = find_partition(i, groups)
                color = hex_to_rgb(colors_hex[group_id % len(colors_hex)])
                color = f'rgb{color}'
        Xn_all.append(positions[i][0])
        Yn_all.append(2 * M - positions[i][1])
        node_colors_all.append(color)
    # Create the plot
    fig1 = go.Figure()

    # Add edges (with different colors based on node mask)
    for edge, color in zip(mask_1_edges + mask_0_edges, edge_colors):
        Xe, Ye = [], []
        Xe += [positions[edge[0]][0], positions[edge[1]][0], None]
        Ye += [2 * M - positions[edge[0]][1], 2 * M - positions[edge[1]][1], None]
        fig1.add_trace(go.Scatter(x=Xe, y=Ye, mode='lines', line=dict(color=color, width=1), hoverinfo='none'))

    # Add nodes (with different colors based on mask)
    fig1.add_trace(go.Scatter(x=Xn_all, y=Yn_all, mode='markers', name='Nodes', 
                                marker=dict(symbol='circle-dot', size=20, color=node_colors_all, 
                                            line=dict(color='rgb(50,50,50)', width=0.5)), 
                                text=[f'Node {i}' for i in range(len(nodes_mask))], hoverinfo='text', opacity=0.8))


    # Update layout
    fig1.update_layout(title='Graph with Masked Nodes and Edges', 
                      showlegend=False, hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showline=False, showgrid=False, zeroline=False),
                      yaxis=dict(showline=False, showgrid=False, zeroline=False))

    return fig1



def visualize_pts(points_list, edges, nodes_mask, junction_mask=None, groups=None):
    fig2 = go.Figure()
    for i, points in enumerate(points_list):
        if nodes_mask[i] == 0:
            color = 'gray'
            fig2.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                    mode='markers', name=f'Object {i}', marker=dict(size=2, color=color)))
        else:
            if groups is None:
                fig2.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                        mode='markers', name=f'Object {i}', marker=dict(size=2)))
            else:
                group_id = find_partition(i, groups)
                color = hex_to_rgb(colors_hex[group_id % len(colors_hex)])
                fig2.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                        mode='markers', name=f'Object {i}, group {group_id}', marker=dict(size=2, color=f'rgb{color}')))
            if not junction_mask is None:
                for j in range(int(junction_mask[i].max())+1):
                    mask = junction_mask[i] == j
                    junction_points = points[mask]
                    random_color = np.random.randint(0, 255, 3)
                    fig2.add_trace(go.Scatter3d(x=junction_points[:, 0], y=junction_points[:, 1], z=junction_points[:, 2],
                                    mode='markers', name=f'Junction{i}-{j}', marker=dict(size=2, color=f'rgb({random_color[0]},{random_color[1]},{random_color[2]})'))
                                    )
    for edge in edges:
        i, j = edge
        points1 = points_list[i]
        points2 = points_list[j]
        points1_center = np.mean(points1, axis=0)
        points2_center = np.mean(points2, axis=0)
        fig2.add_trace(go.Scatter3d(x=[points1_center[0], points2_center[0]],
                                   y=[points1_center[1], points2_center[1]],
                                   z=[points1_center[2], points2_center[2]],
                                   mode='lines', name=f'Edge {i}-{j}'))
    # fig2.show()
    # fig2.write_html('graph.html')
    return fig2

def visualize_furniture_structure(points_list, edges, nodes_mask, visualize=False, junction_mask=None, groups=None):

    fig2 = visualize_graph(points_list, edges, nodes_mask, groups)
    fig1 = visualize_pts(points_list, edges, nodes_mask, junction_mask, groups)
    # Use make_subplots to create a subplot with two columns
    fig = make_subplots(
        rows=1, cols=2,  # 1行2列的布局
        specs=[[{'type': 'scatter3d'}, {'type': 'xy'}]],  # The first column is a 3D subgraph, and the second column is an XY subgraph
        subplot_titles=('3D Scatter', '2D graph')  # Subgraph title
    )

    for trace in fig1.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig2.data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(
        title_text="visualization of furniture structure",
        showlegend=True
    )
    if visualize:
        fig.show()
    return fig

def main(args):
    # directory = 'dataset/parts/Chair/poang_2'
    # directory = '/home/crtie/crtie/IKEA_manual/parts/Bench/applaro'
    directory = args.input_data_dir

    threshold = 0.06 # this should be adjusted based on the size of the parts
    max_attempts = 100
    max_segment = 5
    meshes, file_list = load_obj_files(directory)
    points_list = [sample_points(mesh) for mesh in meshes]
    nodes = list(range(len(points_list)))
    edges = compute_edges(points_list, threshold)
    num_generated_parts_selection = 0
    saved_data_path = []
    current_attempt = 0
    while num_generated_parts_selection < args.num_parts_selection and current_attempt < max_attempts:
        current_attempt += 1
        num_deleted_nodes = random.randint(0, len(nodes) - 2)
        # Delete some nodes randomly
        nodes_mask = remove_nodes(edges, nodes, num_deleted_nodes)
        # print(f"nodes mask: {nodes_mask}")
        valid_file_list = [file_list[i] for i in range(len(file_list)) if nodes_mask[i] == 1]
        # print(f"valid parts: {valid_file_list}")
        # Take out the index of 1 in the nodes mask and convert it to a string
        nodes_mask_str = ''.join([str(i) for i in range(len(nodes_mask)) if nodes_mask[i] == 1])
        # Divide the entire graph into multiple connected subgraphs
        groups = segment_graph(edges, nodes, nodes_mask, max_segment)
        if groups is None:
            continue
        # sort the groups
        for i in range(len(groups)):
            groups[i] = sorted(groups[i])
        groups = sorted(groups, key=lambda x: x[0])
        groups_str = ''
        for i in range(len(groups)):
            groups_str += ''.join([str(j) for j in groups[i]]) + '_'
        groups_str = groups_str[:-1]
        # Calculate the points of junction in the valid point cloud
        junction_mask = find_junction_points(points_list, edges, nodes_mask, groups)

        fig = visualize_furniture_structure(points_list, edges, nodes_mask, args.visualize, junction_mask, groups)

        data_id = directory.split('/')[-2] + '_' + directory.split('/')[-1] + '__' + groups_str
        data = {
        "id": data_id,
        "valid_parts": len(valid_file_list),
        "used meshes": [],
        "groups": groups,
            }
        save_path = args.save_data_dir + '/' + data_id + '/'
        # print(f"save path: {save_path}")
        os.makedirs(save_path, exist_ok=True)
        fig.write_html(save_path + 'furniture_structure.html')
        if os.path.exists(os.path.join(save_path, 'part_selection.json')):
            continue
        for i in range(len(valid_file_list)):
            os.system(f"cp {valid_file_list[i]} {save_path}")
            os.rename(os.path.join(save_path, valid_file_list[i].split('/')[-1]), os.path.join(save_path, f"original_part_{valid_file_list[i].split('/')[-1]}"))
            data["used meshes"].append(f"original_part_{valid_file_list[i].split('/')[-1]}")

        for group in groups:
            pts = []
            mask = []
            for i in group:
                pts.append(points_list[i])
                mask.append(junction_mask[i])
            pts = np.concatenate(pts, axis=0)
            mask = np.concatenate(mask, axis=0)
            np.savetxt(os.path.join(save_path, f"points_group{''.join([str(i) for i in group])}.txt"), pts)
            np.savetxt(os.path.join(save_path, f"points_group{''.join([str(i) for i in group])}_mask.txt"), mask)
        # for i in range(len(file_list)):
        #     if nodes_mask[i] == 0:
        #         continue
        #     else:
        #         np.savetxt(os.path.join(save_path, f"points_{i}.txt"), points_list[i])
        #         np.savetxt(os.path.join(save_path, f"points_{i}_mask.txt"), junction_mask[i])

        with open(os.path.join(save_path, 'part_selection.json'), "w") as f:
            json.dump(data, f, indent=4)
        saved_data_path.append(save_path)
        num_generated_parts_selection += 1
    # print(f"generated {num_generated_parts_selection} valid parts selection")
    return saved_data_path
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_dir', type=str, default= '/home/crtie/crtie/Furniture-Assembly/dataset/parts/Chair/poang_2', help='Path to the mesh data dir')
    parser.add_argument('--save_data_dir', type=str,default='systhesis_data/test_data', help='Path to save the generated data')
    parser.add_argument('--num_parts_selection', type=int, default=3, help='Number of generated parts selection')
    parser.add_argument('--visualize',type=bool, default=False , help='Visualize the parts selection')
    args = parser.parse_args()
    saved_data_path = main(args)
