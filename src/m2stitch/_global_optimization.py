import networkx as nx
import numpy as np
import pandas as pd

from tqdm import tqdm
from ._typing_utils import Int, Float, NumArray
from ._translation_computation import extract_overlap_subregion, ncc
from copy import deepcopy

global ncc_dict
ncc_dict = {'left':{}, 'top':{}}

def compute_maximum_spanning_tree(grid: pd.DataFrame) -> nx.Graph:
    """Compute the maximum spanning tree for grid position determination.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position,
        with columns "{left|top}_{x|y|ncc|valid3}"

    Returns
    -------
    tree : nx.Graph
        the result spanning tree
    """
    connection_graph = nx.Graph()
    for i, g in grid.iterrows():
        for direction in ["left", "top"]:
            if not pd.isna(g[direction]):
                weight = g[f"{direction}_ncc"]
                #if g[f"{direction}_valid3"]:
                #    weight = weight + 10
                connection_graph.add_edge(
                    i,
                    g[direction],
                    weight=weight,
                    direction=direction,
                    f=i,
                    t=g[direction],
                    y=g[f"{direction}_y"],
                    x=g[f"{direction}_x"],
                )
    return nx.maximum_spanning_tree(connection_graph)


def full_tree_weight(grid: pd.DataFrame, images: NumArray, add: Bool = True) -> Float:
    def overlap_ncc(params, im1, im2):
        y, x = params
        subI1 = extract_overlap_subregion(im1, y, x)
        subI2 = extract_overlap_subregion(im2, -y, -x)
        if subI1.shape[0] == 0 or subI2.shape[0] == 0:
            return np.nan
        return ncc(subI1, subI2)

    if add:
        total_ncc = 0
    else:
        total_ncc = 1

    for direction in ["left", "top"]:
        for i2, g2 in grid.iterrows():
            i1 = g2[direction]
            if pd.isna(i1):
                continue

            if i2 not in ncc_dict[direction]:
                ncc_dict[direction][i2] = {}

            g1 = grid.iloc[i1]
            image1 = images[i1]
            image2 = images[i2]
            sizeY, sizeX = image1.shape

            y_overlap = g2["y_pos"] - g1["y_pos"]
            x_overlap = g2["x_pos"] - g1["x_pos"]

            if (y_overlap, x_overlap) not in ncc_dict[direction][i2]:
                ncc_dict[direction][i2][(y_overlap, x_overlap)] = \
                    overlap_ncc([y_overlap, x_overlap], image1, image2)

            if add:
                total_ncc += ncc_dict[direction][i2][(y_overlap, x_overlap)]
            else:
                total_ncc *= ncc_dict[direction][i2][(y_overlap, x_overlap)]
    return total_ncc


def alternative_max_spanning_tree(grid: pd.DataFrame, images: NumArray, max_iterations: Int = 100000) -> nx.Graph:
    connection_graph = nx.Graph()
    for i, g in grid.iterrows():
        for direction in ["left", "top"]:
            if not pd.isna(g[direction]):
                weight = g[f"{direction}_ncc"]
                #if g[f"{direction}_valid3"]:
                #    weight = weight + 10
                connection_graph.add_edge(
                    i,
                    g[direction],
                    weight=weight,
                    direction=direction,
                    f=i,
                    t=g[direction],
                    y=g[f"{direction}_y"],
                    x=g[f"{direction}_x"],
                )
    max_ncc = 0
    max_tree = None
    max_it = None
    all_trees = nx.SpanningTreeIterator(connection_graph, weight='weight', minimum=False)

    nccs = []
    for it, s in tqdm(enumerate(all_trees)):
        # iterate over all spanning trees, from maximum default weight to minimum
        new_grid = compute_final_position(grid, s)
        ncc = full_tree_weight(new_grid, images)
        nccs.append(ncc)
        if ncc > max_ncc:
            max_ncc = deepcopy(ncc)
            max_tree = deepcopy(s)
            max_it = deepcopy(it)
        if it > max_iterations:
            break

    return max_tree, nccs


def compute_final_position(
    grid: pd.DataFrame, tree: nx.Graph, source_index: Int = 0
) -> pd.DataFrame:
    """Compute the final tile positions by the computed maximum spanning tree.

    Parameters
    ----------
    grid : pd.DataFrame
        the dataframe for the grid position
    tree : nx.Graph
        the maximum spanning tree
    source_index : Int, optional
        the source position of the spanning tree, by default 0

    Returns
    -------
    grid : pd.DataFrame
        the result dataframe for the grid position, with columns "{x|y}_pos"
    """
    grid.loc[source_index, "y_pos"] = 0
    grid.loc[source_index, "x_pos"] = 0

    nodes = [source_index]
    walked_nodes = []
    while len(nodes) > 0:
        node = nodes.pop()
        walked_nodes.append(node)
        for adj, props in tree.adj[node].items():
            if not adj in walked_nodes:
                assert (props["f"] == node) & (props["t"] == adj) or (
                    props["t"] == node
                ) & (props["f"] == adj)
                nodes.append(adj)
                y_pos = grid.loc[node, "y_pos"]
                x_pos = grid.loc[node, "x_pos"]

                if node == props["t"]:
                    grid.loc[adj, "y_pos"] = y_pos + props["y"]
                    grid.loc[adj, "x_pos"] = x_pos + props["x"]
                else:
                    grid.loc[adj, "y_pos"] = y_pos - props["y"]
                    grid.loc[adj, "x_pos"] = x_pos - props["x"]
    for dim in "yx":
        k = f"{dim}_pos"
        assert not any(pd.isna(grid[k]))
        grid[k] = grid[k] - grid[k].min()
        grid[k] = grid[k].astype(np.int32)

    return grid
