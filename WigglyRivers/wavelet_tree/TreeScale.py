# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez-Duque
#                           Last revised 2023-07-13
# _____________________________________________________________________________
# _____________________________________________________________________________
"""

Creation of tree scales class
"""

# ------------------------
# Importing Modules
# ------------------------
import copy
import logging
import numpy as np
import pandas as pd
from anytree import Node, PreOrderIter
from anytree.exporter import DotExporter


# Package packages
from ..utilities import utilities as utl
from ..utilities import filesManagement as FM


# ---------------------
# Logging
# ---------------------
logging.basicConfig(handlers=[logging.NullHandler()])


# ------------------------
# Functions
# ------------------------
# ------------------------
# Classes
# ------------------------
class RiverTreeScales:
    """
    This class is the basic form of rivers.

    ===================== =====================================================
    Attribute             Description
    ===================== =====================================================
    uid                   Unique identifier
    trees                 Dictionary of trees
    tree_ids              List of tree ids
    metrics               List of metrics
    metrics_in_name       List of metrics in name
    ===================== =====================================================

    The following are the methods of the class.

    ====================== =====================================================
    Methods                Description
    ====================== =====================================================
    scale_coordinates      Scale coordinates by a given value
    ====================== =====================================================
    """

    def __init__(self, trees=None, uid=0, logger=None):
        """
        Initialization of the class.
        """
        # ---------------------
        # Logging
        # ---------------------
        if logger is None:
            self._logging = logging.getLogger(self.__class__.__name__)
        else:
            self._logging = logger
            self._logging.info(f"Start Logger in {self.__class__.__name__}")

        # ---------------------
        # Attributes
        # ---------------------
        self.uid = uid
        self.database = {}
        if trees is not None:
            self.load_trees_dict(trees)
            self.tree_ids = list(self.trees)
            self.update_database()
        else:
            self.trees = {}
            self.tree_ids = []
        self.node_ids = {}
        self.metrics = [
            "peak_col",
            "peak_row",
            "peak_pwr",
            "s_c",
            "scales_c",
            "wavelength_c",
            "x_c",
            "y_c",
            "r_x",
            "r_y",
            "x_1",
            "x_2",
            "x_3",
            "y_1",
            "y_2",
            "y_3",
            "cc_x",
            "cc_y",
            "beg_idx",
            "end_idx",
            "mid_idx",
            "radius",
            "idx_planimetry_start",
            "idx_planimetry_end",
            "sn",
            "l",
            "lambda_value",
            "fl",
            "sk",
        ]
        self.metrics_in_name = ["radius", "sn", "lambda_value", "peak_pwr"]
        self.metrics_in_name_formatting = [
            "{:.2f}",
            "{:.2f}",
            "{:.2f}",
            "{:.2E}",
        ]

    # ---------------------
    # Core Methods
    # ---------------------
    def __getitem__(self, tree_id):
        """
        This function is used to get the values of the class.
        """
        if tree_id not in self.tree_ids:
            self._logging.error(f"Tree ID {tree_id} not in the tree list")
            raise KeyError(f"Tree ID {tree_id} not in the tree list")
        return self.trees[tree_id]

    # ---------------------
    # Methods
    # ---------------------
    def build_trees_from_tree_scales_dict(self, tree_scales):
        """
        Build trees from tree scales dictionary
        """
        branch_ids = np.array(copy.deepcopy(tree_scales["branch_id"]))
        tree_id = 0
        meander_count = 0

        # Extract all possible branches
        while len(branch_ids) > 0:
            branch_id = branch_ids[0]

            linking_branch = tree_scales["link_branch_by_level"][branch_id][0]
            root_node = self._build_tree_from_linking_branches(
                tree_scales,
                linking_branch,
                tree_id=tree_id,
                meander_count=meander_count,
            )

            self.trees[tree_id] = root_node
            self.tree_ids.append(tree_id)
            idx = np.in1d(branch_ids, linking_branch)
            branch_ids = np.delete(branch_ids, idx, axis=0)
            tree_id += 1

        self.update_database()
        return

    def _build_tree_from_linking_branches(
        self, tree_scales, linking_branch, tree_id=0, meander_count=0
    ):
        """
        Create tree from linking branches in the tree_scale dictionary

        This function creates a complete tree from the linking branches
        """
        node_id = 0
        nodes = []
        node_ids = []
        indicate_var = []
        # linking_branch = tree_scales['link_branch_by_level'][branch_id][0]
        for i_b, b in enumerate(linking_branch):
            # Create tree nodes
            levels_root_leaf = tree_scales["levels_root_leaf"][b]
            for i_l, level in enumerate(levels_root_leaf):
                # Extract Information
                radius = tree_scales["idx_conn"][b][level]
                # Do not go through nodes already taken
                if radius in indicate_var:
                    parent_radius = copy.deepcopy(radius)
                    continue
                indicate_var.append(radius)

                # Extract Information
                meander_in_level = tree_scales["meander_in_level_root_leaf"][b]
                metrics = self._extract_metrics(tree_scales, b, level)
                if meander_in_level == i_l:
                    meander_id = copy.deepcopy(meander_count)
                    meander_count += 1
                    is_meander = 1
                else:
                    meander_id = -1
                    is_meander = 0
                # Create Nodes
                if i_l == 0 and i_b == 0:  # Root Node
                    node = self._create_tree_node(
                        tree_id,
                        node_id,
                        metrics,
                        node_type="root",
                        is_meander=is_meander,
                        meander_id=meander_id,
                    )
                else:
                    idx = indicate_var.index(parent_radius)

                    parent = nodes[idx]
                    if i_l != len(levels_root_leaf) - 1:  # Parent Node
                        node = self._create_tree_node(
                            tree_id,
                            node_id,
                            metrics,
                            parent=parent,
                            node_type="parent",
                            is_meander=is_meander,
                            meander_id=meander_id,
                        )
                    else:  # Leaf node
                        node = self._create_tree_node(
                            tree_id,
                            node_id,
                            metrics,
                            parent=parent,
                            node_type="leaf",
                            is_meander=is_meander,
                            meander_id=meander_id,
                        )
                nodes.append(node)
                node_ids.append(node_id)
                # if tree_id == 1 and node_id in (0, 3):
                #     temp = 2
                node_id += 1
                parent_radius = copy.deepcopy(radius)

        root_node = nodes[0]
        self.node_ids[tree_id] = node_ids
        return root_node

    def _create_tree_node(
        self,
        tree_id,
        node_id,
        metrics,
        parent=None,
        node_type="root",
        is_meander=0,
        meander_id=-1,
    ):
        """
        Create tree node
        """
        # Select type
        if node_type == "root":
            root_node = 1
            parent_node = 0
            leaf_node = 0
        elif node_type == "parent":
            root_node = 0
            parent_node = 1
            leaf_node = 0
        elif node_type == "leaf":
            root_node = 0
            parent_node = 0
            leaf_node = 1

        # Create Node Name
        node_name = [
            f"{tree_id}-{node_id}\n",
            f"{node_type}\n",
            f"meander: {is_meander}\n",
            f"meander_id: {meander_id}\n",
        ]

        node_name += [
            f"{i}: {metrics[i]:.2f}\n"
            for i in self.metrics_in_name
            if i != "peak_pwr"
        ]
        node_name += [f'peak_pwr: {metrics["peak_pwr"]:.2E}']

        node_name = "".join(node_name)

        # Create Node
        node = Node(
            node_name,
            parent=parent,
            tree_id=tree_id,
            node_id=node_id,
            root_node=root_node,
            parent_node=parent_node,
            leaf_node=leaf_node,
            is_meander=is_meander,
            meander_id=meander_id,
            **metrics,
        )
        return node

    def _extract_metrics(self, tree_scales, branch_id, level):
        """
        Extract metrics from tree scales dictionary
        """
        metrics_values = {
            i: tree_scales[i][branch_id][level] for i in self.metrics
        }
        return metrics_values

    def select_node(self, tree_id, node_id):
        """
        Select node with node_id
        """
        root_node = self.trees[tree_id]
        selected_node = list(
            PreOrderIter(root_node, filter_=lambda n: n.node_id == node_id)
        )[0]

        return selected_node

    def filter_nodes(self, tree_ids=None, key=None, value=None):
        """
        Filter nodes by key and value
        """
        nodes = {}
        if tree_ids is None:
            tree_ids = self.tree_ids
        else:
            if isinstance(tree_ids, int):
                tree_ids = [tree_ids]

        for tree_id in tree_ids:
            nodes[tree_id] = list(
                PreOrderIter(
                    self.trees[tree_id],
                    filter_=lambda n: n.__dict__[key] == value,
                )
            )
        return nodes

    def add_parameter_to_nodes(self, tree_ids=None, key=None, value=None):
        """
        Add paramter to node
        """
        return

    def update_database(self):
        """
        Update database
        """
        self.database = {}
        for tree_id in self.tree_ids:
            root_node = self.trees[tree_id]
            nodes = [root_node] + list(self.trees[tree_id].descendants)
            data = {}
            data["from_node_id"] = [-1]
            data["from_node_id"] += [node.parent.node_id for node in nodes[1:]]
            keys = list(nodes[0].__dict__)
            keys.remove("name")
            keys.remove("_NodeMixin__children")
            data.update({j: [node.__dict__[j] for node in nodes] for j in keys})
            data.update({"depth": [node.depth for node in nodes]})
            self.database[tree_id] = pd.DataFrame(data)
        return

    def compile_database(self):
        """
        Compile current trees databases to a single table
        """
        for tree_id in self.tree_ids:
            if tree_id == 0:
                compiled_database = self.database[tree_id]
            else:
                compiled_database = pd.concat(
                    [compiled_database, self.database[tree_id]],
                    ignore_index=True,
                )

        return compiled_database

    def create_connectivity_matrix(self, tree_id):
        """
        Create connectivity matrixrows and columns denote node IDs
        """
        database = self.database[tree_id]
        parent_nodes = database["from_node_id"].values
        node_ids = database["node_id"].values
        # Create connectivity matrix
        connectivity_matrix = np.zeros((len(node_ids), len(node_ids)))
        for i, node_id in enumerate(node_ids):
            from_value = parent_nodes[i]
            if from_value != -1:
                connectivity_matrix[node_id, from_value] = 1
        return connectivity_matrix

    def save_tree(self, path_output="", file_name="tree.p"):
        """
        Save tree dictionary
        """
        FM.save_data(self.trees, path_output=path_output, file_name=file_name)
        return

    def load_trees_dict(self, trees_dict):
        """
        Load data from a trees dictionary
        """
        self.trees = trees_dict
        self.tree_ids = list(self.trees)
        self.update_database()
        return

    def load_tree(self, path_data):
        """
        Load tree dictionary
        """
        self.trees = FM.load_data(path_data)
        self.tree_ids = list(self.trees)
        return

    def save_tree_figure(self, tree_id, path_img="", file_name="tree.png"):
        """
        Save tree figure
        """
        utl.cr_folder(path_img)
        root_node = self.trees[tree_id]
        DotExporter(root_node).to_picture(f"{path_img}{file_name}")
        return

    def prune(self, method="sinuosity", *args, **kwargs):
        prune_methods = {
            "sinuosity": self._prune_by_sinuosity,
            "width": self._prune_by_width,
            "peak_power": self._prune_by_peak_power,
        }
        meanders = self.filter_nodes(key="is_meander", value=1)
        for tree_id in list(meanders):
            already_extracted = []
            for node in meanders[tree_id]:
                prune_methods[method](node, already_extracted, *args, **kwargs)

        return

    @staticmethod
    def _prune_by_sinuosity(
        node, already_extracted=[], sinuosity_var="sn", sinuosity_threshold=1.1
    ):
        """
        Prune tree by sinuosity with a give threshold.
        """
        if node.__dict__[sinuosity_var] < sinuosity_threshold:
            node.__dict__["is_meander"] = 0

        return

    @staticmethod
    def _prune_by_width(
        node,
        already_extracted=[],
        width_var="w_m",
        compare_var="wavelength_c",
        gamma=10,
        additional_threshold=1e6,
        additional_var="radius",
    ):
        """
        Change the meander level in nodes with compare_var lower than
        gamma * width_var
        """
        # Extract meanders
        if node.__dict__[compare_var] < node.__dict__[width_var] * gamma:
            node.__dict__["is_meander"] = 0
            # Convert parent to meander if additional threshold
            # is satisfied
            parent = node.parent
            if parent.__dict__[additional_var] < additional_threshold:
                parent.__dict__["is_meander"] = 1
        else:
            node.__dict__["is_meander"] = 1
        return

    @staticmethod
    def _prune_by_peak_power(
        node,
        already_extracted=[],
        peak_power_var="peak_pwr",
        sign_var="direction_node_to_parent",
        characteristic_length=1e6,
        compare_var="radius",
    ):
        """
        Change the meander level in parent nodes with peak_power_var higher
        than the mean of the children peak_power_var.
        """
        # Get parent node
        node_parent = node.parent
        if (node_parent.node_id in already_extracted) or (
            node_parent.is_meander == 1
        ):
            return
        # Look if parent has other parent nodes as children
        children_nodes = node_parent.children
        descendent_nodes = node_parent.descendants
        already_extracted.append(node_parent.node_id)
        if len(descendent_nodes) > len(children_nodes):
            return

        # Get peak power of parent
        peak_pwr_parent = node_parent.__dict__[peak_power_var]
        peak_pwr_children = [
            child.__dict__[peak_power_var] for child in children_nodes
        ]
        # Extract if node is pointing towards parent (1) or not (-1)
        sign_children = [child.__dict__[sign_var] for child in children_nodes]

        # Calculate the peak power mean of the children
        # that point towards the parent (sign_children == 1)
        mean_peak_pwr_children = np.mean(
            np.array(peak_pwr_children)[np.array(sign_children) == 1]
        )
        if peak_pwr_parent > mean_peak_pwr_children:
            cond = node_parent.is_root
            if node_parent.__dict__[
                compare_var
            ] < characteristic_length and not (cond):
                node_parent.is_meander = 1
                for child in children_nodes:
                    if child.direction_node_to_parent == 1:
                        child.is_meander = 0
                    else:
                        if peak_pwr_parent > child.peak_pwr:
                            child.is_meander = 0
        return
