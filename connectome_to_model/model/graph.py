import torch
import pandas as pd

# Backwards-compatible alias: the model class was renamed to ConnectomicsConvGRU and
# moved to architectures.py. Scripts and notebooks still import `Architecture` from here.
from connectome_to_model.model.architectures import ConnectomicsConvGRU as Architecture

# Per-area parameter columns the CSV must contain (everything after the adjacency block).
REQUIRED_AREA_COLUMNS = [
    'hidden_dim', 'input_dim',
    'input_h', 'input_w',
    'kernel_h', 'kernel_w',
    'basal_topdown_dim', 'apical_topdown_dim',
]
# Columns that must be strictly positive (a zero-sized area/kernel makes no sense).
_POSITIVE_COLUMNS = ['hidden_dim', 'input_dim', 'input_h', 'input_w', 'kernel_h', 'kernel_w']
# Columns that must be non-negative (0 means "no feedback of this type").
_NONNEGATIVE_COLUMNS = ['basal_topdown_dim', 'apical_topdown_dim']


def validate_connectome_csv(graph_df, input_nodes, output_nodes):
    """
    Check a loaded connectome dataframe and raise a readable ValueError on the first problem.

    The expected layout is: the first N columns (N = number of rows) form the square
    area-to-area adjacency matrix, followed by the per-area parameter columns listed in
    REQUIRED_AREA_COLUMNS. Returns the number of areas (rows) on success.
    """
    num_node = graph_df.shape[0]
    if num_node == 0:
        raise ValueError("Connectome CSV has no rows; expected one row per brain area.")

    # 1. Required parameter columns are present.
    missing = [c for c in REQUIRED_AREA_COLUMNS if c not in graph_df.columns]
    if missing:
        raise ValueError(
            f"Connectome CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected columns after the {num_node}-column adjacency block: "
            f"{', '.join(REQUIRED_AREA_COLUMNS)}."
        )

    # 2. There must be at least num_node adjacency columns plus the parameter columns.
    if graph_df.shape[1] < num_node + len(REQUIRED_AREA_COLUMNS):
        raise ValueError(
            f"Connectome CSV has {graph_df.shape[1]} columns but needs at least "
            f"{num_node + len(REQUIRED_AREA_COLUMNS)} "
            f"({num_node} adjacency columns for {num_node} areas + "
            f"{len(REQUIRED_AREA_COLUMNS)} parameter columns)."
        )

    # 3. The adjacency block (first num_node columns) must not overlap the parameter columns.
    adjacency_cols = list(graph_df.columns[:num_node])
    overlap = [c for c in REQUIRED_AREA_COLUMNS if c in adjacency_cols]
    if overlap:
        raise ValueError(
            f"Parameter column(s) {', '.join(overlap)} fall inside the first {num_node} "
            f"(adjacency) columns. The adjacency block must be exactly the first {num_node} "
            f"columns, one per area."
        )

    # 4. Adjacency must be numeric.
    adjacency = graph_df.iloc[:, :num_node]
    if not adjacency.apply(lambda col: pd.api.types.is_numeric_dtype(col)).all():
        raise ValueError(
            "The adjacency block (first columns) contains non-numeric values; "
            "connection strengths must be numbers (e.g. 0/1)."
        )

    # 5. Per-area value sanity, reported with the offending row.
    for col in _POSITIVE_COLUMNS:
        bad = graph_df.index[graph_df[col] <= 0].tolist()
        if bad:
            raise ValueError(f"Column '{col}' must be > 0, but row(s) {bad} have a value <= 0.")
    for col in _NONNEGATIVE_COLUMNS:
        bad = graph_df.index[graph_df[col] < 0].tolist()
        if bad:
            raise ValueError(f"Column '{col}' must be >= 0, but row(s) {bad} have a negative value.")

    # 6. input/output node indices are in range.
    out_list = output_nodes if isinstance(output_nodes, list) else [output_nodes]
    for label, idxs in (("input_nodes", input_nodes), ("output_nodes", out_list)):
        for i in idxs:
            if not (0 <= i < num_node):
                raise ValueError(
                    f"{label} contains index {i}, which is out of range for a graph with "
                    f"{num_node} areas (valid indices 0..{num_node - 1})."
                )

    return num_node

class Node:
    '''
    Represents a single area. Currently assumes 2D input
    
    :param index (int)
        Node ID as it is in connectome file
    :param input_nodes (list of int)
        nodes feeding forward into current node (bottom-up inputs)
    :param out_nodes_indices (list of int)
        nodes current node sends voutput to and receives feedback from (topdown inputs)
        
    :param input_size (tuple of int)
        (height, width) of input to node
    :param input_dim (int)
        channel dim of input to node
        
    :param hidden_dim (int)
        hidden dim of node (corresponds to relative area size)
    :param basal_topdown_dim (int)
        amount of basal feedback received (see S1 in README)
    :param apical_topdown_dim (int)
        amount of apical feedback received (see S1 in README)
    :param kernel_size (tuple of int)
        kernel/receptive field size of node
        
    '''
    def __init__(self, index, input_nodes, output_nodes, fb_nodes, 
                 input_size = (28, 28),
                 input_dim = 1, hidden_dim = 10, 
                 basal_topdown_dim=0, apical_topdown_dim=0,
                 kernel_size = (3,3)):
        
        # connectivity params
        self.index = index
        self.in_nodes_indices = input_nodes
        self.out_nodes_indices = output_nodes
        self.fb_nodes = fb_nodes
        #self.in_strength = [] #connection strengths of in_nodes
        #self.out_strength = [] #connection strength of out_nodes

        # area params
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.basal_topdown_dim = basal_topdown_dim
        self.apical_topdown_dim = apical_topdown_dim
        self.kernel_size = kernel_size
             
    def __eq__(self, other): 
        return self.index == other.index

class Graph(object):
    """ 
    Makes connectome graph pbject from csv connectome data file 
    
    :param graph_loc (str)
        location of csv file connectome data file 
    :param input_nodes (list of int)
        nodes that receive direct stimuli
    :param output_nodes (list of int)
        node to get readout from
    :param bias (bool):
        whether to use bias in convolutions
    :param reciprocal (bool):
        a reciprocal graph only has 1, 0 and assumes all FF connections have equivalent FB
    :param dtype (torch.dtype, optional):
        data type for tensors. If None, automatically chooses based on CUDA availability
    """
    def __init__(self, graph_loc, input_nodes, output_nodes,
                 bias=False, reciprocal=True, dtype=None):
        
        graph_df = pd.read_csv(graph_loc)
        validate_connectome_csv(graph_df, input_nodes, output_nodes)
        self.reciprocal = reciprocal
        self.num_node = graph_df.shape[0]
        self.conn = torch.tensor(graph_df.iloc[:, :self.num_node].values)   # connection_strength_matrix
        #self.conn = self.conn_strength > 0                                           # binary matrix of connections        
        self.node_dims = [row for _, row in graph_df.iterrows()]                     # dimensions of each node/area
        self.nodes = self.generate_node_list(self.conn, self.node_dims)              # turn dataframe into list of graph nodes

        self.input_indices = input_nodes
        self.output_indices = output_nodes if isinstance(output_nodes, list) else [output_nodes] 

        # Set dtype dynamically based on device availability
        if dtype is None:
            self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        else:
            self.dtype = dtype
        self.bias = bias

    def generate_node_list(self,connections, node_dims):
        nodes = []
        # initialize node list
        for n in range(self.num_node):
            input_nodes = torch.nonzero(connections[:, n] > 0)
            output_nodes = torch.nonzero(connections[n, :] > 0)
            fb_nodes = output_nodes if self.reciprocal else torch.nonzero(connections[:, n] < 0)
            
            hidden_dim = node_dims[n].hidden_dim
            input_dim = node_dims[n].input_dim
            basal_topdown_dim=node_dims[n].basal_topdown_dim
            apical_topdown_dim=node_dims[n].apical_topdown_dim
            input_size = (node_dims[n].input_h, node_dims[n].input_w)
            kernel_size = (node_dims[n].kernel_h, node_dims[n].kernel_w)
           
            node = Node(n, input_nodes, output_nodes, fb_nodes,
                        input_dim=input_dim, 
                        input_size=input_size,
                        basal_topdown_dim=basal_topdown_dim, 
                        apical_topdown_dim=apical_topdown_dim, 
                        hidden_dim=hidden_dim,
                       kernel_size=kernel_size)
            nodes.append(node)
            
        return nodes
    
    def find_feedforward_cells(self, node):
        return self.nodes[node].in_nodes_indices

    def find_feedback_cells(self, node):
        return self.nodes[node].out_nodes_indices

    def find_input_sizes(self):
        input_szs = []
        for node in self.nodes:
            if node.index in self.input_indices:
                input_szs.append(node.input_size)
            else:
                input_szs.append((0,0))
        return input_szs
    
    def find_input_dims(self):
        input_dims = []
        for node in self.nodes:
            if node.index in self.input_indices:
                input_dims.append(node.input_dim)
            else:
                input_dims.append(0)
        return input_dims