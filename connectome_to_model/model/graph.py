from connectome_to_model.model.topdown_gru import ConvGRUBasalTopDownCell, ILC_upsampler
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import math

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
    
    def rank_node(self, current_node, output_node_index, rank_val, num_pass):
        ''' ranks each node in the graph'''
        current_node.rank_list.append(rank_val)
        rank_val = rank_val + 1
        for node in current_node.out_nodes_indices:
            num_pass = num_pass + 1
            if (current_node == output_node_index and num_pass == self.num_edge):
                self.max_rank = current_node.rank.max()
                return
            else:
                self.rank_node(self.nodes[node], output_node_index, rank_val, num_pass)

    def find_feedforward_cells(self, node):
        return self.nodes[node].in_nodes_indices

    def find_feedback_cells(self, node, t): 
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