from model.topdown_gru import ConvGRUTopDownCell, ILC_upsampler
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
    :param in_nodes_indices (list of int)
        nodes feeding forward into current node (bottom-up inputs)
    :param out_nodes_indices (list of int)
        nodes current node sends value and receives feedback from (topdown inputs)
        
    :param input_size (tuple of int)
        (height, width) of input to node
    :param input_dim (int)
        channel dim of input to node
    :param hidden_dim (int)
        hidden dim of node
    :param kernel_size (tuple of int)
        kernel/receptive field size of node
        
    '''
    def __init__(self, index, input_nodes, output_nodes, input_size = (28, 28),
                 input_dim = 1, hidden_dim = 10, kernel_size = (3,3)):
        
        # connectivity params
        self.index = index
        self.in_nodes_indices = input_nodes
        self.out_nodes_indices = output_nodes
        #self.in_strength = [] #connection strengths of in_nodes
        #self.out_strength = [] #connection strength of out_nodes

        # area params
        self.input_size = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
             
    def __eq__(self, other): 
        return self.index == other.index

class Graph(object):
    """ 
    Makes connectome graph pbject from csv connectome data file 
    
    :param graph_loc (str)
        location of csv file connectome data file 
    :param input_nodes (list of int)
        nodes that receive direct input
    :param output_nodes (int)
        node to get readout from
    """
    def __init__(self, graph_loc, input_nodes, output_node, directed=True, bias=False, dtype = torch.cuda.FloatTensor):
        
        graph_df = pd.read_csv(graph_loc)
        self.num_node = graph_df.shape[0]
        self.conn_strength = torch.tensor(graph_df.iloc[:, :self.num_node].values)   # connection_strength_matrix
        self.conn = self.conn_strength > 0                                           # binary matrix of connections        
        self.node_dims = [row for _, row in graph_df.iterrows()]                     # dimensions of each node/area
        self.nodes = self.generate_node_list(self.conn, self.node_dims)              # turn dataframe into list of graph nodes
        #self.longest_path_length = self.find_longest_path()
        
        self.input_node_indices = input_nodes
        self.output_node_index = output_node

        self.directed = directed #flag for whether the connections are directed 

        self.dtype = dtype
        self.bias = bias

    def generate_node_list(self,connections, node_dims):
        nodes = []
        # initialize node list
        for n in range(self.num_node):
            input_nodes = torch.nonzero(connections[:, n])
            output_nodes = torch.nonzero(connections[n, :])
            hidden_dim = node_dims[n].hidden_dim
            input_dim = node_dims[n].input_dim
            input_size = (node_dims[n].input_h, node_dims[n].input_w)
            #output_size = (node_dims[n].output_h, node_dims[n].output_w)
            kernel_size = (node_dims[n].kernel_h, node_dims[n].kernel_w)
           
            node = Node(n, input_nodes, output_nodes, 
                        input_dim=input_dim, 
                        input_size=input_size,
                        #output_size=input_size, 
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
    
    def find_longest_path_length(self):
        visited = []
        for node in self.nodes:
            if node.index not in self.visited:
                self.dfs(node, 0)

        return self.longest_path_length

    def dfs(self, node, current_length): # depth-first search (DFS) traversal of the graph
        self.visited.add(node.index)
        current_length += 1

        for out_node in node.output_nodes:
            if out_node.index not in self.visited:
                self.dfs(out_node, current_length)

        if current_length > self.longest_path_length:
            self.longest_path_length = current_length

        self.visited.remove(node.index)

class Architecture(nn.Module):
    def __init__(self, graph, input_sizes, input_dims, 
                 output_size=10,
                 topdown=True, topdown_type = 'multiplicative',
                 stereo=False,
                 dropout=True, dropout_p=0.25, rep=1,
                 proj_hidden_dim=32, device='cuda'):
        '''
        :param graph (Graph object)
        :param input_sizes (list of tuples)
            list of input sizes (height, width) per node. If a node doesn't receive an outside input, it's (0, 0)
        :param output_size (int)
            size of model output
        :param topdown (bool)
        :param topdown_type (str)
            'multiplicative' or 'composite', how to combine top-down info within the block
        :param stereo (bool or list of bools):
            if raw inputs to one area are stereo images. NOTE: stereo images are assumed to be equal in size 
        :param dropout (bool)
        :param dropout_p (float)
            dropout probability if dropout is enabled
        :param rep (int)
            repetitions, i.e how many times to view the sequence in full
        :param proj_hidden_dim (int)
            hyperparam. intermediate dimension of projection convolutions
        '''
        super(Architecture, self).__init__()
        
        assert len(input_sizes) == graph.num_node, "Must give sizes of direct stimuli for all nodes. If a node receive direct input, use (0,0)."
        assert len(input_dims) == graph.num_node, "Must give channels of direct stimuli for all nodes. If a node receive direct input, use 0."

        self.graph = graph
        self.rep = rep
        self.use_dropout = dropout
        self.dropout = nn.Dropout(dropout_p)
        self.topdown = topdown
        self.stereo = stereo
        self.proj_hidden_dim = proj_hidden_dim
        
        self.bottomup_projections = []
        self.topdown_projections = []
        
        # The "brain areas" of the model
        cell_list = []
        for node in range(graph.num_node):
            cell_list.append(ConvGRUTopDownCell(input_size=((graph.nodes[node].input_size[0], graph.nodes[node].input_size[1])), 
                                         input_dim=graph.nodes[node].input_dim, 
                                         hidden_dim = int(graph.nodes[node].hidden_dim),
                                         kernel_size=graph.nodes[node].kernel_size, 
                                         topdown_type = topdown_type,
                                         bias=graph.bias,
                                         dtype=graph.dtype))
        self.cell_list = nn.ModuleList(cell_list)
        
        # Readout layer
        h, w = self.graph.nodes[self.graph.output_node_index].input_size
        self.fc1 = nn.Linear(h * w * self.graph.nodes[self.graph.output_node_index].hidden_dim, 100) 
        self.fc2 = nn.Linear(100, output_size)
        
        
        # PROJECTIONS FOR BOTTOM-UP INTERLAYER CONNECTIONS
        for end_node in range(graph.num_node):
            per_input_projections = []
            num_inputs = len(self.graph.nodes[end_node].in_nodes_indices)
            
            # if node receives raw input, make a projection layer for that too
            if end_node in self.graph.input_node_indices:
                # calculate convolution dimensions
                stride, padding = self.calc_stride_padding(input_sizes[end_node], 
                                                      graph.nodes[end_node].input_size, 
                                                      graph.nodes[end_node].kernel_size)
                
                if self.stereo:
                    num_inputs += 2
                    proj = []
                    for i in range(2):
                        proj.append(nn.Conv2d(in_channels=input_dims[end_node], 
                                 out_channels=self.proj_hidden_dim,
                                kernel_size=graph.nodes[end_node].kernel_size,
                                stride=stride,
                                padding=padding,
                                device=device))
                    per_input_projections.append(nn.ModuleList(proj))
                else:
                    num_inputs += 1
                    proj = nn.Conv2d(in_channels=input_dims[end_node], 
                                 out_channels=self.proj_hidden_dim,
                                kernel_size=graph.nodes[end_node].kernel_size,
                                stride=stride,
                                padding=padding,
                                device=device)
                    per_input_projections.append(proj)
            
            # dealing with the layer-to-layer projections
            for start_node in self.graph.nodes[end_node].in_nodes_indices:
                
                # calculate convolution dimensions
                stride, padding = self.calc_stride_padding(graph.nodes[start_node].input_size, 
                                                      graph.nodes[end_node].input_size, 
                                                      graph.nodes[end_node].kernel_size)
                
                # projection from single node
                proj = nn.Conv2d(in_channels=graph.nodes[start_node].hidden_dim, 
                                 out_channels=self.proj_hidden_dim,
                                kernel_size=graph.nodes[end_node].kernel_size,
                                stride=stride,
                                padding=padding,
                                device=device)
                per_input_projections.append(proj)
            
            # Final conv to integrate all inputs. This convolution does not change shape of image
            integrator_conv = nn.Conv2d(in_channels=self.proj_hidden_dim*num_inputs, 
                                        out_channels=graph.nodes[end_node].input_dim,
                                       kernel_size=3, padding=1, device=device)
            per_input_projections.append(integrator_conv)
            
            self.bottomup_projections.append(nn.ModuleList(per_input_projections))
        print(self.bottomup_projections)
                
        # PROJECTIONS FOR TOPDOWN INTERLAYER CONNECTIONS        
        for end_node in range(graph.num_node):
            # number of nodes it projects to (i.e everyone it receives feedback from)
            num_inputs = len(self.graph.nodes[end_node].out_nodes_indices)
            per_input_projections = []
            
            for start_node in self.graph.nodes[end_node].out_nodes_indices:
                bottom_size = graph.nodes[end_node].input_size
                top_size = graph.nodes[start_node].input_size
            
                # upsample
                if top_size[0]*top_size[1] < bottom_size[0]*bottom_size[1]:
                    stride = [o//i for o, i in zip(bottom_size, top_size)]
                    #proj = ILC_upsampler(in_channel=graph.nodes[start_node].hidden_dim,
                    #                     out_channel=self.proj_hidden_dim,
                    #                    stride=stride, device=device)
                    proj = nn.ConvTranspose2d(in_channels=graph.nodes[start_node].hidden_dim,
                                         out_channels=self.proj_hidden_dim,
                                              kernel_size=stride, stride=stride, device=device)
                else: # or downsample
                    stride, padding = self.calc_stride_padding(top_size, bottom_size, 
                                                      graph.nodes[start_node].kernel_size)
                    proj = nn.Conv2d(in_channels=graph.nodes[start_node].hidden_dim, 
                                     out_channels=self.proj_hidden_dim,
                                    kernel_size=graph.nodes[end_node].kernel_size,
                                    stride=stride, padding=padding, device=device)
                per_input_projections.append(proj)
            
            # Final conv to integrate all inputs. This convolution does not change shape of image
            integrator_conv = nn.Conv2d(in_channels=self.proj_hidden_dim*num_inputs, 
                                        out_channels=graph.nodes[end_node].input_dim + graph.nodes[end_node].hidden_dim,
                                       kernel_size=3, padding=1, device=device)
            per_input_projections.append(integrator_conv)
            
            self.topdown_projections.append(nn.ModuleList(per_input_projections))
            #self.topdown_projections = nn.ModuleList(self.topdown_projections)
            
        print(self.topdown_projections)

    def forward(self, all_inputs, batch=True):
        """
        :param a list of tensor of size n, each consisting a input_tensor: (b, t, c, h, w). 
            n as the number of input signals. Order should correspond to self.graph.input_node_indices
        :param topdown: size (b,hidden,h,w) 
        :return: label 
        """
        #find the time length of the longest input signal + enough extra time for the last input to go through all nodes
        seq_len = max([i[0].shape[1] if isinstance(i, list) else i.shape[1] for i in all_inputs])
        process_time = seq_len + self.graph.num_node - 2
        
        if batch==True:
            batch_size = all_inputs[0][0].shape[0] if self.stereo else all_inputs[0].shape[0]
        else:
            batch_size = 1
            all_inputs = [torch.unsqueeze(inp, 0) for inp in all_inputs]
            
        # init hidden states
        hidden_states = self._init_hidden(batch_size=batch_size)
        hidden_states_prev = self._init_hidden(batch_size=batch_size)
        active_nodes = self.graph.input_node_indices.copy()
        #time = seq_len * self.rep
        
        # Each time you look at the sequence
        for rep in range(self.rep):
            
            # For each image
            for t in range(process_time):

                # Go through each node and see if anything should be processed
                for node in active_nodes:
                    #print('processing')

                    ########################################
                    # Find bottomup inputs
                    bottomup = []
                    input_num = 0                             # index of bottomup-input, used to fetch projection convs
                    projs = self.bottomup_projections[node]   # relevant bottom-up projections for this node
                    #print(node)
                    #print(self.graph.input_node_indices)

                    # direct stimuli if node receives it + the sequence isn't done
                    if node in self.graph.input_node_indices and t < seq_len:
                        inp = all_inputs[self.graph.input_node_indices.index(node)]
                        if self.stereo:
                            bottomup.append(projs[input_num][0](inp[0][:, t, :, :, :]))
                            bottomup.append(projs[input_num][1](inp[1][:, t, :, :, :]))
                        else:
                            bottomup.append(projs[input_num](inp[:, t, :, :, :]))
                        input_num += 1
                        
                    # bottom-up input from other nodes
                    for i, bottomup_node in enumerate(self.graph.nodes[node].in_nodes_indices):
                        bottomup.append(projs[input_num](hidden_states_prev[bottomup_node]))
                        input_num += 1
                        
                    if not bottomup:  #only triggers after sequence has ended
                        continue
                    
                    # Concatenate all inputs and integrate
                    bottomup = torch.cat(bottomup, dim=1)
                    bottomup = projs[-1](bottomup)
                    
                    ##################################
                    # Find topdown inputs, assumes every feedforward connection out of node has feedback
                    # Note there's no external topdown input
                    topdown_projs = self.topdown_projections[node]
                    
                    if self.topdown and (rep != 0 or t!=0) and self.graph.nodes[node].out_nodes_indices.nelement()!=0: 
                        topdown = []

                        for i, topdown_node in enumerate(self.graph.nodes[node].out_nodes_indices):
                            topdown.append(topdown_projs[i](hidden_states_prev[topdown_node]))
                            print(node)
                            print(hidden_states_prev[topdown_node].shape)
                            print(topdown_node)
                            print(topdown[i].shape)
                        
                        topdown = topdown_projs[-1](torch.cat(topdown, dim=1))
                        #print(topdown.shape)
                        #print(topdown)
                            
                    else:  
                        topdown = None # if this is the beginning of sequence, there's no topdown info
                    
                    #################################################
                    # Finally, pass to layer
                    h = self.cell_list[node](bottomup, hidden_states[node], topdown)
                    
                    # Dropout
                    if self.use_dropout:
                        h = self.dropout(h)
                    hidden_states[node] = h
                    
                # flag the areas to process in next iteration
                old_nodes = active_nodes
                active_nodes = []
                for prev_node in old_nodes:
                    for next_node in self.graph.nodes[prev_node].out_nodes_indices:
                        active_nodes.append(next_node.item())
                        
                # flag direct input areas too if the sequence is not fully seen yet        
                for inp_idx, seq in enumerate(all_inputs):
                    seq_shape = seq.shape[0][1] if self.stereo else seq.shape[1]
                    if t + 1 < seq_shape:
                        active_nodes.append(self.graph.input_node_indices[inp_idx])
                
                # copy hidden state
                hidden_states_prev = hidden_states

        pred = self.fc1(F.relu(torch.flatten(hidden_states[self.graph.output_node_index], start_dim=1)))
        pred = self.fc2(F.relu(pred))
          
        return pred

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.graph.num_node):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
    
    def calc_stride_padding(self, input_sz, output_sz, kernel_sz):
        
        stride = [math.ceil((i - k) / (o - 1)) for i, o, k in zip(input_sz, output_sz, kernel_sz)]
        padding = [math.ceil(((o - 1)*s + k - i)/2) for i, o, k, s in zip(input_sz, output_sz, kernel_sz, stride)]
        #print(stride)
        #print(padding)
        
        return stride, padding
    
    def calc_padding_transpose(self, input_sz, output_sz, kernel_sz):
        
        padding = [math.ceil((o-i+k-1)/2) for i, o, k in zip(input_sz, output_sz, kernel_sz)]
        #print(padding)
        
        return padding
        