from model.topdown_gru import ConvGRUTopDownCell
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import math

#def conv_output_size(input_size, kernel_size, stride=1):
#    padding = kernel_size[0] // 2
#    return ((input_size - kernel_size) // stride) + 1

class Node:
    def __init__(self, index, input_nodes, output_nodes, input_size = (28, 28), output_size = (28, 28),
                 input_dim = 1, hidden_dim = 10, kernel_size = (3,3)):
        self.index = index
        self.in_nodes_indices = input_nodes #nodes passing values into current node #contains Node index (int)
        self.out_nodes_indices = output_nodes #nodes being passed with values from current node #contains Node index (int)
        self.in_strength = [] #connection strengths of in_nodes
        self.out_strength = [] #connection strength of out_nodes
        self.rank_list = [-1] #default value. if the Node end up with only -1 as its rank_list element, then 


        #cell params
        self.input_size = input_size #Height and width of input tensor as (height, width).
        self.output_size = output_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = (self.calc_stride(input_size[0], output_size[0], kernel_size[0]), 
                      self.calc_stride(input_size[1], output_size[1], kernel_size[1]))
        self.padding = (self.calc_padding(input_size[0], output_size[0], kernel_size[0], self.stride[0]), 
                      self.calc_padding(input_size[1], output_size[1], kernel_size[1], self.stride[1]))
             

    def __eq__(self, other): 
        return self.index == other.index
    
    def calc_stride(self, input_size, output_size, kernel_size):
        return (input_size - kernel_size) // (output_size - 1)

    def calc_padding(self, input_size, output_size, kernel_size, stride):
        return ((output_size - 1) * stride + kernel_size - input_size)
    


class Graph(object):
    """ 
    A brain architecture graph object, directed by default. 
    
    :param connections (n x n)
        directed binary matrix of connections row --> column 
    :param connection_strength
    """
    def __init__(self, graph_loc, input_nodes, output_node, directed=True, bias=False, dtype = torch.cuda.FloatTensor):
        
        graph_df = pd.read_csv(graph_loc)
        self.num_node = graph_df.shape[0]
        self.conn_strength = torch.tensor(graph_df.iloc[:, :self.num_node].values)   # connection_strength_matrix
        self.conn = self.conn_strength > 0                                           # binary matrix of connections
        self.node_dims = [row for _, row in graph_df.iterrows()]                     # dimensions of each node/area
        self.nodes = self.generate_node_list(self.conn, self.node_dims)              # turn dataframe into list of graph nodes
        
        self.input_node_indices = input_nodes
        self.output_node_index = output_node

        self.directed = directed #flag for whether the connections are directed 

        self.dtype = dtype
        self.bias = bias

        #for input_node in self.input_node_indices:
        #    self.rank_node(self.nodes[input_node],self.nodes[self.output_node_index], 0, 0)

        # for node in range (self.num_node):
            
        #     print(self.nodes[node].rank_list)



    def generate_node_list(self,connections, node_dims):
        nodes = []
        # initialize node list
        for n in range(self.num_node):
            input_nodes = torch.nonzero(connections[:, n])
            output_nodes = torch.nonzero(connections[n, :])
            hidden_dim = node_dims[n].hidden_dim
            input_dim = node_dims[n].input_dim
            input_size = (node_dims[n].input_h, node_dims[n].input_w)
            output_size = (node_dims[n].output_h, node_dims[n].output_w)
            kernel_size = (node_dims[n].kernel_h, node_dims[n].kernel_w)
           
            node = Node(n, input_nodes, output_nodes, 
                        input_dim=input_dim, 
                        input_size=input_size,
                        output_size=input_size, 
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

    def build_architecture(self):
        architecture = Architecture(self).cuda().float()
        return architecture

    def find_feedforward_cells(self, node):
        return self.nodes[node].in_nodes_indices

    def find_feedback_cells(self, node, t):
        return self.nodes[node].out_nodes_indices
        # if ((t+1) in self.nodes[n].rank_list):
        #     in_nodes.append(n)
        # return in_nodes

    def find_longest_path_length(self): #TODO: 
        return 42
        #I think this is not needed but we'll see

class Architecture(nn.Module):
    def __init__(self, graph, input_sizes, input_dims, topdown=True,
                 output_size=10, dropout=True, dropout_p=0.25, rep=1,
                 proj_hidden_dim=16, device='cuda'):
        '''
        :param graph (Grapph object)
            object containing architecture graph
        :param input_sizes (list of tuples)
            list of input sizes (height, width) per node. If a node doesn't receive an outside input, it's (0, 0)
        :param input_dims (list of ints)
            list of input dimension sizes per node. If a node doesn't receive an outside input, it's 0
        :param dropout (bool)
        :param rep (int)
            time steps to revisit the sequence
        '''
        super(Architecture, self).__init__()

        self.graph = graph
        self.rep = rep # repetition, i.e how many times to view the sequence in full
        self.use_dropout = dropout
        self.dropout = nn.Dropout(dropout_p)
        self.topdown = topdown
        
        self.bottomup_projections = []
        self.topdown_projections = []
        
        # intermediate dimension of projection convolutions
        self.proj_hidden_dim = proj_hidden_dim
        
        # The "brain areas" of the model
        cell_list = []
        for node in range(graph.num_node):
            cell_list.append(ConvGRUTopDownCell(input_size=((graph.nodes[node].input_size[0], graph.nodes[node].input_size[1])), 
                                         input_dim=graph.nodes[node].input_dim, 
                                         hidden_dim = int(graph.nodes[node].hidden_dim),
                                         kernel_size=graph.nodes[node].kernel_size,
                                         bias=graph.bias,
                                         dtype=graph.dtype))
        self.cell_list = nn.ModuleList(cell_list)
        
        # Dimensionality of output layer
        h, w = self.graph.nodes[self.graph.output_node_index].input_size
        self.fc1 = nn.Linear(h * w * self.graph.nodes[self.graph.output_node_index].hidden_dim, 100) 
        self.fc2 = nn.Linear(100, output_size)
        
        
        # PROJECTIONS FOR BOTTOM-UP INTERLAYER CONNECTIONS
        for end_node in range(graph.num_node):
            per_input_projections = []
            num_inputs = len(self.graph.nodes[end_node].in_nodes_indices)
            
            # if node receives raw input, make a projection layer for that too
            if end_node in self.graph.input_node_indices:
                num_inputs += 1
            
                stride, padding = self.calc_stride_padding(input_sizes[end_node], 
                                                      graph.nodes[end_node].input_size, 
                                                      graph.nodes[end_node].kernel_size)
                
                proj = nn.Conv2d(in_channels=input_dims[end_node], 
                                 out_channels=self.proj_hidden_dim,
                                kernel_size=graph.nodes[end_node].kernel_size,
                                stride=stride,
                                padding=padding,
                                device=device)
                per_input_projections.append(proj)
            
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
        #print(self.bottomup_projections)
            #self.bottomup_projections = nn.ModuleList(self.bottomup_projections)
        
        
        # PROJECTIONS FOR TOPDOWN INTERLAYER CONNECTIONS        
        for end_node in range(graph.num_node):
            # number of nodes it projects to (i.e everyone it receives feedback from)
            num_inputs = len(self.graph.nodes[end_node].out_nodes_indices)
            per_input_projections = []
            
            for start_node in self.graph.nodes[end_node].out_nodes_indices:
            
                # calculate convolution dimensions
                padding = self.calc_padding_transpose(graph.nodes[start_node].input_size, 
                                                      graph.nodes[end_node].input_size, 
                                                      graph.nodes[end_node].kernel_size)
                
                # projection from single node
                proj = nn.ConvTranspose2d(in_channels=graph.nodes[start_node].hidden_dim, 
                                 out_channels=self.proj_hidden_dim,
                                kernel_size=graph.nodes[end_node].kernel_size,
                                padding=padding, device=device)
                per_input_projections.append(proj)
            
            # Final conv to integrate all inputs. This convolution does not change shape of image
            integrator_conv = nn.Conv2d(in_channels=self.proj_hidden_dim*num_inputs, 
                                        out_channels=graph.nodes[end_node].input_dim,
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
        seq_len = max([i.shape[1] for i in all_inputs])
        process_time = seq_len + self.graph.num_node - 1
        if batch==True:
            batch_size = all_inputs[0].shape[0]
        else:
            batch_size = 1
            all_inputs = [torch.unsqueeze(inp, 0) for inp in all_inputs]
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
                    print(node)
                    #print(self.graph.input_node_indices)

                    # direct stimuli if node receives it + the sequence isn't done
                    if node in self.graph.input_node_indices and (t < seq_len):
                        inp = all_inputs[self.graph.input_node_indices.index(node)]
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
                    print(bottomup.shape)
                    
                    ##################################
                    # Find topdown inputs, assumes every feedforward connection out of node has feedback
                    # Note there's no external topdown input
                    topdown_projs = self.bottomup_projections[node]
                    
                    if self.topdown and t!=0: 
                        topdown = []

                        for i, topdown_node in enumerate(self.graph.nodes[node].out_nodes_indices):
                            topdown.append(topdown_projs[i](hidden_states_prev[topdown_node]))
                            print(topdown[0].shape)
                        
                        topdown = topdown_projs[-1](torch.cat(topdown, dim=1))
                        print(topdown.shape)
                        print('topdown')
                            
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
                    for next_node in self.graph.nodes[node].out_nodes_indices:
                        active_nodes.append(next_node)
                    
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
        