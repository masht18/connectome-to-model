from connectome_to_model.model.topdown_gru import ConvGRUBasalTopDownCell, ILC_upsampler
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import math

class ConnectomicsConvGRU(nn.Module):
    def __init__(self, graph, input_sizes, input_dims,
                 topdown=True,
                 dropout=True, dropout_p=0.25,
                 proj_hidden_dim=32):
        '''
        :param graph (Graph object)
        
        STIMULI-SPECIFIC PARAMETERS
        :param input_sizes (list of tuples)
            list of input sizes (height, width) per node. If a node doesn't receive an outside input, its input size if (0, 0)
        :param input_dims (list of ints)
            list of input channel sizes per node. If a node doesn't receive an outside input, its input dim is 0
            
        NEURAL NETWORK PARAMETERS
        :param topdown (bool):
            if enabled, uses at least multiplicative topdown
        :param dropout (bool)
        :param dropout_p (float)
            dropout probability if dropout is enabled
        :param proj_hidden_dim (int)
            hyperparam. intermediate dimension of projection convolutions
            
        '''
        super(ConnectomicsConvGRU, self).__init__()
        
        assert len(input_sizes) == graph.num_node, "Must give sizes of direct stimuli for all nodes. If a node receive direct input, use (0,0)."
        assert len(input_dims) == graph.num_node, "Must give channels of direct stimuli for all nodes. If a node receive direct input, use 0."

        self.graph = graph
        self.use_dropout = dropout
        self.dropout = nn.Dropout(dropout_p)
        self.topdown = topdown
        self.proj_hidden_dim = proj_hidden_dim
        
        self.bottomup_projections = nn.ModuleDict()
        self.topdown_projections = nn.ModuleDict()
        
        # The "brain areas" of the model
        cell_list = []
        for node in range(graph.num_node):
            cell_list.append(ConvGRUBasalTopDownCell(input_size=((graph.nodes[node].input_size[0], graph.nodes[node].input_size[1])), 
                                         input_dim=graph.nodes[node].input_dim, 
                                         hidden_dim = int(graph.nodes[node].hidden_dim),
                                         kernel_size=graph.nodes[node].kernel_size, 
                                        basal_topdown_dim = graph.nodes[node].basal_topdown_dim,
                                        apical_topdown_dim = graph.nodes[node].apical_topdown_dim,
                                         bias=graph.bias,
                                         dtype=graph.dtype))
        self.cell_list = nn.ModuleList(cell_list)
        
        # Store output sizes for readout
        self.output_sizes = []
        for i in self.graph.output_indices:
            h, w = self.graph.nodes[i].input_size
            self.output_sizes.append((h, w, self.graph.nodes[i].hidden_dim))
        
        self._build_bottomup_projections(graph, input_sizes, input_dims)
        self._build_topdown_projections(graph)

    def _build_bottomup_projections(self, graph, input_sizes, input_dims):
        """Build projection layers for bottom-up connections"""
        for end_node in range(graph.num_node):
            node_key = f"node_{end_node}"
            node_projections = nn.ModuleDict()
            num_inputs = 0
            
            # if node receives raw input, make a projection layer for that too
            if end_node in self.graph.input_indices:
                # calculate convolution dimensions
                stride, padding = self.calc_stride_padding(input_sizes[end_node], 
                                                      graph.nodes[end_node].input_size, 
                                                      graph.nodes[end_node].kernel_size)
              
                num_inputs += 1
                node_projections['input'] = nn.Conv2d(
                    in_channels=input_dims[end_node], 
                    out_channels=self.proj_hidden_dim,
                    kernel_size=graph.nodes[end_node].kernel_size,
                    stride=stride,
                    padding=padding)
            
            # dealing with the layer-to-layer projections
            for idx, start_node in enumerate(self.graph.nodes[end_node].in_nodes_indices):
                
                # calculate convolution dimensions
                stride, padding = self.calc_stride_padding(graph.nodes[start_node].input_size, 
                                                      graph.nodes[end_node].input_size, 
                                                      graph.nodes[end_node].kernel_size)
                
                # projection from single node
                node_projections[f'layer_{idx}'] = nn.Conv2d(
                    in_channels=graph.nodes[start_node].hidden_dim, 
                    out_channels=self.proj_hidden_dim,
                    kernel_size=graph.nodes[end_node].kernel_size,
                    stride=stride,
                    padding=padding)
                num_inputs += 1
            
            # Final conv to integrate all inputs. This convolution does not change shape of image
            if num_inputs > 0:
                node_projections['integrator'] = nn.Conv2d(
                    in_channels=self.proj_hidden_dim*num_inputs, 
                    out_channels=graph.nodes[end_node].input_dim,
                    kernel_size=3, padding=1)
            
            self.bottomup_projections[node_key] = node_projections

    def _build_topdown_projections(self, graph):
        """Build projection layers for top-down connections"""
        for end_node in range(graph.num_node):
            node_key = f"node_{end_node}"
            node_projections = nn.ModuleDict()
            
            # number of nodes it projects to (i.e everyone it receives feedback from)
            num_inputs = len(self.graph.nodes[end_node].fb_nodes)
            
            for idx, start_node in enumerate(self.graph.nodes[end_node].fb_nodes):
                bottom_size = graph.nodes[end_node].input_size
                top_size = graph.nodes[start_node].input_size
            
                # upsample
                if top_size[0]*top_size[1] < bottom_size[0]*bottom_size[1]:
                    stride = [o//i for o, i in zip(bottom_size, top_size)]
                    node_projections[f'feedback_{idx}'] = nn.ConvTranspose2d(
                        in_channels=graph.nodes[start_node].hidden_dim,
                        out_channels=self.proj_hidden_dim,
                        kernel_size=stride, stride=stride)
                else: # or downsample
                    stride, padding = self.calc_stride_padding(top_size, bottom_size, 
                                                      graph.nodes[start_node].kernel_size)
                    node_projections[f'feedback_{idx}'] = nn.Conv2d(
                        in_channels=graph.nodes[start_node].hidden_dim, 
                        out_channels=self.proj_hidden_dim,
                        kernel_size=graph.nodes[end_node].kernel_size,
                        stride=stride, padding=padding)
            
            # Final conv to integrate all inputs. This convolution does not change shape of image
            if num_inputs > 0:
                topdown_end_dim = graph.nodes[end_node].input_dim + graph.nodes[end_node].hidden_dim + graph.nodes[end_node].basal_topdown_dim + 2*graph.nodes[end_node].apical_topdown_dim
                node_projections['integrator'] = nn.Conv2d(
                    in_channels=self.proj_hidden_dim*num_inputs, 
                    out_channels=topdown_end_dim,
                    kernel_size=3, padding=1)
            
            self.topdown_projections[node_key] = node_projections

    def _compute_bottomup_input(self, node, t, seq_len, all_inputs, hidden_states_prev):
        """Compute bottom-up input for a specific node at time t"""
        node_key = f"node_{node}"
        if node_key not in self.bottomup_projections:
            return None
            
        node_projs = self.bottomup_projections[node_key]
        bottomup = []
        
        # Handle direct external input
        if node in self.graph.input_indices and 'input' in node_projs:
            inp = all_inputs[self.graph.input_indices.index(node)]
            
            if t < seq_len:
                # Active input
                bottomup.append(node_projs['input'](inp[:, t, :, :, :]))
            else:
                # Input finished, use zeros (network ruminating)
                bottomup.append(node_projs['input'](torch.zeros_like(inp[:, 0, :, :, :])))
        
        # Handle inputs from other nodes
        for idx, bottomup_node in enumerate(self.graph.nodes[node].in_nodes_indices):
            layer_key = f'layer_{idx}'
            if layer_key in node_projs:
                bottomup.append(node_projs[layer_key](hidden_states_prev[bottomup_node]))
        
        if not bottomup:
            return None
        
        # Concatenate and integrate all bottom-up inputs
        bottomup = torch.cat(bottomup, dim=1)
        if 'integrator' in node_projs:
            bottomup = node_projs['integrator'](bottomup)
        
        return bottomup

    def _compute_topdown_input(self, node, t, hidden_states_prev):
        """Compute top-down input for a specific node"""
        if not self.topdown:
            return None
        
        # No topdown at the beginning of sequence
        if t == 0:
            return None
        
        # Check if node has any feedback connections
        if self.graph.nodes[node].fb_nodes.nelement() == 0:
            return None
        
        node_key = f"node_{node}"
        if node_key not in self.topdown_projections:
            return None
            
        node_projs = self.topdown_projections[node_key]
        topdown = []
        
        for idx, topdown_node in enumerate(self.graph.nodes[node].fb_nodes):
            feedback_key = f'feedback_{idx}'
            if feedback_key in node_projs:
                topdown.append(node_projs[feedback_key](hidden_states_prev[topdown_node]))
        
        if not topdown:
            return None
        
        # Concatenate and integrate all top-down inputs
        topdown = torch.cat(topdown, dim=1)
        if 'integrator' in node_projs:
            topdown = node_projs['integrator'](topdown)
        
        return topdown

    def _update_active_nodes(self, old_active_nodes, all_inputs, t):
        """Update which nodes should be active in the next iteration"""
        active_nodes = []
        
        # Add nodes that receive input from currently active nodes
        for prev_node in old_active_nodes:
            for next_node in self.graph.nodes[prev_node].out_nodes_indices:
                if next_node not in active_nodes:
                    active_nodes.append(next_node.item())
        
        # Add direct input nodes if sequence isn't finished
        for inp_idx, input_node in enumerate(self.graph.input_indices):
            seq = all_inputs[inp_idx]
            seq_shape = seq.shape[1]
            if t + 1 < seq_shape:
                if input_node not in active_nodes:
                    active_nodes.append(input_node)
        
        return active_nodes

    def forward(self, all_inputs, batch=True, process_time=None, return_all=False):
        """
        :param all_inputs 
               list of tensor of size n, each consisting a input_tensor: (b, t, c, h, w). 
               n as the number of input streams. Order should correspond to self.graph.input_indices
        :return: list of outputs
            number of outputs equal to output indices 
        """
        # Find the time length of the longest input signal + enough extra time for processing
        seq_len = max([i.shape[1] for i in all_inputs])

        if process_time is None:
            process_time = seq_len + self.graph.num_node - 1
        
        # Handle batching
        if batch:
            batch_size = all_inputs[0].shape[0]
        else:
            batch_size = 1
            all_inputs = [torch.unsqueeze(inp, 0) for inp in all_inputs]
        
        # Initialize circular buffer with two sets of hidden states
        hidden_states_A = self._init_hidden(batch_size=batch_size)
        hidden_states_B = self._init_hidden(batch_size=batch_size)
        active_nodes = self.graph.input_indices.copy()
        
        # Main processing loop with circular buffer
        for t in range(process_time):
            
            # Alternate between the two state buffers
            current_states = hidden_states_A if t % 2 == 0 else hidden_states_B
            prev_states = hidden_states_B if t % 2 == 0 else hidden_states_A
            
            # Process each active node
            for node in active_nodes:
                
                # Compute bottom-up input using previous states
                bottomup = self._compute_bottomup_input(node, t, seq_len, all_inputs, prev_states)
                if bottomup is None:
                    continue
                
                # Compute top-down input using previous states
                topdown = self._compute_topdown_input(node, t, prev_states)
                
                # Process through the cell, reading from current_states[node] and writing to current_states[node]
                h = self.cell_list[node](bottomup, current_states[node], topdown)
                
                # Apply dropout if enabled
                if self.use_dropout:
                    h = self.dropout(h)
                
                # Write result back to current state buffer
                current_states[node] = h
            
            # Update active nodes for next iteration
            active_nodes = self._update_active_nodes(active_nodes, all_inputs, t)
        
        # Return final states from the last used buffer
        final_states = hidden_states_A if (process_time - 1) % 2 == 0 else hidden_states_B
        
        if return_all:
            return final_states
        
        if len(self.graph.output_indices) == 1:
            return final_states[self.graph.output_indices[0]] # Return single output if only one output node exists
        else:
            return [final_states[i] for i in self.graph.output_indices]

    def _init_hidden(self, batch_size):
        """Initialize hidden states for all nodes"""
        init_states = []
        for i in range(self.graph.num_node):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states
    
    def calc_stride_padding(self, input_sz, output_sz, kernel_sz):
        stride = [math.ceil((i - k) / (o - 1)) for i, o, k in zip(input_sz, output_sz, kernel_sz)]
        padding = [math.ceil(((o - 1)*s + k - i)/2) for i, o, k, s in zip(input_sz, output_sz, kernel_sz, stride)]
        return stride, padding
    
    def calc_padding_transpose(self, input_sz, output_sz, kernel_sz):
        padding = [math.ceil((o-i+k-1)/2) for i, o, k in zip(input_sz, output_sz, kernel_sz)]
        return padding