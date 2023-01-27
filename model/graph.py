from model.topdown_gru import ConvGRUTopDownCell
import torch
from torch import nn
import torch.nn.functional as F

class Node:
    def __init__(self, index, input_size = (28, 28), input_dim = 1, hidden_dim = 10, kernal_size = (3,3)):
        self.index = index
        self.in_nodes_indices = [] #nodes passing values into current node #contains Node index (int)
        self.out_nodes_indices = [] #nodes being passed with values from current node #contains Node index (int)
        self.in_strength = [] #connection strengths of in_nodes
        self.out_strength = [] #connection strength of out_nodes
        self.rank_list = [] #default value


        #cell params
        (self.input_height, self.input_width) = input_size #Height and width of input tensor as (height, width).
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernal_size 

    def __eq__(self, other): 
        return self.index == other.index


class Graph(object):
    """ A brain architecture graph object, directed by default. """
    def __init__(self, connections, conn_strength, input_node_indices, output_node_index, signal_length, input_node_params = [], output_size = 10, directed=True, dtype = torch.cuda.FloatTensor, topdown = True, bias = False, reps = 2):
        self.input_node_indices = input_node_indices
        self.output_node_index = output_node_index
        self.input_node_params = input_node_params #a list of length len(input_node_indices), each consist of a list of 3: c,h,w 
        self.conn = connections #adjacency matrix
        self.conn_strength = conn_strength
        self.signal_length = signal_length
        self.directed = directed #flag for whether the connections are directed 
        self.num_node = len(self.conn)
        self.max_rank = -1
        self.num_edge = 0 #assuming directed edge. Will need to change if graph is undirected
        self.topdown = topdown
        self.nodes = self.generate_node_list(conn_strength)
        self.dtype = dtype 
        self.bias = bias
        self.output_size = output_size
        self.reps = reps
        #self.nodes[output_node].dist = 0
        #self.max_length = self.find_longest_path_length()
        for input_node in self.input_node_indices:
            self.rank_node(self.nodes[input_node],self.nodes[output_node_index], 0, 0)

        # for node in range (self.num_node):
            
        #     print(self.nodes[node].rank_list)



    def generate_node_list(self,conn_strength):
        nodes = []
        # initialize node list
        for n in range(self.num_node):
            if n in self.input_node_indices:
                i = self.input_node_indices.index(n)
                #node = Node(n, input_size= (self.input_node_params[i][1],self.input_node_params[i][2]), input_dim = self.input_node_params[i][0])
                if n == 3:
                    node = Node(n,input_dim = 10)
                else:
                    node = Node(n)
                
            else:
                node = Node(n)
            nodes.append(node)

        for in_node in range(self.num_node):
            for out_node in range(self.num_node):
                if (self.conn[in_node][out_node] != 0 ):
                    self.num_edge = self.num_edge + 1 # i dont want to use numpy so this is how im calculating the number of connections
                    nodes[in_node].out_nodes_indices.append(out_node)
                    nodes[out_node].in_nodes_indices.append(in_node) 
                    #TODO: add conn strength
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

    def find_feedforward_cells(self, node, t):
        in_nodes = []
        for n in self.nodes[node].in_nodes_indices:
            if ((t-1) in self.nodes[n].rank_list):
                in_nodes.append(n)
        return in_nodes

    def find_feedback_cells(self, node, t):
        return self.nodes[node].in_nodes_indices
        # if ((t+1) in self.nodes[n].rank_list):
        #     in_nodes.append(n)
        # return in_nodes

    def find_longest_path_length(self): #TODO: 
        return 42
        #I think this is not needed but we'll see

class Architecture(nn.Module):
    def __init__(self, graph):
        super(Architecture, self).__init__()

        self.graph = graph
        self.time = 4 #TODO: calculate time
        cell_list = []
        for node in range(graph.num_node):
            cell_list.append(ConvGRUTopDownCell(input_size=((self.graph.nodes[node].input_height, self.graph.nodes[node].input_width)), 
                                         input_dim=self.graph.nodes[node].input_dim, 
                                         hidden_dim = self.graph.nodes[node].hidden_dim,
                                         topdown_dim= self.graph.nodes[node].hidden_dim, #TODO: discuss with Mashbayar
                                         kernel_size=self.graph.nodes[node].kernel_size,
                                         bias=graph.bias,
                                         dtype=graph.dtype))
        self.cell_list = nn.ModuleList(cell_list)
        #this is some disgusting line of code
        dim1 = self.graph.nodes[self.graph.output_node_index].input_height * self.graph.nodes[self.graph.output_node_index].input_width * self.graph.nodes[self.graph.output_node_index].hidden_dim
        self.fc1 = nn.Linear(dim1, 100) 
        self.fc2 = nn.Linear(100, self.graph.output_size)
        

        #assuming all cells have the same input dim and the same hidden dim
        input_conv_list = []
        for input in range(len(self.graph.input_node_indices)):
            input_conv_list.append(nn.Conv2d(in_channels= self.graph.nodes[self.graph.input_node_indices[input]].input_dim,
                                out_channels= self.graph.nodes[self.graph.input_node_indices[input]].hidden_dim,
                                kernel_size=3,
                                padding=1,
                                device = 'cuda'))
        self.input_conv_list = nn.ModuleList(input_conv_list)

        #TODO: line #147 assumes same size&dimension across nodes AND assumes topdown input signal size of 10*10 and dimension of 10 -> 10*10*10
        self.topdown_input_proj = nn.Linear(10*10*10, self.graph.nodes[0].hidden_dim*self.graph.nodes[0].input_height *self.graph.nodes[0].input_height)
        
        bottomup_gru_list = []
        
        for node in range(self.graph.num_node):
            proj = nn.Linear((self.graph.nodes[node].hidden_dim*self.graph.num_node)*self.graph.nodes[node].input_height*self.graph.nodes[node].input_width, self.graph.nodes[node].input_dim*self.graph.nodes[node].input_height*self.graph.nodes[node].input_width)
            bottomup_gru_list.append(proj)
        self.bottomup_gru_list = nn.ModuleList(bottomup_gru_list)
        
        
        #to avoid the effort of a double loop, we find out the node that receives the most modulatory input at spme timestep t across all timesteps.
        topdown_gru_proj = []
        #max_mod_num_list = [] #a list of length n (number of nodes)
        #for node in range(self.graph.num_node):
            #max_mod_num = 0
            #for t in range(self.time):
                #if (max_mod_num < len(self.graph.find_feedback_cells(node, t))):
                   # max_mod_num = len(self.graph.find_feedback_cells(node, t))
          #  max_mod_num_list.append()
        
        topdown_gru_proj = []
        for node in range(self.graph.num_node):
            proj = nn.Linear((self.graph.nodes[node].hidden_dim * self.graph.num_node)*self.graph.nodes[node].input_height*self.graph.nodes[node].input_width, (self.graph.nodes[node].hidden_dim + self.graph.nodes[node].input_dim)*self.graph.nodes[node].input_height*self.graph.nodes[node].input_width)
            topdown_gru_proj.append(proj)
        self.topdown_gru_proj = nn.ModuleList(topdown_gru_proj)
            
                

        #self.topdown_gru_proj = nn.Linear(mod_sig.shape[1]*mod_sig.shape[2]*mod_sig.shape[3], (self.graph.nodes[node].input_dim+node_hidden_state.shape[1])*self.graph.nodes[node].input_height*self.graph.nodes[node].input_width)         
            
        
        
        #linear projection layer list
        #bottomup_linear_list = []
        #for i in range(0, self.num_node):
            #bottomup_linear_list.append(nn.Linear(bottomup.shape[1]*bottomup.shape[2]*bottomup.shape[3],
                                                #self.graph.nodes[node].input_dim*self.graph.nodes[node].input_height*self.graph.nodes[node].input_width)
        #self.bottomup_linear_list = nn.ModuleList(feedback_linear_list)


    def forward(self, input_tensor_list):
        """
        :param a list of tensors of size n, each consisting a input_tensor: (b, t, c, h, w). 
            n as the number of input signals. Order should correspond to self.graph.input_node_indices
        :param topdown: size (b,hidden,h,w) 
        :return: label 
        """
        temp = []
        num_inputs = len(input_tensor_list)
        for i in range(num_inputs):
            temp.append(input_tensor_list[i].shape[1])
        seq_len = max(temp) #find the time length of the longest input signal
        batch_size = input_tensor_list[0].shape[0]
        
        
        

        # Images in sequence goes through all layers one by one
        for rep in range(self.graph.reps):
            hidden_state_prev = self._init_hidden(batch_size=batch_size) #hidden_state_prev shape [n,b,c,h,w]
            hidden_state_cur = self._init_hidden(batch_size=batch_size)
            #for t in range(max(seq_len, self.graph.max_rank)):  #contains error
            for t in range(self.time):
                #before permute: b c t h w
                #TODO: error checks if input signal time series is longer than self.time
                current_inputs = [] #same shape and order as self.graph.input_node_indices # shape [n,b,c,h,w]
                for input in range(num_inputs):
                    if (t < self.graph.signal_length):
                    #current_inputs.append(self.input_conv_list[input](input_tensor_list[input][:, t, :, :, :]))
                        current_inputs.append(self.input_conv_list[input](input_tensor_list[input][:, t, :, :, :]))
                #current_input = self.input_conv(input_tensor[:, :, :, :])#[32,1,28,28]

                for node in range(self.graph.num_node):
                    #print('currently:','rep: ',rep,' t ',t,' node ',node)
                    bottomup_has_info = False
                    # Current state
                    node_hidden_state = hidden_state_prev[node] 

                    input_cells = self.graph.find_feedforward_cells(node, t)
                    # Bottom-up signal is either the state from the bottom layer or the input signal if it's the bottom-most layer
                    
                    #record direct input signal at the given time step
                    if node in self.graph.input_node_indices and t < self.graph.signal_length: #assumes signal input starts from t0
                        bottomup =  current_inputs[self.graph.input_node_indices.index(node)] #assumes the input node indices and the input signal indices matches in order e.g.: first input node receives first input signal
                        if (1 == self.graph.input_node_indices.index(node)): #if input signal is the topdown input signal, then use linear projection layer to reshape
                            bottomup = self.topdown_input_proj(torch.flatten(bottomup,start_dim=1))
                            bottomup = torch.reshape(bottomup, (bottomup.shape[0],hidden_state_prev[i].shape[1],  hidden_state_prev[i].shape[2],hidden_state_prev[i].shape[3]))
                            
                        bottomup_has_info = True

                    # processes the state from the bottom layer
                    for i in (input_cells):
                        if (i != node): #makes sure we do not consider current node's own previous state as input signal
                            if bottomup_has_info:                           
                                bottomup = torch.cat([bottomup, self.graph.conn_strength[i][node]*hidden_state_prev[i]], dim=1)
                            else: 
                                bottomup = self.graph.conn_strength[i][node]*hidden_state_prev[i]
                                bottomup_has_info = True
                      
                    if (bottomup.shape[1] < self.graph.nodes[node].hidden_dim*self.graph.num_node ):
                        pad_rows = self.graph.nodes[node].hidden_dim*self.graph.num_node - bottomup.shape[1]
                        pad = torch.zeros((bottomup.shape[0], pad_rows, bottomup.shape[2], bottomup.shape[3]),device=torch.device('cuda'))
                        bottomup = torch.cat([bottomup,pad], dim = 1)
                    

                    #now we handle ->modulartory<- signal •⌄•
                    mod_cells = self.graph.find_feedback_cells(node, t)
                    
                    #if there is no bottomup input, then ignore topdown feedback
                    if (bottomup_has_info == False or len(mod_cells) == 0): 
                        mod_sig = torch.zeros((hidden_state_prev[i].shape[0], self.graph.nodes[node].input_dim+node_hidden_state.shape[1], self.graph.nodes[node].input_height, self.graph.nodes[node].input_width)).to('cuda')
 
                    #modulartory signal present 
                    else: 
                        is_first = True #boolean to judge if is first pass
                        for i in range(self.graph.num_node):
                            if i in mod_cells: #then save the actual modulatory information
                                signal = self.graph.conn_strength[i][node]*hidden_state_prev[i]
                            else: #if the i th node has no modulartory signal for the current node, then pass along an array of zeros with shape
                                signal = torch.zeros((hidden_state_prev[i].shape)).to('cuda')
                            
                            if is_first:
                                mod_sig = signal
                            else:
                                mod_sig = torch.cat([mod_sig, signal], dim=1) 
                            is_first = False
                        
                        mod_sig = self.topdown_gru_proj[node](torch.flatten(mod_sig,start_dim=1))
                        mod_sig = torch.reshape(mod_sig, (mod_sig.shape[0], self.graph.nodes[node].input_dim+node_hidden_state.shape[1], self.graph.nodes[node].input_height, self.graph.nodes[node].input_width))

                    #now we are done with gathering bottomup and modulatory inputs for current cell
                    
                    #Update hidden state of this layer by feeding bottom-up, top-down and current cell state into gru cell
                    if (bottomup_has_info == False):
                        bottomup = torch.zeros(hidden_state_prev[i].shape[0], 1, self.graph.nodes[node].input_height, self.graph.nodes[node].input_width).to('cuda')
                    else:
                        bottomup = self.bottomup_gru_list[node](torch.flatten(bottomup,start_dim=1)) 
                        bottomup = torch.reshape(bottomup, (bottomup.shape[0],self.graph.nodes[node].input_dim, self.graph.nodes[node].input_height, self.graph.nodes[node].input_width))
                    
                    h = self.cell_list[node](input_tensor=bottomup, h_cur=node_hidden_state, topdown=mod_sig)


                    hidden_state_cur[node] = h
                #we are done with iterating through all cells at the current timestep
                hidden_state_prev = hidden_state_cur
        pred = self.fc1(F.relu(torch.flatten(hidden_state_cur[self.graph.output_node_index], start_dim=1)))
        pred = self.fc2(F.relu(pred))
          
        return pred

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.graph.num_node):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states