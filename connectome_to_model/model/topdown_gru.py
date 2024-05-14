import os
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class ConvGRUBasalTopDownCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, 
                 basal_topdown_dim=0,
                 apical_topdown_dim=0,
                 bias=True,
                 device='cuda',
                 dtype=torch.cuda.FloatTensor):
        """
        Single ConvGRU block with topdown
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param apical_mechanism (str)
            'multiplicative' or 'composite', how to combine top-down info within the block
        :param basal_topdown_dim (int)
            if there's no basal topdown input, use 0
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUBasalTopDownCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        self.apical_topdown_dim = apical_topdown_dim
        self.basal_topdown_dim = basal_topdown_dim
        self.device = device

        # Basal compartment
        if basal_topdown_dim == 0:
            self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                        out_channels=2*self.hidden_dim,  # for update_gate,reset_gate + 2*topdown
                                        kernel_size=kernel_size,
                                        padding= (kernel_size[0] // 2, kernel_size[1] // 2),
                                        bias=self.bias)
        else:
            self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim + basal_topdown_dim,
                                        out_channels=2*self.hidden_dim,  # for update_gate,reset_gate + 2*topdown
                                        kernel_size=kernel_size,
                                        padding= (kernel_size[0] // 2, kernel_size[1] // 2),
                                        bias=self.bias)
        
        # Apical compartment
        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim+apical_topdown_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur, topdown):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: topdown: (b, c_topdown, h, w),
            topdown signal, either a direct clue or hidden of top layer
        """
        b, in_dim, h, w = input_tensor.shape
        mult_topdown_dim = in_dim + self.hidden_dim + 2*self.apical_topdown_dim
        if topdown == None:
            topdown = torch.zeros(b, mult_topdown_dim + self.basal_topdown_dim, h, w, device=self.device)
        
        # BASAL COMPARTMENT
        if self.basal_topdown_dim != 0:
            if topdown == None:
                basal_topdown = torch.zeros(b, self.basal_topdown_dim, h, w, device=self.device)
            else:
                basal_topdown, topdown = torch.split(topdown, (self.basal_topdown_dim, topdown.shape[1]-self.basal_topdown_dim), dim=1)
            combined = torch.cat([input_tensor, h_cur, basal_topdown], dim=1)
        else:
            combined = torch.cat([input_tensor, h_cur], dim=1)
            
        combined_conv = self.conv_gates(combined)
     
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        # APICAL COMPARTMENT
        if self.apical_topdown_dim != 0:
            add_topdown, mult_topdown = torch.split(topdown, (self.apical_topdown_dim, mult_topdown_dim - self.apical_topdown_dim), dim=1)
            combined = torch.cat([input_tensor, reset_gate*h_cur, add_topdown], dim=1) * (F.relu(mult_topdown) + 1)
        else:
            # multiplicative topdown
            combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1) * (F.relu(topdown) + 1)
            
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        # MEMORY UPDATE
        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next
    
    
class ConvGRUTopDownCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, 
                 topdown_type='multiplicative', 
                 bias=True, 
                 dtype=torch.cuda.FloatTensor):
        """
        Single ConvGRU block with topdown
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param topdown_type (str)
            'multiplicative' or 'composite', how to combine top-down info within the block
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUTopDownCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype
        self.topdown_type = topdown_type

        if self.topdown_type == 'multiplicative':
            self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                        out_channels=2*self.hidden_dim,  # for update_gate,reset_gate + 2*topdown
                                        kernel_size=kernel_size,
                                        padding= (kernel_size[0] // 2, kernel_size[1] // 2),
                                        bias=self.bias)
        elif self.topdown_type == 'composite':
            self.conv_gates = nn.Conv2d(in_channels=(input_dim + hidden_dim)*2,
                                        out_channels=2*self.hidden_dim,  # for update_gate,reset_gate + 2*topdown
                                        kernel_size=kernel_size,
                                        padding= (kernel_size[0] // 2, kernel_size[1] // 2),
                                        bias=self.bias)
        else:
            raise Warning('Topdown mechanism not implemented')
            
        
        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim, # for candidate neural memory
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur, topdown):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: topdown: (b, c_topdown, h, w),
            topdown signal, either a direct clue or hidden of top layer
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        if topdown == None:
            topdown = torch.zeros_like(combined)
        if self.topdown_type == 'composite':
            combined = torch.cat([input_tensor, h_cur, topdown], dim=1)
            
        combined_conv = self.conv_gates(combined)
     
        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1) * (F.relu(topdown) + 1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next
    
class ILC_upsampler(nn.Module):
    def __init__(self, in_channel, out_channel, stride, device='cuda'):
        """
        Projection layer for upsampling. Prevents checkerboard effect of a single ConvT
        
        :param in_channel (int)
        :param out_channel (int)
        :param stride (int)
        """
        super(ILC_upsampler, self).__init__()
        self.c1 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channel,
                       out_channels=out_channel,
                       kernel_size=(1,1),
                       stride=stride, device=device
                      ),
                        nn.ReLU(),
                           nn.ZeroPad2d((0,1,0,1))
                          )
        self.c2 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channel,
                       out_channels=out_channel,
                       kernel_size=(1,1),
                       stride=stride, device=device
                      ),
                        nn.ReLU(),
                           nn.ZeroPad2d((1,0,1,0))
                          )
        self.c3 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channel,
                       out_channels=out_channel,
                       kernel_size=(1,1),
                       stride=stride, device=device
                      ),
                        nn.ReLU(),
                           nn.ZeroPad2d((1,0,0,1))
                          )
        self.c4 = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channel,
                       out_channels=out_channel,
                       kernel_size=(1,1),
                       stride=stride, device=device
                      ),
                        nn.ReLU(),
                           nn.ZeroPad2d((0,1,1,0))
                          )
        
    def forward(self, z):
        c1_out=self.c1(z)
        c2_out=self.c2(z)
        c3_out=self.c3(z)
        c4_out=self.c4(z)
        upsampled_z=c1_out+c2_out+c3_out+c4_out
        
        return upsampled_z