import torch
from torch import nn
from typing import *

from transformers import Trainer
from transformers.adapters import AdapterConfig
from transformers.adapters.modeling import Adapter

from transformers.activations import ACT2FN

from torch.nn import init
from torch import Tensor


class SimpleGate(nn.Module):
    def __init__(self, size=768):
        super(SimpleGate, self).__init__()
        self.tweights = nn.Parameter(torch.ones(size))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self):
        return self.sigmoid(self.tweights)

class BottleNeckCNNAdapter(nn.Module):
    def __init__(self, c_in, c_out, rank=32, kernel_size=1, stride=1, bias=False):
        super(BottleNeckCNNAdapter, self).__init__()
        self.conv1 = nn.Conv1d(c_in, rank, kernel_size=kernel_size, stride=stride, bias=bias)
        # self.activ = nn.ReLU()
        self.activ1= nn.GELU() 
        self.conv2 = nn.Conv1d(rank, c_out, kernel_size=1, groups=rank, stride=1, bias=bias, padding='same')
        self.activ2= nn.GELU() 
        self.layer_norm = nn.LayerNorm(c_out, elementwise_affine=True)


    def forward(self, x, res):
        o = self.conv1(x)
        o = self.activ1(o)
        # o = o.transpose(1, 2)
        # o = self.layer_norm(o)
        # o = o.transpose(1, 2)
        o = self.conv2(o)  
        o = self.activ2(o + res)
        # o = o + res
        return o
    
class DepthWiseCNNAdapter(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=1, stride=1, bias=False):
        super(DepthWiseCNNAdapter, self).__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=kernel_size, groups=c_in, stride=stride, bias=bias)
        # self.activ = nn.ReLU()
        self.activ = nn.GELU() 
        self.conv2 = nn.Conv1d(c_out, c_in, kernel_size=kernel_size, groups=c_in, stride=1, bias=bias, padding='same')

    def forward(self, x, res):
        o = self.conv1(x)
        o = self.activ(o)
        o = self.conv2(o)    
        o = o + res
        return o

    

class BottleneckAdapter(nn.Module):
	def __init__(self, adapter_name, input_size, down_sample):
		super(BottleneckAdapter, self).__init__()
		self.config = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")
		self.bottleneck_adapter = Adapter(adapter_name, input_size=input_size, down_sample=down_sample, config=self.config)
	def forward(self, x, residual_input):
		output, down, up = self.bottleneck_adapter(x, residual_input)
		return output


class BottleneckCNNAdapter(nn.Module):
    def __init__(self,):
        super(BottleneckCNNAdapter, self).__init__()

class SimpleCNNAdapter(nn.Module):
    def __init__(self, size):
        self.cnn_layer_weights = nn.Parameter(torch.ones(size, 1, 1))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, ):
        return self.sigmoid(self.cnn_layer_weights)


