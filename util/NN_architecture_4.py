import torch
import torch.nn as nn
from MODELS import cbam
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(x.size(0), -1)
class Smooth(nn.Module):
    def forward(self,x):
        x=torch.sqrt(torch.square(x))
        return x

class AT_CNN_block(nn.Module):
    def __init__(self,input_channel=1,output_channel=1,kernel_size=2,stride=1,padding=2,drop_out_rate=0.15,no_spatial=False):
        super(AT_CNN_block, self).__init__()
        self.input_channel=input_channel
        self.output_channel=output_channel
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.drop_out_rate=drop_out_rate
        self.no_spatial=no_spatial
        self.CNNB=torch.nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm1d(output_channel),
            nn.ReLU(),
            cbam.CBAM(gate_channels=self.output_channel,reduction_ratio=16,no_spatial=self.no_spatial),
            nn.Dropout(drop_out_rate)
        )
    def forward(self,x):
        return self.CNNB(x)
    



class GEXNN_AT_para(nn.Module):
    def __init__(self,feature_len=200,gr_len=84,config_para=dict()):
        super(GEXNN_AT_para, self).__init__()
        self.feature_len=feature_len
        self.gr_len=gr_len
        self.config_para=config_para
        # kernel size must be odd
        self.CNNAT=torch.nn.Sequential(
            AT_CNN_block(input_channel=1,output_channel=self.config_para['output1'],kernel_size=self.config_para['kernel1'],stride=1,padding=(self.config_para['kernel1']-1)//2,drop_out_rate=self.config_para['dropout1'],no_spatial=False),
            AT_CNN_block(input_channel=self.config_para['output1'],output_channel=self.config_para['output2'],kernel_size=self.config_para['kernel2'],stride=1,padding=(self.config_para['kernel2']-1)//2,drop_out_rate=self.config_para['dropout2'],no_spatial=False), 
            AT_CNN_block(input_channel=self.config_para['output2'],output_channel=self.config_para['output3'],kernel_size=self.config_para['kernel3'],stride=1,padding=(self.config_para['kernel3']-1)//2,drop_out_rate=self.config_para['dropout3'],no_spatial=False),
            AT_CNN_block(input_channel=self.config_para['output3'],output_channel=self.config_para['output4'],kernel_size=self.config_para['kernel4'],stride=1,padding=(self.config_para['kernel4']-1)//2,drop_out_rate=self.config_para['dropout4'],no_spatial=False),
            Flatten(),
            nn.Linear(self.config_para['output4']*self.feature_len, self.config_para['linear1']),
            nn.SELU(),
            nn.Linear(self.config_para['linear1'],self.gr_len))
    def forward(self,x):
        return self.CNNAT(x)
