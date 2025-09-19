import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F


class _quantize_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, step_size, half_lvls):
        # ctx is a context object that can be used to stash information
        # for backward computation
        ctx.step_size = step_size
        ctx.half_lvls = half_lvls
        output = F.hardtanh(input,
                            min_val=-ctx.half_lvls * ctx.step_size.item(),
                            max_val=ctx.half_lvls * ctx.step_size.item())

        output = torch.round(output / ctx.step_size)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone() / ctx.step_size

        return grad_input, None, None


quantize = _quantize_func.apply


class quan_Conv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True):
        super(quan_Conv2d, self).__init__(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=bias)
        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place change MSB to negative

    def forward(self, input):
        if self.inf_with_weight:
            return F.conv2d(input, self.weight * self.step_size, self.bias,
                            self.stride, self.padding, self.dilation,
                            self.groups)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            return F.conv2d(input, weight_quan, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # return F.conv2d(input, self.weight, self.bias, self.stride,
        #                     self.padding, self.dilation, self.groups)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reduction_weight__(self):
        self.weight.data=self.weight*self.step_size

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True



class quan_fixed_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,weight,index,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, pni='layerwise', w_noise=False): 
        super(quan_fixed_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation,
                                        groups=groups, bias=None)
        self.bias=bias
        self.pni = pni
        if self.pni == 'layerwise':
            self.alpha_w = nn.Parameter(torch.Tensor([0.25]), requires_grad=True)
        elif self.pni == 'elementwise':
            self.alpha_w = nn.Parameter(self.weight.clone().fill_(0.1), requires_grad=True)
        self.w_noise = w_noise
        self.N_bits = 8
        self.full_lvls = 2 ** self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(
            2 ** torch.arange(start=self.N_bits - 1, end=-1, step=-1).unsqueeze(-1).float(),
            requires_grad=False)

        self.b_w[0] = -self.b_w[0]  # in-place change MSB to negative

        # Duplicate one out channel's weights while keeping others unchanged
        self.index=index    #卷积核索引位置
        self.weight=weight
        self.duplicate_last_out_channel()

    def duplicate_last_out_channel(self):
        # Duplicate the weights of the last output channel
        # l=[self.index]
        with torch.no_grad():
            for index in self.index:     #多通道改为self.index
                original_weight = self.weight.clone()
                dupli_weight=original_weight[index,:,:,:].unsqueeze(0)
                new_weight = torch.cat([original_weight, dupli_weight], dim=0)
                self.weight = nn.Parameter(new_weight, requires_grad=True)
                # Update bias if present
                if self.bias is not None:
                    new_bias = torch.cat([self.bias, self.bias[self.index].clone().unsqueeze(0)], dim=0)
                    self.bias = nn.Parameter(new_bias.cuda(), requires_grad=True)
                    # new_bias = torch.cat([self.bias, self.bias[self.index].clone()], dim=0)
                    # self.bias = nn.Parameter(new_bias.cuda(), requires_grad=True)
    
    
           
    def forward(self, input):
        
        if self.inf_with_weight:
            output = F.conv2d(input, self.weight * self.step_size, self.bias, self.stride, self.padding,
                              self.dilation, self.groups)
        else:
            weight_quan = quantize(self.weight, self.step_size, self.half_lvls) * self.step_size
            output = F.conv2d(input, weight_quan, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # Split the output into original and duplicated parts, and combine them
        # original_output = output[:, :-1, :, :]
        # duplicated_output = output[:, -1:, :, :]
        # final_output = 0.5 * (original_output + duplicated_output)

        for i, index in enumerate(self.index):
            temp1 = output[:, self.out_channels + i, :, :]
            temp2 = output[:, index, :, :]
            std_1 = torch.std(temp1.flatten())  # 重复通道的标准差
            std_2 = torch.std(temp2.flatten())  # 原始通道的标准差
            
            if std_1 < std_2:  # 替换为标准差更小的通道
                output[:, index, :, :] = temp1.clone()

        # 返回前 self.out_channels 个通道
    
        res = output[:, :self.out_channels, :, :]
        return res

    def __add_weight__(self,index):
        self.index.append(index)
        with torch.no_grad():
            original_weight = self.weight.clone()
            dupli_weight=original_weight[index,:,:,:].unsqueeze(0)
            new_weight = torch.cat([original_weight, dupli_weight], dim=0)
            self.weight = nn.Parameter(new_weight, requires_grad=True)
            # Update bias if present
            if self.bias is not None:
                new_bias = torch.cat([self.bias, self.bias[self.index].clone().unsqueeze(0)], dim=0)
                self.bias = nn.Parameter(new_bias.cuda(), requires_grad=True)

    def __reduction_weight__(self):
        self.weight.data = self.weight * self.step_size

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls

    def __reset_weight__(self):
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size, self.half_lvls)
        self.inf_with_weight = True



class quan_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(quan_Linear, self).__init__(in_features, out_features, bias=bias)

        self.N_bits = 8
        self.full_lvls = 2**self.N_bits
        self.half_lvls = (self.full_lvls - 2) / 2
        # Initialize the step size
        self.step_size = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        self.__reset_stepsize__()
        # flag to enable the inference with quantized weight or self.weight
        self.inf_with_weight = False  # disabled by default

        # create a vector to identify the weight to each bit
        self.b_w = nn.Parameter(2**torch.arange(start=self.N_bits - 1,
                                                end=-1,
                                                step=-1).unsqueeze(-1).float(),
                                requires_grad=False)

        self.b_w[0] = -self.b_w[0]  #in-place reverse

    def forward(self, input):
        if self.inf_with_weight:
            return F.linear(input, self.weight * self.step_size, self.bias)
        else:
            self.__reset_stepsize__()
            weight_quan = quantize(self.weight, self.step_size,
                                   self.half_lvls) * self.step_size
            return F.linear(input, weight_quan, self.bias)
        # return F.linear(input, self.weight, self.bias)

    def __reset_stepsize__(self):
        with torch.no_grad():
            self.step_size.data = self.weight.abs().max() / self.half_lvls
    def __reduction_weight__(self):
        self.weight.data=self.weight*self.step_size

    def __reset_weight__(self):
        '''
        This function will reconstruct the weight stored in self.weight.
        Replacing the orginal floating-point with the quantized fix-point
        weight representation.
        '''
        # replace the weight with the quantized version
        with torch.no_grad():
            self.weight.data = quantize(self.weight, self.step_size,
                                        self.half_lvls)
        # enable the flag, thus now computation does not invovle weight quantization
        self.inf_with_weight = True
