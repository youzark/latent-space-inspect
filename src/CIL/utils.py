#!/usr/bin/env python3
import torch
from typing import Dict, Tuple,Type,List
from torch import nn

# add mask to rectangle part of weight matrix of conv2d so that it's not updated during backward pass.{{{
def mask_conv_feature_block(
    weight,
    input_range_percentage: Tuple[float,float],
    output_range_percentage:Tuple[float,float]
    ):
    out_len,in_len,height,width = weight.shape
    in_start_per, in_end_per = input_range_percentage
    out_start_per, out_end_per = output_range_percentage
    in_start, out_start = (int)(in_start_per * in_len), (int)(out_start_per * out_len)
    in_end, out_end = (int)(in_end_per * in_len), (int)(out_end_per * out_len)
    def mask(grad):
        with torch.no_grad():
            grad[out_start:out_end,in_start:in_end,:,:] *= 0.
        return grad
    return weight.register_hook(mask)
# }}}

#  remove_conv_feature_block(): set part of conv2d weight matrix to zero.{{{
def remove_conv_feature_block(
    weight, 
    # start_percentage: Tuple[float,float], 
    # end_percentage: Tuple[float,float],
    input_range_percentage: Tuple[float,float],
    output_range_percentage:Tuple[float,float],
    ):
    """
    remove input features so that old task will be disturbed by parameters of new tasks
    start_percentage: (input_feature_percentage, out_feature_percentage)
    end_percentage: (input_feature_percentage, out_feature_percentage)
    """
    out_len,in_len,height,width = weight.shape
    in_start_per, in_end_per = input_range_percentage
    out_start_per, out_end_per = output_range_percentage
    in_start, out_start = (int)(in_start_per * in_len), (int)(out_start_per * out_len)
    in_end, out_end = (int)(in_end_per * in_len), (int)(out_end_per * out_len)
    with torch.no_grad():
        weight[out_start:out_end,in_start:in_end,:,:] *= 0.
# }}}

# remove_bn_feature_range() : set part of batchnorm2d weight and bias 

