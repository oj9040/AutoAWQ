import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy

class KVQuant:
    
    @classmethod
    def quantize(cls, x, bit=4, group_size=128, zero_point=True, shift_scale=False, alg=None, name=None, save_dir=None):

        export_fig = False
        
        if export_fig:
            x0_np = x[0,:].detach().cpu().numpy()
            climit = [x0_np.min(), x0_np.max()]
            kv_heatmap(x0_np, save_dir=save_dir, filename=f"{name}_org.png", climit=climit)
        
        #x_fq, _, _ = pseudo_quantize(x, bit, group_size, zero_point, shift_scale)
        #x_fq_error = torch.mean(torch.abs(x - x_fq))
        #print(f"{name} / kv_org_qerror = {x_fq_error}")
        
        if alg == "sageatten":
            #x_mean = torch.mean(x, dim=(0,1), keepdim=True) # reduce_dim = batch, seq -> (1, 1, channel)
            x_mean = torch.mean(x, dim=1, keepdim=True)  # reduce_dim = seq -> (batch, 1, channel)
            x_sub = x - x_mean
            
            if export_fig:
                x0_sub_np = x_sub[0,:].detach().cpu().numpy()
                kv_heatmap(x0_sub_np, save_dir=save_dir, filename=f"{name}_meansub.png", climit=climit)

            x_sub_fq, _, _ = pseudo_quantize(x_sub, bit, group_size, zero_point, shift_scale)   
            x_sub_fq = x_sub_fq + x_mean
            #x_sub_fq_error = torch.mean(torch.abs(x - x_sub_fq))
            #print(f"{name} / kv_sageatten_qerror = {x_sub_fq_error}")
            x = x_sub_fq
            
        elif alg == "smoothatten":
            raise NotImplementedError
        else:
            x, _, _ = pseudo_quantize(x, bit, group_size, zero_point, shift_scale)

        return x


def kv_heatmap(x: np.ndarray, save_dir="./", filename: str="fig.png", climit: list=None):
    plt.imshow(x, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.xlabel("Channel")
    plt.ylabel("Sequence")
    #"\n".join(wrap("Some really really long long long title I really really need - and just can't - just can't - make it any - simply any - shorter - at all.", 60))
    #plt.title("\n".join(wrap(filename[:-4], 30)), loc="center")
    plt.title(f"{filename[:-4]}", loc="center", wrap=True)
    if climit is not None:
        plt.clim(climit[0], climit[1])
    plt.savefig(save_dir + "/" + filename)
    plt.close()


def pseudo_quantize(x, bit=4, group_size=128, zero_point=True, shift_scale=False):

    org_x_shape = x.shape
    
    if group_size > 0:
        assert org_x_shape[-1] % group_size == 0, f"org_w_shape ({org_x_shape[-1]}) must be a multiple of group_size ({group_size})!"
        x = x.reshape(-1, group_size)
    assert x.dim() == 2
    assert torch.isnan(x).sum() == 0

    if zero_point:
        max_val = x.amax(dim=-1, keepdim=True)
        min_val = x.amin(dim=-1, keepdim=True)
        max_int = 2**bit - 1
        min_int = 0

        if shift_scale:
            shifts = torch.floor(torch.log2((max_val - min_val).clamp(min=1e-5))) - bit
            scales = 2**shifts
        else:
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        x = (
            torch.clamp(torch.round(x / scales) + zeros, min_int, max_int) - zeros
        ) * scales
        zeros = zeros.view(org_x_shape[0], org_x_shape[1], -1)
    else:
        max_val = x.abs().amax(dim=-1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (bit - 1) - 1
        min_int = -(2 ** (bit - 1))
        
        if shift_scale:
            shifts = torch.floor(torch.log2(max_val.clamp(min=1e-5))) - (bit - 1)
            scales = 2**shifts
        else:
            scales = max_val / max_int
        
        ## original block
        #zeros = None
        #w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales
        ## 
        
        ## awq does not support signed int kernel, so instead use unsigned int and fixed zero point for symmetric quant
        ## modified by jude.jh.oh (2025-02-24)
        zeros = -1 * min_int * torch.ones_like(scales)
        x = (
            torch.clamp(torch.round(x / scales) + zeros, 0, max_int-min_int) - zeros
        ) * scales
        zeros = zeros.view(org_x_shape[0], org_x_shape[1], -1)
        
    scales = scales.view(org_x_shape[0], org_x_shape[1], -1)
    x = x.reshape(org_x_shape)
        
    return x, scales, zeros
        
        
        
        