import torch
import copy


group_size = 64
w_bit = 4

w = torch.rand((1024, 1024), dtype=torch.float16)
w_org = copy.deepcopy(w)
org_w_shape = w.shape

max_val_f = w.abs().amax(dim=1, keepdim=True)
min_val_f = max_val_f.clamp(min=1e-5)
bit_f = 8
max_int_f = 2 ** (bit_f - 1) - 1
min_int_f = -(2 ** (bit_f - 1))
scales_f = max_val_f / max_int_f
w_f = torch.clamp(torch.round(w / scales_f), min_int_f, max_int_f) * scales_f
w = w_f  # overwrite w with int8_fakequant(w)

if group_size > 0:
    assert org_w_shape[-1] % group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({group_size})!"
    w = w.reshape(-1, group_size)
assert w.dim() == 2
assert torch.isnan(w).sum() == 0

max_val = w.amax(dim=1, keepdim=True)
min_val = w.amin(dim=1, keepdim=True)
max_int = 2**w_bit - 1
min_int = 0

# shift scale
shifts = torch.floor(torch.log2((max_val - min_val).clamp(min=1e-5))) - w_bit
scales = 2**shifts

zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
w = (
    torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
) * scales
w1 = w.reshape(org_w_shape)

error1 = torch.mean(torch.abs(w1 - w_org))
print(f"quant error 1 = {error1}")

#####################################################

w = copy.deepcopy(w_org)
org_w_shape = w.shape
group_size = 64

max_val_f = w.abs().amax(dim=1, keepdim=True)
min_val_f = max_val_f.clamp(min=1e-5)
bit_f = 8
max_int_f = 2 ** (bit_f - 1) - 1
min_int_f = -(2 ** (bit_f - 1))
scales_f = max_val_f / max_int_f
w_q = torch.clamp(torch.round(w / scales_f), min_int_f, max_int_f)

if group_size > 0:
    assert org_w_shape[-1] % group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({group_size})!"
    w = w.reshape(-1, group_size)
assert w.dim() == 2
assert torch.isnan(w).sum() == 0

max_val = w_q.amax(dim=1, keepdim=True)
min_val = w_q.amin(dim=1, keepdim=True)
max_int = 2**w_bit - 1
min_int = 0

# shift scale
shifts = torch.floor(torch.log2((max_val - min_val).clamp(min=1e-5))) - w_bit
scales = 2**shifts

zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
w_qd = (
    torch.clamp(torch.round(w_q / scales) + zeros, min_int, max_int) - zeros
) * scales

w = w_qd * scales_f
w2 = w.reshape(org_w_shape)

error2 = torch.mean(torch.abs(w2 - w_org))
print(f"quant error 2 = {error2}")

import pdb; pdb.set_trace()