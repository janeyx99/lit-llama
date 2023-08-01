import torch
import torch.nn.functional as F

def dequantize_per_tensor(int_repr, scale, zero_point, out_dtype=torch.float32):
    y = int_repr.to(out_dtype)
    if zero_point is not None:
        y -= zero_point
    return y * scale

def dynamically_quantize_per_tensor(
    x,
    quant_min,
    quant_max,
    target_dtype,
    qscheme=torch.per_tensor_affine,  # for now, reuse existing qscheme enum
):
    # assumes affine quantization

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    if qscheme == torch.per_tensor_affine:

        # get min and max
        # TODO(future): make torch.aminmax work on cpu-half
        # min_val, max_val = torch.aminmax(x)
        min_val = torch.min(x)
        max_val = torch.max(x)

        # calculate scale and zero point based on min and max
        # reference: https://fburl.com/code/srbiybme
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))
        device = min_val_neg.device

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        # TODO(future): make torch.clamp with scalar work on cpu-half
        scale = torch.clamp(scale, min=eps).reshape(1)
        zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)

        # quantize based on qmin/qmax/scale/zp
        # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
        quant = torch.clamp(torch.round(x / scale) + zero_point, quant_min, quant_max).to(target_dtype)

    else:
        assert qscheme == torch.per_tensor_symmetric, f"unsupported qscheme {qscheme}"
        # assert quant_min == -1 * quant_max, "unsupported quant_min/quant_max"
        amax = torch.max(torch.abs(x))
        scale = amax / (float(quant_max - quant_min) / 2)
        scale = torch.clamp(scale, min=eps).reshape(1)
        quant = torch.clamp(torch.round(x / scale), quant_min, quant_max).to(target_dtype)
        # do not create a tensor for zero_point as this is expensive
        zero_point = None

    return quant, scale, zero_point

class WeightOnlyQuantLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        w_int8 = kwargs.pop('w_int8')
        scale = kwargs.pop('scale')
        super().__init__(*args, **kwargs)
        self.w_int8 = torch.nn.Parameter(w_int8, requires_grad=False)
        self.scale = torch.nn.Parameter(scale, requires_grad=False)
        # self.register_buffer('w_int8', w_int8)
        # self.register_buffer('scale', scale)
        # self.w_int8 = w_int8
        # self.scale = scale

    def forward(self, x):
        w_fp16 = dequantize_per_tensor(
            self.w_int8, self.scale, zero_point=None, out_dtype=x.dtype)
        return F.linear(x, w_fp16, self.bias)
    
    @classmethod
    def from_float(cls, mod):
        w_fp32 = mod.weight 
        w_int8, scale, _zp = dynamically_quantize_per_tensor(
            w_fp32, -128, 127, torch.int8, torch.per_tensor_symmetric)
        # create the new module with a toy size to ensure initialization is fast
        fake_in_features, fake_out_features = 8, 8
        new_mod = cls(
            fake_in_features, fake_out_features, bias = mod.bias is not None,
            w_int8=w_int8, scale=scale)
        new_mod.in_features = mod.in_features
        new_mod.out_features = mod.out_features
        del new_mod.weight
        new_mod.bias = mod.bias
        device_to_use = next(mod.parameters()).device
        new_mod.to(device_to_use)
        return new_mod

def replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=''
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)` 
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == '':
            new_fqn = name
        else:
            new_fqn = f'{cur_fqn}.{name}'
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn)

def apply_weight_only_quant(model):
    replace_with_custom_fn_if_matches_filter(
        model,
        WeightOnlyQuantLinear.from_float,
        lambda mod, fqn: isinstance(mod, torch.nn.Linear))   