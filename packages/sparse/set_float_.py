# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:07:16 2024

@author: balazs
"""

import os 
import taichi as ti

# List of Taichi primitive types
primitive_types = [
    ti.i8, ti.i16, ti.i32, ti.i64,
    ti.u8, ti.u16, ti.u32, ti.u64,
    ti.f32, ti.f64
]

ti_types = {'f32': 'float32',
            'f64': 'float64',
            'i32': 'int32',
            'i64': 'int64'}

# Map Taichi primitive types to string representations of Torch dtypes
ti_to_torch_map = {
    ti.i8: 'torch.int8',
    ti.i16: 'torch.int16',
    ti.i32: 'torch.int32',
    ti.i64: 'torch.int64',
    ti.f16: 'torch.float16',
    ti.f32: 'torch.float32',
    ti.f64: 'torch.float64',
    # Torch does not have unsigned integer types
    ti.u8: None,
    ti.u16: None,
    ti.u32: None,
    ti.u64: None,
}

# Map Taichi primitive types to string representations of CuPy dtypes
ti_to_cupy_map = {
    ti.i8: 'cp.int8',
    ti.i16: 'cp.int16',
    ti.i32: 'cp.int32',
    ti.i64: 'cp.int64',
    ti.f16: 'cp.float16',
    ti.f32: 'cp.float32',
    ti.f64: 'cp.float64',
    # CuPy supports unsigned integer types
    ti.u8: 'cp.uint8',
    ti.u16: 'cp.uint16',
    ti.u32: 'cp.uint32',
    ti.u64: 'cp.uint64',
}

# Create the dictionary with to_string() as keys and the types as values
type_dict = {ptype.to_string(): ptype for ptype in primitive_types}

class DefaultFloat:
    def __init__(self, float_type = None):
        if not float_type:
            try:
                self._float = type_dict[os.environ['ti_defaul_float']]
            except KeyError:
                self._float = ti.f32
                os.environ['ti_defaul_float'] = 'f32'
        else:
            assert float_type in primitive_types
            self._float = float_type
            os.environ['ti_defaul_float'] = float_type.to_string()
    
    @property 
    def float_(self):
        return self._float 
    
    @float_.setter
    def float_(self, val):
        assert val in primitive_types
        self._float = val
        os.environ['ti_defaul_float'] = val.to_string()

float_ = DefaultFloat().float_

def to_std(ti_type):
    return ti_types[ti_type.to_string()]

def to_torch(taichi_type):
    if taichi_type not in ti_to_torch_map:
        raise ValueError(f"Unsupported Taichi type: {taichi_type}")
    
    torch_dtype_str = ti_to_torch_map[taichi_type]
    if torch_dtype_str is None:
        raise TypeError(f"Unsigned Taichi types are not supported in Torch: {taichi_type}")

    # Import torch dynamically
    import torch
    return eval(torch_dtype_str)

def to_cupy(taichi_type):
    if taichi_type not in ti_to_cupy_map:
        raise ValueError(f"Unsupported Taichi type: {taichi_type}")
    
    cupy_dtype_str = ti_to_cupy_map[taichi_type]
    if cupy_dtype_str is None:
        raise TypeError(f"Unsupported Taichi type for CuPy: {taichi_type}")

    # Import cupy dynamically
    import cupy as cp
    return eval(cupy_dtype_str)
    