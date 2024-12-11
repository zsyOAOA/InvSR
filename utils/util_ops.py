#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2024-08-15 16:25:07

def append_dims(x, target_dims:int):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]
