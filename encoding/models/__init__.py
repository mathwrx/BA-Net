# -*- coding: utf-8 -*-
# @File    : __init__.py.py


from .BANet import *
from .base import *

def get_segmentation_model(name, **kwargs):
    return get_model(name, **kwargs)


