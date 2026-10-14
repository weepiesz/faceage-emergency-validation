"""Shared utilities for the FaceAge emergency study.

Single import point for path configuration, data loading, statistical tests and plotting: the
analysis scripts do `from utils import ...` rather than reaching into the submodules directly."""

__version__ = '1.0.0'
__author__ = 'FaceAge ICU Research Team'

# 导入主要功能
from .config import *
from .data_loader import *
from .statistical_tests import *
from .visualization import *
