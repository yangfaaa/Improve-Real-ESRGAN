"""Real-ESRGAN 改进模块包

    退化程度自适应预处理 → adaptive.py
    重叠分块高斯加权融合 → tiling.py
    画质后处理矫正 → postprocess.py
    全链路编排 → pipeline.py（ImprovedUpscaler）
"""
from .adaptive import (
    AdaptiveUpsampler, classify, classify_image, degradation_score, extract_features, preprocess
)
from .config import (
    AdaptiveParams, LEVELS, LEVEL_NAMES_CN, MILD, MODERATE, SEVERE, STRATEGY, THRESHOLDS, get_params
)

__all__ = [
    'AdaptiveParams', 'AdaptiveUpsampler', 'LEVELS', 'LEVEL_NAMES_CN', 'MILD', 'MODERATE', 'SEVERE',
    'STRATEGY', 'THRESHOLDS', 'classify', 'classify_image', 'degradation_score', 'extract_features',
    'get_params', 'preprocess'
]
