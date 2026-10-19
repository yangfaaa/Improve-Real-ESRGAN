"""退化程度自适应预处理的集中配置：特征提取参数、退化分级阈值、三档自适应策略表。
可调参数：
  - 本文件：特征提取参数、特征权重、默认阈值、策略表(AdaptiveParams)；
  - 同目录 thresholds.json：由 tools/calibrate_thresholds.py 标定后产出，优先级高于
    本文件里的 DEFAULT_THRESHOLDS。阈值可根据数据集微调只需要改 json，不用动代码。
"""

import json
import os
from dataclasses import dataclass, asdict

# ------------------------------- 特征提取参数---------------------------------
FEATURE_SIZE = 512   # 归一化图像长边尺寸
BLOCK_SIZE = 32      # 统计"块方差离散度"时的块边长
# 噪声块尺寸与分位是 2026-10-11 在 100 源图上实测的结论（块 2/3/4/6/8/10/12/16 × 分位 0.2~10 全扫过）：
# 3×3 + 0.3% 给出 mild↔moderate 3.33σ / moderate↔severe 8.63σ（4×4 + 0.5% 只有 2.95σ / 7.95σ）。
# 再大的块（6/8/16）会吃到纹理，使干净图的噪声读数从 0 一路铺到 1.0、与中度档重叠。
NOISE_BLOCK = 3      # 噪声的块边长：比 BLOCK_SIZE 小更容易找到平坦图像块。
# 取最低 0.3% 的块求方差底。压这么低是为了推开 mild↔moderate；代价是真平坦块样本变少，
# 长边 179px 的图上只剩个位数个块，标定与部署遇到小图时要留意。
NOISE_PERCENTILE = 0.3
LOW_FREQ_RADIUS = 0.15  # FFT 低频半径，取 min(H,W) 的比例

# ------------------------------ 分块推理参数----------------------------------
TILE_SIZE = 512      # 默认分块尺寸，0 为不分块；图小于该值时不触发分块
TILE_OVERLAP = 64    # 相邻分块的重叠像素数（越大越平滑）

# ------------------------------ 退化等级定义----------------------------------
MILD = 'mild'          # 轻度退化
MODERATE = 'moderate'  # 中度退化
SEVERE = 'severe'      # 重度退化
LEVELS = (MILD, MODERATE, SEVERE)
LEVEL_NAMES_CN = {MILD: '轻度退化', MODERATE: '中度退化', SEVERE: '重度退化'}

# -------------------------- 预处理/后处理的门槛damage -------------------------
# 损伤度低于此值视为"干净图"，整条预处理与后处理链一步不走。
# 分级器会把细节丰富的干净图误判成退化，若直接用连续损伤度当缩放系数，误判的图仍会
# 按比例被降噪/矫正而受损。加一道门槛后只有明确判定为退化的图才执行。
# 该值由 calibrate_thresholds.py  自动标定（取干净样本damage 的高分位），写入
#  thresholds.json 的 damage_gate ；缺失时回退到默认值。
# 同样取自最近一次标定（干净集 damage 的 p75）：默认取 p95 会挡掉 83% 的中度图。
DEFAULT_DAMAGE_GATE = 0.146

# ------------------------------ 分级阈值----------------------------------
# 每项为 (lower, higher)：lower 是"最干净"端，higher 是"最退化"端，用于把原始特征线性归一到 [0,1]
# 前四项是特征本身的归一化区间，score 项是综合退化分 → 三档的切分点

# 先验默认阈值；可用的阈值应由 calibrate_thresholds.py 标定。
# 下面这组取自最近一次标定（degraded_set_div2k，×4 尺度），只为让"删掉 thresholds.json"
# 时的回退不至于失效 —— 旧的一组（gradient 0.005~0.06、hf_ratio 0.01~0.12 等）与实际特征
# 尺度差 2.6~9 倍，会让 gradient / hf_ratio / var_dispersion 全部饱和、分级失去区分度。
DEFAULT_THRESHOLDS = {
    'gradient': (0.155, 0.546),        # 梯度均值，越低越模糊
    'noise': (0.000, 0.014),           # 平坦块方差底，越高噪声越强
    'hf_ratio': (0.455, 0.799),        # 高频能量占比，越低细节越缺失
    'var_dispersion': (0.477, 1.524),  # 块方差离散度，越高方差越"紊乱"
    'score': (0.2864, 0.5377),         # 综合退化分 → 三档切分点
}

# 四个特征在综合退化分里的权重（统一"越大越退化"方向），同样取自最近一次标定
DEFAULT_FEATURE_WEIGHTS = {
    'blur': 0.25,   # 由 gradient 反推
    'noise': 0.75,  # 由 noise 正推
    'hf': 0.0,      # 由 hf_ratio 反推（非单调，标定权重为 0）
    'var': 0.0,     # 由 var_dispersion 正推（非单调，标定权重为 0）
}
_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thresholds.json')


    #读取标定文件，返回 (thresholds_raw, weights_raw)；缺失或损坏时返回空 dict
def _read_calibration():
    if not os.path.isfile(_CALIB_PATH):
        return {}, {}
    try:
        with open(_CALIB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (ValueError, OSError):
        return {}, {}
    if not isinstance(data, dict):
        return {}, {}
    return data.get('thresholds', {}) or {}, data.get('feature_weights', {}) or {}


    #默认阈值 + thresholds.json 覆盖，返回 plain dict
def load_thresholds():
    th = {k: tuple(v) for k, v in DEFAULT_THRESHOLDS.items()}
    raw, _ = _read_calibration()
    for key, value in raw.items():
        if key in th and isinstance(value, (list, tuple)) and len(value) == 2:
            th[key] = (float(value[0]), float(value[1]))
    return th


    #默认权重 + thresholds.json 覆盖，返回 plain dict
def load_feature_weights():
    weights = dict(DEFAULT_FEATURE_WEIGHTS)
    _, raw = _read_calibration()
    for key, value in raw.items():
        if key in weights:
            weights[key] = float(value)
    return weights


    #读取标定出的损伤门槛；缺失或非数值时返回 None
def _read_damage_gate():
    if not os.path.isfile(_CALIB_PATH):
        return None
    try:
        with open(_CALIB_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (ValueError, OSError):
        return None
    value = data.get('damage_gate') if isinstance(data, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


    #标定结果写入 thresholds.json，供后续推理复用
def save_calibration(thresholds, feature_weights, damage_gate=None):
    payload = {
        'thresholds': {k: [float(x) for x in v] for k, v in thresholds.items()},
        'feature_weights': {k: float(v) for k, v in feature_weights.items()},
    }
    if damage_gate is not None:
        payload['damage_gate'] = float(damage_gate)
    with open(_CALIB_PATH, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return _CALIB_PATH


THRESHOLDS = load_thresholds()
FEATURE_WEIGHTS = load_feature_weights()
THRESHOLDS_PATH = _CALIB_PATH

# 损伤门槛优先用标定值，缺失时回退默认值
_CALIBRATED_GATE = _read_damage_gate()
DAMAGE_GATE = DEFAULT_DAMAGE_GATE if _CALIBRATED_GATE is None else _CALIBRATED_GATE


# -------------------------------- 自适应策略表--------------------------------
    #一个退化等级对应的一整套推理超参数替代原生 Real-ESRGAN 的固定参数模式
@dataclass
class AdaptiveParams:
    level: str
    dni_alpha: float      # GAN 增强权重 α：0=纯 RealESRNet，1=纯 RealESRGAN
    pre_denoise: float    # 超分前的降噪强度 0~1
    pre_usm: float        # 超分前的非锐化掩模锐化强度 0~1
    brightness: float     # 自适应亮度均衡 0~1
    white_balance: float  # 灰世界白平衡 0~1
    post_strength: float  # 后处理强度
    note: str = ''

    def to_dict(self):
        return asdict(self)


# 轻度 → 弱化 GAN 增强权重，杜绝过锐化、假纹理；
# 中度 → GAN与PSNR折中，均衡增强，兼顾细节修复与噪点抑制；
# 重度 → 高GAN + 强化降噪，大量清理噪点、修复模糊。
STRATEGY = {
    MILD: AdaptiveParams(
        level=MILD,
        dni_alpha=0.25,         # α 下限
        pre_denoise=0.0,        # 超分前预处理降噪强度[0‑1]
        pre_usm=0.0,            # 超分前非锐化掩模锐化强度[0‑1]
        brightness=0.0,         # 超分前亮度朝中灰调整的混合强度[0‑1]
        white_balance=0.0,      # 白平衡校正强度[0‑1]
        post_strength=0.20,     # 超分后综合处理强度[0‑1]
        note='弱化GAN;干净图不触发任何预处理',
    ),
    MODERATE: AdaptiveParams(
        level=MODERATE,
        dni_alpha=0.5,
        pre_denoise=0.30,
        pre_usm=0.0,
        brightness=0.30,
        white_balance=0.30,
        post_strength=0.5,
        note='均衡增强，兼顾细节修复与噪点抑制',
    ),
    SEVERE: AdaptiveParams(
        level=SEVERE,
        dni_alpha=0.80,
        pre_denoise=0.75,
        pre_usm=0.50,
        brightness=0.60,
        white_balance=0.50,
        post_strength=0.75,
        note='高GAN + 强化降噪 + 高频重建，大量清理噪点与模糊',
    ),
}

    #按等级取策略；未知等级回退到中度
def get_params(level):
    return STRATEGY.get(level, STRATEGY[MODERATE])
