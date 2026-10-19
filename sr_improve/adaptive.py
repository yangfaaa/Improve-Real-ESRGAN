"""
图像退化程度自适应增强（判别 → 预处理 → 自适应超分）。
    1 退化程度判别
       提取四维判别特征，归一化后加权成综合退化分，按阈值切成轻/中/重三档。
    2 自适应预处理
       按等级动态施加降噪 / 白平衡 / 亮度均衡 / 锐化，强度
       干净图不改动。
    3 自适应超分
       按等级选超分器，GAN 增强权重由深度网络插值真实生效。
"""

import os
import warnings

import cv2
import numpy as np

from .config import (
    BLOCK_SIZE, DAMAGE_GATE, FEATURE_SIZE, FEATURE_WEIGHTS, LEVELS, LOW_FREQ_RADIUS, MILD, MODERATE,
    NOISE_BLOCK, NOISE_PERCENTILE, SEVERE, THRESHOLDS, TILE_OVERLAP, get_params
)
from .image_ops import unpack

# 1 退化程度判别
# ------------------------------图像规整------------------------------------
def to_gray_float(img):
    """BGR / BGRA / 灰度 + uint8 / uint16 → [0,1] 的 float32 单通道灰度图。"""
    if img.ndim == 3:
        arr = img[:, :, :3] if img.shape[2] == 4 else img
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # 位深判断：
    is_16bit = img.dtype == np.uint16
    gray = gray.astype(np.float32)
    gray /= 65535.0 if is_16bit else 255.0
    return gray


    """统一归一化长边尺寸，消除图像尺寸对统计量的影响。
    **双向都归一**（缩用 INTER_AREA、放用 INTER_CUBIC）：块方差是尺度相关的统计量，
    必须让所有图落到同一尺度上再量。曾经是"只缩不放"，小图停在原生尺度上被量，
    双三次下采样抬高逐像素块方差 → 干净的低清图被读成"有噪声"（实测 9 张干净 LR
    有 5 张被判重度）。放大确有插值高频的副作用，但远小于尺度不统一的危害。
    """
def _resize_long_side(gray, size=FEATURE_SIZE):
    h, w = gray.shape[:2]
    scale = size / float(max(h, w))
    if abs(scale - 1.0) < 1e-3:
        return gray
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                      interpolation=interp)


# ---------------------------- 四个特征--------------------------------------
def _variance(gray):
    return float(np.var(gray))
    #像素方差：反映整体均匀度。
    """把灰度图切成 block×block 的块，返回每块的方差向量；图太小返回 None。
    块方差是下面两个特征的共同基础：低分位块反映噪声底，块方差之间的
    离散度反映纹理/噪声分布是否"紊乱"。
    """
def _block_variances(gray, block=BLOCK_SIZE):
    h, w = gray.shape[:2]
    bh, bw = h // block, w // block
    if bh < 2 or bw < 2:
        return None
    patches = gray[:bh * block, :bw * block].reshape(bh, block, bw, block).transpose(0, 2, 1,3).reshape(-1, block * block)

    return patches.var(axis=1)


    #块方差离散度 = (各块方差)std标准差 / (各块方差)mean均值。
def _var_dispersion(block_var):
    if block_var is None:
        return 0.0
    mean_var = float(block_var.mean())
    if mean_var <= 1e-8:
        return 0.0
    return float(block_var.std() / mean_var)


    """噪声强度估计（sigma）：取方差最低的一批块，用其平均方差开方。
    平坦块里的方差基本只来自噪声。
    """
def _noise_floor(block_var, percentile=NOISE_PERCENTILE):
    if block_var is None:
        return None
    flattest = block_var[block_var <= np.percentile(block_var, percentile)]
    if flattest.size == 0:
        return None
    return float(np.sqrt(max(float(flattest.mean()), 0.0)))


    #小图：块数不足时退回中值滤波残差
def _residual_noise(gray):
    u16 = np.clip(gray * 65535.0, 0, 65535).astype(np.uint16)
    med = cv2.medianBlur(u16, 3).astype(np.float32) / 65535.0
    return float(np.std(gray - med))


    #Sobel 横纵梯度均值：越低越模糊
def _gradient(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.hypot(gx, gy)))


    #二维 FFT 高频能量占比：越低细节缺失越严重
def _hf_ratio(gray):
    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = gray.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    total = float(mag.sum())
    if total <= 0:
        return 0.0
    mask = radius > LOW_FREQ_RADIUS * min(h, w)
    return float(mag[mask].sum() / total)


    #提取四维（+方差）判别特征，返回 plain dict
def extract_features(img):
    gray = _resize_long_side(to_gray_float(img))
    coarse_var = _block_variances(gray, BLOCK_SIZE)      # 用于"方差紊乱"度
    fine_var = _block_variances(gray, NOISE_BLOCK)       # 用于噪声底估计
    noise = _noise_floor(fine_var)
    if noise is None:                                    # 图太小，块统计不可用
        noise = _residual_noise(gray)
    return {
        'variance': _variance(gray),                    # 整图灰度方差
        'var_dispersion': _var_dispersion(coarse_var),  # 四维特征1：块方差离散度
        'gradient': _gradient(gray),                    # 四维特征2：Sobel梯度均值
        'hf_ratio': _hf_ratio(gray),                    # 四维特征3：高频能量占比
        'noise': noise,                                 # 四维特征4：噪声强度
    }


# ------------------------------- 归一化与分级-----------------------------------
    #线性归一到 [0,1]。(lower,higher) 来自阈值表，lower 为干净端。
def _norm(value, lower, higher):
    if higher <= lower:
        return 0.0
    return float(np.clip((value - lower) / (higher - lower), 0.0, 1.0))



def score_terms(feats, th=None):
    """把四维特征折成四个"越大越退化"的分项
     gradient 与 hf_ratio 是"越低越退化"，所以取 1 - norm。
    """
    th = THRESHOLDS if th is None else th
    return {
        'blur': 1.0 - _norm(feats['gradient'], *th['gradient']),
        'noise': _norm(feats['noise'], *th['noise']),
        'hf': 1.0 - _norm(feats['hf_ratio'], *th['hf_ratio']),
        'var': _norm(feats['var_dispersion'], *th['var_dispersion']),
    }


def damage_gate(damage, floor=DAMAGE_GATE):
    """把连续损伤度折成"开关"：低于 floor 归零，floor~1 之间线性放开。
    直接拿 damage 当缩放系数不够：分级器会把细节丰富的干净图误判成中度退化，
    damage 为 None（关闭自适应）时返回 1.0，
    """
    if damage is None:
        return 1.0
    d = float(np.clip(damage, 0.0, 1.0))
    if d <= floor:
        return 0.0
    return float((d - floor) / max(1.0 - floor, 1e-6))


def detail_score(feats, th=None):
    """细节缺失程度 [0,1]：越大越缺细节 → GAN 权重就该越大。
    不含 noise：噪声本身也是高频，混进来会把"图很脏"误读成"细节丰富"
    """
    th = THRESHOLDS if th is None else th
    blur = 1.0 - _norm(feats['gradient'], *th['gradient'])
    hf = 1.0 - _norm(feats['hf_ratio'], *th['hf_ratio'])
    return float(np.clip((blur + hf) / 2.0, 0.0, 1.0))


def damage_score(feats, th=None):
    """可修复损伤程度 [0,1]：即噪声强度，越大越该做降噪。
    一张干净的低清图 detail_score 很高但 damage_score 为 0（没有噪声）
    它需要的是**更强的 GAN去生成细节**，而不是降噪。
    """
    th = THRESHOLDS if th is None else th
    return float(_norm(feats['noise'], *th['noise']))


#综合退化分，范围 [0,1]，越大越退化。
def degradation_score(feats, th=None):
    terms = score_terms(feats, th)
    weight_sum = sum(FEATURE_WEIGHTS.values())
    score = sum(FEATURE_WEIGHTS[k] * v for k, v in terms.items()) / weight_sum
    return float(np.clip(score, 0.0, 1.0))


    #按综合退化分的两个切分点判级，返回 'mild' / 'moderate' / 'severe'。
def classify(feats, th=None):
    th = THRESHOLDS if th is None else th
    score = degradation_score(feats, th)
    lower, higher = th['score']
    if score < lower:
        return MILD
    if score < higher:
        return MODERATE
    return SEVERE


    #(退化等级,特征张量,置信分数) → (level, feats, score)。
def classify_image(img, th=None):
    feats = extract_features(img)
    return classify(feats, th), feats, degradation_score(feats, th)



# 2 自适应预处理
# -------------------------- 单步算子----------------------------------------
    #降噪双边滤波而非高斯：过度高斯的会连真实细节一起抹掉。strength=0 时原图返回。
def denoise(bgr, strength):
    if strength <= 0:
        return bgr
    d = int(3 + 4 * strength) * 2 + 1          # 半径 3~7，即核 7~15
    # 输入在 [0,1]，sigmaColor 必须取小值，防止抹掉所有边缘，变成油画效果
    sigma_color = 0.02 + 0.06 * strength    #灰度差值
    sigma_space = 3.0 + 5.0 * strength      #空间距离
    filtered = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
    return cv2.addWeighted(bgr, 1.0 - strength, filtered, strength, 0.0)


    #灰世界白平衡，按 strength 混合。
def adaptive_white_balance(bgr, strength):
    if strength <= 0:
        return bgr
    channel_means = bgr.reshape(-1, 3).mean(axis=0)
    if float(channel_means.min()) <= 1e-6:
        return bgr
    # 计算增益：三通道总均值 ÷ 单通道均值
    gains = float(channel_means.mean()) / channel_means
    if float(np.abs(gains - 1.0).max()) < 0.03:
        return bgr
    gains = np.clip(gains, 0.85, 1.15)
    gains = 1.0 + strength * (gains - 1.0)
    return np.clip(bgr * gains.reshape(1, 1, 3), 0.0, 1.0)


    #全局亮度朝中灰(50)调整，按 strength 混合。
def adaptive_brightness(bgr, strength):
    if strength <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0]
    mean_lum = float(lum.mean())
    target = 50.0
    if abs(mean_lum - target) < 15.0:
        return bgr
    # 计算亮度增益，限制增益区间，防止调整幅度过大
    gain = float(np.clip(1.0 + (target - mean_lum) / max(mean_lum, 1e-3), 0.90, 1.10))
    applied = 1.0 + strength * (gain - 1.0)
    lab[:, :, 0] = np.clip(lum * applied, 0.0, 100.0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


    #非锐化掩模锐化算法，原图 − 模糊图 = 得到边缘细节
def usm_sharpen(bgr, strength, radius=2.0):
    if strength <= 0:
        return bgr
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=radius)
    return np.clip(bgr + 0.6 * strength * (bgr - blurred), 0.0, 1.0)


# ------------------------------ 主入口------------------------------------
def preprocess(img, params, damage=None, return_info=False):
    """按策略表对一张图做自适应预处理。
    Args:
        img: BGR / BGRA / 灰度;uint8 或 uint16。
        params (AdaptiveParams): 由退化等级选出的策略（见 config.STRATEGY)。
        damage (float|None): 可修复损伤程度 [0,1]。
            降噪强度：干净图 damage≈0;降噪整步跳过，不会把干净图抹糊。
            亮度/白平衡自带"门控色彩增益"接近 1,说明色彩本身很均衡，自动跳过白平衡校正。
        return_info (bool): 为 True 时额外返回实际生效的步骤 dict。

    Returns:
        处理后的图像（同 dtype / 同通道数);return_info=True 时为 (图像, info)。
    """

    unpacked = unpack(img)
    work = unpacked.color
    gate = damage_gate(damage)

    info = {}
    for name, strength, op in (
        ('denoise', params.pre_denoise * gate, denoise),
        ('white_balance', params.white_balance * gate, adaptive_white_balance),
        ('brightness', params.brightness * gate, adaptive_brightness),
        ('usm', params.pre_usm * gate, usm_sharpen),
    ):
        if strength > 0:
            work = op(work, strength)
            info[name] = strength

    out = unpacked.pack(work)
    return (out, info) if return_info else out



# 3 自适应超分（GAN 增强权重）
# Real-ESRGAN "GAN 增强权重"这个参数。但 GAN 模型 RealESRGAN_x4plus
# 与非 GAN 模型 RealESRNet_x4plus 结构完全相同（都是 23 块 RRDBNet），把两者逐层
# 插值 [1-α, α] 就得到一条从"纯 峰值信噪比 导向"到"纯 GAN"的连续过渡，α 即 GAN 增强权重：
#     α = 0.8 → 细节最锐利，但最容易过锐化/出假纹理（重度退化）
#     α = 0.5 → 明显抑制假纹理（中度退化）
#     α = 0.25 → 最保守（轻度退化）
#插值结果按 α 缓存在 weights/dni_cache/ 下。

GAN_WEIGHT = 'RealESRGAN_x4plus.pth'
NET_WEIGHT = 'RealESRNet_x4plus.pth'

# 官方权重按优先级探测
_CHECKPOINT_KEYS = ('params_ema', 'params')


#权重从本地 weights/ 目录读取
def ensure_weight(path):
    return os.path.isfile(path)


    #指定 weights_only=False，完整加载 checkpoint，允许读取字典、
    #对象等全部存储内容。避免 torch 2.4  FutureWarning。
def _torch_load(path):
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _checkpoint_key(ckpt):
    for key in _CHECKPOINT_KEYS:
        if isinstance(ckpt, dict) and key in ckpt:
            return key
    return None


    #检查权重是否可用
def _dni_ready(gan_path, net_path):
    try:
        gan_key = _checkpoint_key(_torch_load(gan_path))
        net_key = _checkpoint_key(_torch_load(net_path))
    except (OSError, RuntimeError, ValueError):
        return False
    return gan_key is not None and net_key is not None


    #双网络插值：把非 GAN 与 GAN 权重按 [1-α, α] 逐层混合，返回路径。
    #(W_{interp} = (1-\alpha)*W_{net} + alpha*W_{gan})
def blend_dni(net_path, gan_path, alpha, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'dni_a{alpha:.3f}.pth')
    if os.path.isfile(cache_path):
        return cache_path

    import torch
    net_ckpt, gan_ckpt = _torch_load(net_path), _torch_load(gan_path)
    net_key, gan_key = _checkpoint_key(net_ckpt), _checkpoint_key(gan_ckpt)
    if net_key is None or gan_key is None:
        raise KeyError(f'权重里找不到 {_CHECKPOINT_KEYS} 中的任何一个索引')

    blended = {}
    for name, value in net_ckpt[net_key].items():
        if not torch.is_floating_point(value):
            blended[name] = value
            continue
        blended[name] = (1.0 - alpha) * value.float() + alpha * gan_ckpt[gan_key][name].float()

    torch.save({'params': blended}, cache_path)
    return cache_path


    #按退化等级构建并缓存 RealESRGANer，每个等级一套 GAN 权重。
    #超分器懒加载：只有某个等级真的被用到才构建，避免 3 个模型常驻显存。
class AdaptiveUpsampler:
    def __init__(self,
                 weights_dir,
                 tile=0,
                 tile_overlap=TILE_OVERLAP,
                 tile_pad=10,
                 pre_pad=0,
                 half=True,
                 gpu_id=None,
                 allow_dni=True,
                 blend=True):
        self.weights_dir = weights_dir
        self.tile = tile
        self.tile_overlap = tile_overlap
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.half = half
        self.gpu_id = gpu_id
        self.blend = blend
        self.cache_dir = os.path.join(weights_dir, 'dni_cache')

        self.gan_path = os.path.join(weights_dir, GAN_WEIGHT)
        self.net_path = os.path.join(weights_dir, NET_WEIGHT)
        self._cache = {}

        if not ensure_weight(self.gan_path):
            raise FileNotFoundError(
                f'本地缺少 GAN 权重 {self.gan_path}，请先将其放入 {weights_dir}/')

        # 只有在两个权重都可用且键名兼容时，GAN 权重才真正可调
        self.dni_available = bool(
            allow_dni and ensure_weight(self.net_path)
            and _dni_ready(self.gan_path, self.net_path))
        if allow_dni and not self.dni_available:
            warnings.warn(
                f'GAN 增强权重不可用（缺少或键名不匹配 {NET_WEIGHT}），'
                '所有等级将退回同一个 GAN 模型，dni_alpha 实际不生效。'
                f'请把 {NET_WEIGHT} 放入 {weights_dir}/',
                RuntimeWarning)

    #显式解析设备
    def _device(self):
        import torch
        if self.gpu_id is not None and torch.cuda.is_available():
            return torch.device(f'cuda:{self.gpu_id}')
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _build(self, alpha):
        # tiling 顶层依赖 torch，延迟导入以保持本模块的轻量导入特性
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        from .tiling import BlendedRealESRGANer
        # x4plus 与非 GAN 版结构一致，共用同一套定义
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        if self.dni_available and alpha < 1.0:
            model_path = blend_dni(self.net_path, self.gan_path, alpha, self.cache_dir)
        else:
            model_path = self.gan_path
        # blend=True 时用重叠分块高斯加权融合的子类（重叠 + 高斯加权融合），否则退回上游硬拼接
        cls = BlendedRealESRGANer if self.blend else RealESRGANer
        return cls(
            scale=4,
            model_path=model_path,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=self.half,
            device=self._device(),
            **({'tile_overlap': self.tile_overlap} if self.blend else {}),
        )

    #按退化等级取超分器（懒加载 + 缓存）。alpha 显式给出时覆盖策略表里的档位权重。
    def get(self, level, alpha=None):
        if alpha is None:
            alpha = self.alpha_used(level)
        if not self.dni_available:      # 插值不可用时只能跑纯 GAN
            alpha = 1.0

        # 到 dni_cache/ 并常驻内存
        alpha = round(round(float(alpha) / 0.05) * 0.05, 3)
        if alpha not in self._cache:
            self._cache[alpha] = self._build(alpha)
        return self._cache[alpha]

    #该等级实际生效的 GAN 权重（dni 不可用时恒为 1.0）。
    def alpha_used(self, level):
        if not self.dni_available:
            return 1.0
        return get_params(level).dni_alpha

    def enhance(self, img, level, outscale=None, alpha=None):
        return self.get(level, alpha).enhance(img, outscale=outscale), get_params(level)

    #列出各等级实际生效的 GAN 权重
    def describe(self):
        return {
            level: {
                'dni_alpha': self.alpha_used(level),
                'dni_active': self.dni_available and self.alpha_used(level) < 1.0,
            } for level in LEVELS
        }
