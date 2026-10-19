"""
轻量化画质后处理矫正（解决 GAN 失真缺陷）
Real-ESRGAN 基于 GAN,细节增强清晰但存在固有缺陷：色彩偏移、饱和度异常、
局部过曝、高频伪纹理。原生模型没有任何后处理矫正机制。

构建四步矫正 pipeline,由策略表的 post_strength 驱动：
    1. correct_brightness      亮度自适应矫正（对齐参考图亮度 + 压缩过曝高光）
    2. restore_saturation      饱和度还原校准（限幅 + 按参考图矫正色偏）
    3. suppress_pseudo_texture 高频伪纹理抑制
    4. final_adjust            整体锐度微调

参考图是"双线性上采样后的原始低清图"。它没有 GAN 引入的色偏与过饱和，
是个天然、无参数的色彩/亮度锚点。

"""
import cv2
import numpy as np

from .image_ops import match_size, unpack

    #把全局亮度朝参考图对齐，并压缩过曝高光。
def correct_brightness(bgr, reference, strength):
    if strength <= 0:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    cur = float(lab[:, :, 0].mean())
    ref = float(ref_lab[:, :, 0].mean())
    gain = 1.0 + (ref - cur) / max(cur, 1e-3)
    applied = float(np.clip(1.0 + strength * (gain - 1.0), 0.85, 1.15))
    lab[:, :, 0] = np.clip(lab[:, :, 0] * applied, 0.0, 100.0)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 过曝高光压缩：把最亮的 0.5% 往中间压，避免 GAN 局部过曝成死白
    highlight = float(np.percentile(out, 99.5))
    if highlight > 0.92:
        over = out > highlight
        out = np.where(over, highlight + (out - highlight) * 0.5, out)
    return np.clip(out, 0.0, 1.0)


    #限制饱和度上下限，再按参考图的通道均值矫正色偏。
def restore_saturation(bgr, reference, strength):
    if strength <= 0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    lower, higher = np.percentile(sat, [2.0, 98.0])
    if higher - lower > 1e-6:
        clamped = np.clip(sat, lower, higher)
        hsv[:, :, 1] = (1.0 - strength) * sat + strength * clamped
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 色偏矫正：以参考图的 B/G/R 均值为锚，整体拉回真实色调
    cur_mean = out.reshape(-1, 3).mean(axis=0)
    ref_mean = reference.reshape(-1, 3).mean(axis=0)
    gains = ref_mean / np.maximum(cur_mean, 1e-6)
    gains = np.clip(1.0 + strength * (gains - 1.0), 0.9, 1.1)   # 限幅，防止反而改坏
    return np.clip(out * gains.reshape(1, 1, 3), 0.0, 1.0)


def suppress_pseudo_texture(bgr, strength):
    """高频残差衰减,抑制GAN产生的虚假纹理:
    HF = img - bilateral(img)，输出 img - k·HF。
    双边滤波保边;因此被衰减的主要是细碎的高频起伏(GAN 容易编造
    重复纹理);真实边缘因为被保住、HF 很小，几乎不受影响。
    """
    if strength <= 0:
        return bgr
    smoothed = cv2.bilateralFilter(bgr, 7, 0.08, 5.0)
    high_freq = bgr - smoothed
    return np.clip(bgr - 0.6 * strength * high_freq, 0.0, 1.0)


    #整体锐度微调(USM)。放在最后，补回平滑掉的边缘清晰度
def final_adjust(bgr, strength):
    if strength <= 0:
        return bgr
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.5)
    return np.clip(bgr + 0.5 * strength * (bgr - blurred), 0.0, 1.0)


def postprocess(img, params, reference=None, return_info=False, strength_scale=1.0):
    """按策略表对一张超分结果做后处理矫正。

    Args:
        img: 超分输出,BGR / BGRA / 灰度,uint8 或 uint16。
        params (AdaptiveParams): 由退化等级选出的策略，用其中的 post_strength。
        reference: 原始低清图（双线性放大到同尺寸）。为 None 时跳过
            依赖色彩锚点的前两步，只做伪纹理抑制与锐化。
        return_info (bool): 为 True 时额外返回各步实际生效的强度。
        strength_scale (float): 总强度缩放系数，会乘到 post_strength 上。用
            它按"可修复损伤程度"关门:损伤为 0 时整段跳过；退化图才全额生效。

    Returns:
        矫正后的图像（同 dtype / 同通道数):return_info=True 时为 (图像, info)。
    """
    unpacked = unpack(img)
    work = unpacked.color

    ref_color = None
    if reference is not None:
        ref_color = match_size(work, unpack(reference).color)

    strength = params.post_strength * float(strength_scale)
    info = {}
    if ref_color is not None:
        work = correct_brightness(work, ref_color, strength)
        info['brightness'] = strength
        work = restore_saturation(work, ref_color, strength)
        info['saturation'] = strength
    work = suppress_pseudo_texture(work, strength)
    info['pseudo_texture'] = strength
    work = final_adjust(work, strength)
    info['final_adjust'] = strength

    out = unpacked.pack(work)
    return (out, info) if return_info else out
