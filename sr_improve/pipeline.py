"""
优化后的完整超分链路:
    输入图 → 退化程度判别 → 自适应预处理 → 按等级选超分器 → 重叠分块高斯融合 → 画质后处理矫正 → 输出图
      → 退化程度判别(四维特征 → 三档)
      → 自适应预处理(降噪 / 白平衡 / 亮度 / 锐化)
      → 按等级选超分器(GAN 增强权重由 dni 深度网络插值实现)
      → 重叠分块高斯融合(重叠分块 + 高斯加权融合)
      → 画质后处理矫正(亮度 / 饱和度 / 伪纹理 / 锐度)
"""

from .adaptive import (
    AdaptiveUpsampler, classify_image, damage_gate, damage_score, detail_score, preprocess
)
from .config import MODERATE, TILE_OVERLAP, TILE_SIZE, get_params
from .postprocess import postprocess


class ImprovedUpscaler:
    """优化超分器。超分器内部按等级懒加载并缓存。
    Args:
        weights_dir: 权重目录。
        tile: 分块尺寸,0 表示不分块。大图建议 512 起步。
        tile_overlap: 相邻块重叠像素数,0 附近等价于硬拼接。
        adaptive: 是否启用退化程度自适应预处理
        blend: 是否启用重叠分块高斯加权融合
        post: 是否启用画质后处理矫正
    """

    def __init__(self,
                 weights_dir='weights',
                 tile=TILE_SIZE,            # 分块推理单块尺寸
                 tile_overlap=TILE_OVERLAP, # tile块之间重叠像素,消除分块拼接接缝
                 tile_pad=10,               # 块边缘向外扩展像素，消除边缘伪影
                 pre_pad=0,                 # 图片预处理全局填充
                 half=True,                 # FP16半精度推理
                 gpu_id=None,               # GPU编号
                 adaptive=True,             # 自适应退化检测
                 blend=True,                # 重叠分块高斯加权融合
                 post=True):                # 画质后处理矫正
        self.adaptive = adaptive
        self.post = post
        self.upsampler = AdaptiveUpsampler(
            weights_dir=weights_dir,
            tile=tile,
            tile_overlap=tile_overlap,
            tile_pad=tile_pad,
            pre_pad=pre_pad,
            half=half,
            gpu_id=gpu_id,
            allow_dni=adaptive,
            blend=blend)


    #跑完链路，返回 (输出图, info)。
    def enhance(self, img, outscale=None, force_level=None):
        #   detail  缺不缺细节 → 决定 GAN 权重（缺细节就该让 GAN 多生成）
        #   damage  有没有可修复损伤 → 决定要不要做降噪等预处理
        alpha = None
        if self.adaptive:
            level, feats, score = classify_image(img)
            if force_level:
                level = force_level
            detail = detail_score(feats)
            damage = damage_score(feats)
            floor_a = get_params(level).dni_alpha
            alpha = floor_a + (1.0 - floor_a) * detail
        else:
            # 关掉退化程度自适应预处理时用"均衡增强"这一档，而不是某个极端等级
            level, feats, score, detail, damage = MODERATE, None, None, None, None

        params = get_params(level)
        effective_alpha = self.upsampler.alpha_used(level) if alpha is None else alpha
        info = {
            'level': level if self.adaptive else None,
            'score': score,
            'features': feats,
            'detail': detail,
            'damage': damage,
            'dni_alpha': effective_alpha,
            'post_strength': params.post_strength,
        }

        prepared = preprocess(img, params, damage=damage) if self.adaptive else img
        result, _ = self.upsampler.enhance(prepared, level, outscale=outscale, alpha=alpha)
        output = result[0]

        if self.post:
            # 后处理同样按损伤程度关门：干净图上这一整套矫正是净损失
            # 只有真的退化图才全额生效。
            output = postprocess(output, params, reference=img,
                                 strength_scale=damage_gate(damage))
        return output, info

    #各等级实际生效的 GAN 权重打印
    def describe(self):
        return self.upsampler.describe()
