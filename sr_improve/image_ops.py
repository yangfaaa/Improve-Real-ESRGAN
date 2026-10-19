"""预处理与后处理的图像拆包/打包工具。
外部输入 `cv2.imread` 读取的原图，支持 BGR/BGRA/ 灰度，数据类型可为 uint8、uint16；
内部运算统一转为 `[0,1]` 区间 float32 的 BGR 格式，处理完成后通过 `pack()` 恢复回原始的数据类型与通道数。
一套逻辑，兼容 16 位图像、RGBA 图像，供预处理、后处理两处逻辑复用。
"""

from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class Unpacked:
    color: np.ndarray       # [0,1] float32，BGR 三通道
    alpha: object           # 原始的 alpha 透明度通道，无则 None
    is_gray: bool           # True=灰度图，False=彩色图
    is_rgba: bool           # True=RGBA，False=BGR/BGA/灰度
    max_range: float        # 像素值范围，255.0 或 65535.0
    dtype: object           # 原始图片数据类型。 dtype

    #把 [0,1] float32 的 BGR 还原成原始 dtype / 通道数
    def pack(self, color):
        out = np.rint(np.clip(color * self.max_range, 0.0, self.max_range))
        if self.is_gray:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = out.astype(self.dtype)
        if self.is_rgba:
            out = np.dstack([out, self.alpha])
        return out


    #拆分图像，灰度图会被提升成 3 通道便于统一运算，pack() 时会转回单通道。
def unpack(img):
    if img is None:
        raise ValueError('收到 None，应先过滤掉读取失败的图片')
    if img.ndim == 2:
        is_gray, is_rgba = True, False
    elif img.shape[2] == 4:
        is_gray, is_rgba = False, True
    else:
        is_gray, is_rgba = False, False

    if is_rgba:
        alpha = img[:, :, 3].copy()
        color_src = img[:, :, :3]
    else:
        alpha = None
        color_src = img

    max_range = 65535.0 if int(img.max()) > 256 else 255.0
    color = color_src.astype(np.float32) / max_range
    if is_gray:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)

    return Unpacked(color=color, alpha=alpha, is_gray=is_gray, is_rgba=is_rgba,
                    max_range=max_range, dtype=img.dtype)


    #把参考图缩放到与 基准图同尺寸（后处理做色彩锚点）
def match_size(color, reference):
    h, w = color.shape[:2]
    if reference.shape[:2] == (h, w):
        return reference
    return cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA if
                      reference.shape[0] > h else cv2.INTER_LINEAR)
