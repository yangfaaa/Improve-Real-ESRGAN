"""
重叠分块 + 高斯加权融合（消除大图拼接断层）。
原生 RealESRGANer.tile_process() 是"切块 → 独立推理 → 直接切片赋值",
块与块之间没有任何过渡处理，大图超分后会出现边界断层、色差割裂、纹理错位。
用"加权累加"替代硬拼接：
    累加图像 acc  += 子块输出 x 高斯权重窗口
    权重累加图 wsum += 高斯权重窗口
    最终输出 = acc / wsum
中心区域权重接近 1,越靠块边缘权重越小。重叠区里离块中心更近的那一块权重更大，
自然形成"前块边缘低权重 + 后块边缘高权重"的交叉淡入淡出。
非重叠区只有一块覆盖,acc/wsum 恰好还原原始输出，主体画质零损失。

"""
import numpy as np
import torch
from tqdm import tqdm
from realesrgan import RealESRGANer
from .config import TILE_OVERLAP
_WINDOW_EPS = 1e-4      # 窗口下限，防止 wsum 为 0


    #二维高斯权重窗口G(r) = exp(-r²/(2σ²))：中心为 1,向四周平滑递减，下限 eps。
    #sigma 取短边的 1/4:这样块边界处的权重仍有约 0.13 而不是趋近 0
def gaussian_window(height, width, eps=_WINDOW_EPS):
    sigma = max(min(height, width) / 4.0, 1.0)
    yy = np.arange(height, dtype=np.float32) - (height - 1) / 2.0
    xx = np.arange(width, dtype=np.float32) - (width - 1) / 2.0
    gy = np.exp(-(yy ** 2) / (2.0 * sigma ** 2))
    gx = np.exp(-(xx ** 2) / (2.0 * sigma ** 2))
    window = np.outer(gy, gx)
    return np.clip(window / window.max(), eps, None)


    #切块起点：保证首块从 0 开始、末块对齐到边界
def tile_starts(size, tile, stride):
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] != size - tile:
        starts.append(size - tile)      # 收尾对齐，避免漏掉右/下边缘
    return starts


class BlendedRealESRGANer(RealESRGANer):
    """分块推理时用高斯加权融合替代硬拼接。

    Args:
        tile_overlap (int): 相邻块的重叠像素数（原图坐标系，默认 64）。
        其余参数与 RealESRGANer 完全一致。
    """

    def __init__(self, *args, tile_overlap=TILE_OVERLAP, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_overlap = tile_overlap

    def tile_process(self):
        batch, channel, height, width = self.img.shape
        scale = self.scale
        # 按维度分别夹取，否则长方形图会被切成一个多余的方形块网格
        tile_h = min(self.tile_size, height)
        tile_w = min(self.tile_size, width)

        # 单块则走原生单次推理，省掉融合开销与精度损失
        if tile_h >= height and tile_w >= width:
            self.process()
            return

        overlap_h = max(min(self.tile_overlap, tile_h // 2), 1)
        overlap_w = max(min(self.tile_overlap, tile_w // 2), 1)
        ys = tile_starts(height, tile_h, max(tile_h - overlap_h, 1))
        xs = tile_starts(width, tile_w, max(tile_w - overlap_w, 1))

        # 在 float32 上累加，防止掉精度
        acc = torch.zeros((batch, channel, height * scale, width * scale),
                          dtype=torch.float32, device=self.img.device)
        wsum = torch.zeros((batch, 1, height * scale, width * scale),
                           dtype=torch.float32, device=self.img.device)

        pad = max(self.tile_pad, 0)

        for oy in tqdm(ys, desc='\t分块融合', unit='行', leave=False):
            for ox in xs:
                cur_h = min(tile_h, height - oy)
                cur_w = min(tile_w, width - ox)

                # 向外额外多截取pad像素，抵消卷积零填充带来的边界失真
                # 推理完成后裁掉pad区域，只保留拥有真实上下文的有效图像部分
                y0, y1 = max(oy - pad, 0), min(oy + cur_h + pad, height)
                x0, x1 = max(ox - pad, 0), min(ox + cur_w + pad, width)
                tile_in = self.img[:, :, y0:y1, x0:x1]

                with torch.no_grad():
                    out_tile = self.model(tile_in).float()

                top, left = (oy - y0) * scale, (ox - x0) * scale
                out_kept = out_tile[:, :, top:top + cur_h * scale, left:left + cur_w * scale]

                out_h, out_w = cur_h * scale, cur_w * scale
                window = torch.from_numpy(gaussian_window(out_h, out_w)).to(
                    device=self.img.device, dtype=torch.float32).view(1, 1, out_h, out_w)

                acc[:, :, oy * scale:(oy + cur_h) * scale,
                    ox * scale:(ox + cur_w) * scale] += out_kept * window
                wsum[:, :, oy * scale:(oy + cur_h) * scale,
                     ox * scale:(ox + cur_w) * scale] += window

        self.output = acc / wsum.clamp_min(_WINDOW_EPS)
