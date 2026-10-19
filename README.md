# 基于 Real-ESRGAN 的图像超分辨率改进

在原生 Real-ESRGAN 内核之上，针对其三个工程/算法缺陷做系统性改进，全部为算法策略层面的实现，
不改变原模型结构与预训练权重。

## 三个优化点
| # | 优化点 | 解决的问题 | 实现 |
| 1 | 图像退化程度自适应预处理 | 固定退化建模：干净图被无谓降噪/锐化，重度退化图反而处理不足 | [`sr_improve/adaptive.py`] |
| 2 | 重叠分块高斯加权融合 | 硬切块拼接导致大图边界断层、色差割裂 | [`sr_improve/tiling.py`] |
| 3 | 轻量化画质后处理矫正 | GAN 固有的色偏、过饱和、局部过曝、高频伪纹理 | [`sr_improve/postprocess.py`] |

## 与原版的关系

- 上游 `realesrgan/` 与 `BasicSR/` 未做任何修改（原版推理入口 `inference_realesrgan.py`
  未随本仓库分发，本仓库只提供 `inference_improved.py`）。
- 重叠分块高斯加权融合通过继承 `RealESRGANer` 并重写 `tile_process()` 实现.
  复用 `pre_process` / `post_process` / `enhance` 的 16bit、灰度、RGBA、outscale 全套逻辑.
