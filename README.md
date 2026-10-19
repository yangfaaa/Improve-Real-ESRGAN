# 基于 Real-ESRGAN 的图像超分辨率改进

在原生 Real-ESRGAN 内核之上，针对其三个工程/算法缺陷做系统性改进，全部为算法策略层面的实现，
不改变原模型结构与预训练权重。

## 三个优化点
| # | 优化点 | 解决的问题 | 实现 |
| 1 | 图像退化程度自适应预处理 | 固定退化建模：干净图被无谓降噪/锐化，重度退化图反而处理不足 | [`sr_improve/adaptive.py`] |
| 2 | 重叠分块高斯加权融合 | 硬切块拼接导致大图边界断层、色差割裂 | [`sr_improve/tiling.py`] |
| 3 | 轻量化画质后处理矫正 | GAN 固有的色偏、过饱和、局部过曝、高频伪纹理 | [`sr_improve/postprocess.py`] |

完整推理链路（见 [`sr_improve/pipeline.py`]）：
```
输入图 → 退化程度判别 → 自适应预处理 → 按等级选超分器 → 重叠分块高斯融合 → 画质后处理矫正 → 输出图
```

## 环境
- conda 环境名 `Real-ESRGAN`，Python 3.8.20
- torch 2.4.1+cu121、torchvision 0.19.1、opencv 5.0.0、numpy 1.24.4
- scikit-image 0.21.0、scipy 1.10.1、lpips 0.1.4（仅客观评测用）、tqdm、Pillow
- `basicsr` 1.4.2 与 `realesrgan` 以 develop 模式安装（`pip install -e`），指向本仓库的
  `BasicSR/` 与 `realesrgan/`

```bash
conda activate Real-ESRGAN
```

**权重需事先放入 `weights/`**：

| 文件 | 用途 |
| `weights/RealESRGAN_x4plus.pth` | GAN 模型 |
| `weights/RealESRNet_x4plus.pth` | 非 GAN / PSNR 导向模型，用于 GAN 权重插值 |
| `weights/dni_cache/` | 深度网络插值结果缓存（自动生成） |

### 常用参数
| 参数 | 说明 |
| `-i/--input` | 输入图片或目录 |
| `-o/--output` | 输出目录，默认 `results/improved` |
| `-t/--tile` | 分块尺寸，默认 512。**图小于该值时不触发分块**，要观察重叠分块融合请调小 |
| `--tile_overlap` | 相邻块重叠像素数，默认 64 |
| `-s/--outscale` | 输出缩放倍数，默认 4 |
| `--force-level` | 强制指定 `mild`/`moderate`/`severe`，绕过判别器 |
| `--report` | 打印每张图的退化分、等级、生效 GAN 权重与耗时 |
| `--overwrite` | 覆盖同名输出（默认跳过已存在文件） |
| `--no-adaptive` / `--no-blend` / `--no-post` | 调试开关，单独关闭某个优化点以定位问题 |

## 项目结构
```
├─ options/               # 配置文件
├─ realesrgan/            # 源码包
└─ realesrgan.egg-info/   # python包数据，
sr_improve/                   优化模块包
├── config.py                 阈值、特征权重、三档策略表、损伤门槛、分块参数
├── adaptive.py               优化：退化分级 + 自适应预处理 + 自适应超分
├── tiling.py                 优化：上下文裁剪 + 重叠高斯加权融合
├── postprocess.py            优化：四步画质矫正
├── image_ops.py              预处理/后处理共用的拆包工具
├── pipeline.py               ImprovedUpscaler 全链路编排
└── thresholds.json           标定产物（删除即回退到默认阈值）
tools/
├── make_degraded_set.py      三档退化样本集（--downscale 造同尺度退化；--clean 产干净 LR 集）
├── calibrate_thresholds.py   统计标定阈值与特征权重，并输出档间分离度验收报告
DIV2K_valid_HR/               DIV2K 验证集 HR 原图（标定集与干净集的来源）
degraded_set_div2k/           三档退化标定集（make_degraded_set.py 的产物）
clean_set_div2k/              干净 LR 参考集（--clean 的产物，供 --clean-dir 标定损伤门槛）
results/                      推理输出
inference_improved.py         优化推理入口
```

## 优化实现要点

### 优化点1：退化程度自适应预处理

- **四维特征**：块方差离散度、Sobel 梯度均值、FFT 高频能量占比、噪声底估计。
  噪声不用"原图减滤波残差"（会把纹理误判成噪声），而用**平坦块方差底**：
  取 **3×3** 块中方差最低的 **0.3%** 开方，对纹理不敏感。块尺寸与分位是实测结论：
  16×16、8×8 的块都会跨到纹理边缘、把纹理方差读成噪声（8×8 时干净图的噪声读数会从 0
  一路铺到 1.0，与中度档重叠）；3×3 + 0.3% 把 mild↔moderate 分离度从 0.53σ 提到 3.33σ。
- **尺度归一**：提取特征前把长边双向归一到 512（缩用 INTER_AREA、放用 INTER_CUBIC）。
  块方差这类统计量是尺度相关的，只缩不放会让小图停在原生尺度上被误读 ——
  实测同一批干净内容，下采样后的 LR 噪声读数比原图高一个数量级。
- **阈值与权重全部数据驱动**：`tools/calibrate_thresholds.py` 用 (p5,p95) 分位定各特征
  归一化区间；权重在**单调**分项上做网格搜索，最大化**最弱的相邻档分离度**（与验收门槛同源），
  非单调的分项线性组合给不出稳定排序、权重置零。
  注意不能按"跨三档的全局配对信噪比"分配 —— 那会把只在单个档界面上有效的特征判死：
  blur 在 mild→moderate 上有 1.26σ，但到 moderate 就饱和，全局信噪比只有 0.93 会被归零，
  于是 mild↔moderate 无人可管（实测仅 0.53σ）。
- **细节缺失与可修复损伤分开决策**（本项目的关键设计）：
  - `detail_score`（由 gradient / hf_ratio 推）→ 决定 GAN 增强权重
  - `damage_score`（由噪声推）→ 决定要不要做降噪等预处理
  一张干净的低清图 detail 很高（细节确实缺）但 damage 为 0（没有噪声），
  它需要的是**更强的 GAN 去生成细节**而不是降噪。把两者加权成一个分数会让这类图被判"重度"
  并施以降噪+锐化。
- **损伤门槛** `DAMAGE_GATE`：损伤低于门槛时预处理与后处理整条链一步不走，
  门槛之上按 `(damage - gate) / (1 - gate)` 线性放开。
  分级器会把细节丰富的干净图误判成退化，加门槛后它们完全不被动。
  该值由标定脚本自动定：`calibrate_thresholds.py --clean-dir <干净图目录> --gate-percentile N`
  取干净样本 `damage` 的 N 分位作门槛，写入 `thresholds.json` 的 `damage_gate` 项。
  **分位不能照搬默认的 p95**：本项目干净集的 damage 从 0 铺到 0.51，p95 会高到挡掉
  68% 的中度图、让中度档名存实亡；实测取 **p75** 时保护 73% 干净图、只牺牲 1% 中度图
  （当前 0.146）。
- **GAN 增强权重真实生效**：Real-ESRGAN 推理期没有这个参数，本方案用深度网络插值
  把 GAN 模型 `RealESRGAN_x4plus` 与非 GAN 模型 `RealESRNet_x4plus` 按 `[1-α, α]`
  逐层混合，α 即 GAN 增强权重，插值结果按 α 缓存。
- **α 由"细节缺失度"逐图插值，不按档位取值**：
  `α = dni_alpha + (1 - dni_alpha) × detail_score`，三档下限 0.25 / 0.50 / 0.80。
  缺细节就拉满 GAN 生成细节，输入本身够锐才落到下限、避免编造假纹理。
  档位量的是"有多脏"，α 该看的是"缺多少细节"，两者不同源 ——
  按档位强行压低 α（轻度恒取 0.25）会拉低 PSNR(峰值信噪比)。
  参数见 `config.py` STRATEGY = {dni_alpha}。


### 优化点2：重叠分块高斯加权融合

用加权累加替代硬拼接：累加图 `acc += 子块输出 × 高斯窗口`，权重图 `wsum += 高斯窗口`，
最终 `输出 = acc / wsum`。重叠区里离块中心更近的那块权重更大，自然形成交叉淡入淡出；
非重叠区只有一块覆盖，结果恰好还原原始输出。
**关键是配合 `tile_pad` 上下文裁剪。** 每块多切一圈真实上下文、推理后再裁回，
否则块边缘本身就是卷积零填充导致的受损输出，融合两份受损输出并不会变好。


### 优化点3：画质后处理矫正

四步 pipeline，每步强度由策略表驱动、可单独关闭，并整体受损伤门槛约束：
1. **亮度自适应矫正** —— 以双三次上采样原图为参照对齐整体亮度，压缩过曝高光
2. **饱和度还原校准** —— 按自身分布 p2/p98 限幅，再按参考图通道均值矫正色偏
3. **高频伪纹理抑制** —— 保边滤波高频残差衰减
4. **整体锐度微调** —— USM非锐化掩膜



## 与原版的关系

- 上游 `realesrgan/` 与 `BasicSR/` 未做任何修改（原版推理入口 `inference_realesrgan.py`
  未随本仓库分发，本仓库只提供 `inference_improved.py`）。
- 重叠分块高斯加权融合通过继承 `RealESRGANer` 并重写 `tile_process()` 实现.
  复用 `pre_process` / `post_process` / `enhance` 的 16bit、灰度、RGBA、outscale 全套逻辑.
