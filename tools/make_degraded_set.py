"""
合成三档退化样本集，供图像退化程度自适应增强的阈值标定与分级自检使用。

  --downscale N  先下采样 N 倍再施加退化，**默认 4**。标定集必须与部署输入同尺度：
  --clean        只输出下采样后的干净图（不施加退化），供 calibrate_thresholds.py
                 的 --clean-dir 标定损伤门槛。与退化集共用同一套下采样逻辑，保证同尺度。
"""

import argparse
import os
import sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEVELS = ('mild', 'moderate', 'severe')


def gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    ksize = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


def add_gaussian_noise(img, sigma, rng):
    if sigma <= 0:
        return img
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def jpeg_compress(img, quality):
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError('JPEG 编码失败')
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


    #对一张 BGR 图施加指定等级的退化。
def degrade(img, level, rng):
    if level == 'mild':
        return jpeg_compress(img, 85)
    if level == 'moderate':
        out = gaussian_blur(img, 1.0)
        out = jpeg_compress(out, 60)
        return add_gaussian_noise(out, 5.0, rng)
    if level == 'severe':
        out = gaussian_blur(img, 3.0)
        out = jpeg_compress(out, 30)
        out = jpeg_compress(out, 30)           # 二次压缩
        return add_gaussian_noise(out, 20.0, rng)
    raise ValueError(f'未知等级: {level}')


def main():
    parser = argparse.ArgumentParser(description='合成三档退化样本集')
    parser.add_argument('-i', '--input', type=str, default='DIV2K_valid_HR/DIV2K_valid_HR', help='干净参考图目录')
    parser.add_argument('-o', '--output', type=str, default='degraded_set_div2k', help='输出根目录')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--downscale', type=int, default=4,
                        help='先按此倍数下采样、再施加退化，默认 4。标定集必须与部署输入同尺度：'
                             '推理时输入是 x4 下采样后的 LR，若在 HR 原生尺寸上造退化，'
                             '退化会在特征提取把长边缩回 512 时被抹平（实测 mild↔moderate '
                             '只剩 0.86σ、准确率 80%；×4 则是 2.95σ、96.7%）。'
                             '传 1 可关掉下采样。')
    parser.add_argument('--clean', action='store_true',
                        help='只输出下采样后的干净图（不退化），供 calibrate_thresholds.py '
                             '的 --clean-dir 标定损伤门槛；输出目录下不再建三档子目录。')
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.webp'))

    if not paths:
        raise SystemExit(f'在 {args.input} 下没找到图片')

    keys = ('clean',) if args.clean else LEVELS
    counts = {lvl: 0 for lvl in keys}
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'跳过（无法读取）: {path}')
            continue
        # 统一成 BGR 三通道：合成数据集只用于标定统计量，alpha 通道无意义
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:                      # 16bit 归一到 8bit
            img = (img.astype(np.float32) / 65535.0 * 255.0).round().astype(np.uint8)

        if args.downscale > 1:
            h, w = img.shape[:2]
            h, w = h - h % args.downscale, w - w % args.downscale
            img = cv2.resize(img[:h, :w], (w // args.downscale, h // args.downscale),
                             interpolation=cv2.INTER_CUBIC)

        stem = os.path.splitext(os.path.basename(path))[0]
        if args.clean:
            os.makedirs(args.output, exist_ok=True)
            cv2.imwrite(os.path.join(args.output, f'{stem}.png'), img)
            counts['clean'] += 1
            continue
        for level in LEVELS:
            out_dir = os.path.join(args.output, level)
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f'{stem}.png')
            cv2.imwrite(save_path, degrade(img, level, rng))
            counts[level] += 1

    print(f'已生成{"干净样本集（未退化）" if args.clean else "退化样本集"}: {args.output}')
    for k in keys:
        print(f'  {k:<9} {counts[k]} 张')


if __name__ == '__main__':
    main()
