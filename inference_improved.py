"""
Real-ESRGAN 优化版推理入口：整合后的完整工程。
    输入图 → 退化程度判别 → 自适应预处理 → 自适应 GAN 权重超分→ 重叠分块高斯加权融合 → 画质后处理矫正 → 输出图
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sr_improve.config import LEVEL_NAMES_CN, LEVELS, TILE_OVERLAP, TILE_SIZE  # noqa: E402
from sr_improve.pipeline import ImprovedUpscaler  # noqa: E402

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
DEFAULT_OUTPUT = os.path.join('results', '')


def list_inputs(input_arg):
    """列出待处理图片。按扩展名过滤，避免 glob('*') 把目录/非图片也卷进来。"""
    if os.path.isfile(input_arg):
        return [input_arg]
    if not os.path.isdir(input_arg):
        raise SystemExit(f'输入路径不存在: {input_arg}')
    return sorted(
        os.path.join(input_arg, f) for f in os.listdir(input_arg)
        if os.path.splitext(f)[1].lower() in IMG_EXTS)


def build_parser():
    parser = argparse.ArgumentParser(description='Real-ESRGAN 优化版推理')
    parser.add_argument('-i', '--input', type=str, default='inputs', help='输入图片或目录')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT, help='输出目录')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='输出缩放倍数')
    parser.add_argument('--suffix', type=str, default='out', help='输出文件名后缀')
    parser.add_argument('-t', '--tile', type=int, default=TILE_SIZE,help='分块尺寸，0 为不分块（图小于该值时不触发分块）')
    parser.add_argument('--tile_overlap', type=int, default=TILE_OVERLAP,help='相邻分块的重叠像素数')
    parser.add_argument('--tile_pad', type=int, default=10, help='分块边缘填充')
    parser.add_argument('--pre_pad', type=int, default=0, help='预处理边缘填充')
    parser.add_argument('--fp32', action='store_true', help='使用 FP32 精度')
    parser.add_argument('--ext', type=str, default='auto', help='输出扩展名，auto 为跟随输入')
    parser.add_argument('-g', '--gpu-id', type=int, default=None, help='GPU 序号')
    parser.add_argument('--weights', type=str, default='weights', help='权重目录')
    parser.add_argument('--force-level', choices=LEVELS, default=None,help='强制指定退化等级，绕过判别器')
    parser.add_argument('--report', action='store_true', help='打印每张图的判别结果与耗时')
    parser.add_argument('--overwrite', action='store_true', help='覆盖同名输出文件')

    debug = parser.add_argument_group('调试处理')
    debug.add_argument('--no-adaptive', action='store_true', help='取消（判别 + 预处理 + GAN 权重）')
    debug.add_argument('--no-blend', action='store_true', help='取消（重叠高斯加权融合）')
    debug.add_argument('--no-post', action='store_true', help='取消（画质后处理矫正）')
    return parser

def print_pipeline_state(upscaler):
    print('优化情况：'
          f'1: 退化自适应={"开" if upscaler.adaptive else "关"}  '
          f'2: 重叠分块融合={"开" if upscaler.upsampler.blend else "关"}  '
          f'3: 后处理矫正={"开" if upscaler.post else "关"}')

    if upscaler.adaptive:
        state = '可用' if upscaler.upsampler.dni_available else '不可用（已降级为单一 GAN 模型）'
        print(f'GAN 增强权重(dni):{state}')
        if upscaler.upsampler.dni_available:
            for level in LEVELS:
                print(f'  {LEVEL_NAMES_CN[level]:<8} GAN α = {upscaler.upsampler.alpha_used(level):.2f}')


def main():
    args = build_parser().parse_args()

    upscaler = ImprovedUpscaler(
        weights_dir=args.weights,           #权重目录
        tile=args.tile,                     #分块推理尺寸
        tile_overlap=args.tile_overlap,     #分块之间重叠像素
        tile_pad=args.tile_pad,             #每个图像分块的边缘填充像素，减少分块边缘伪影
        pre_pad=args.pre_pad,               #预处理边缘填充像素，减少预处理边缘伪影
        half=not args.fp32,                 #32位精度推理
        gpu_id=args.gpu_id,                 #GPU号
        adaptive=not args.no_adaptive,      #自适应模式（退化判别 + 自适应预处理 + 自适应 GAN 权重）
        blend=not args.no_blend,            #重叠分块高斯加权融合
        post=not args.no_post,              #画质后处理矫正
    )
    print_pipeline_state(upscaler)

    os.makedirs(args.output, exist_ok=True)
    paths = list_inputs(args.input)
    if not paths:
        raise SystemExit(f'{args.input} 下没有找到图片')

    level_counts = {level: 0 for level in LEVELS}
    if args.report:
        print('\n' + '-' * 68)
        print(f'{"文件":<18}{"输出尺寸":>12}  {"等级":<9}{"退化分":>8}{"GAN α":>8}{"耗时":>8}')
        print('-' * 68)

    if args.report:
        print('-' * 68)


    done = skipped = failed = 0
    t_start = time.time()

    for path in paths:
        name, ext = os.path.splitext(os.path.basename(path))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'[跳过] 无法解码: {path}')
            skipped += 1
            continue

        # RGBA 必须存 png，否则 alpha 通道丢失
        is_rgba = img.ndim == 3 and img.shape[2] == 4
        out_ext = 'png' if is_rgba else (ext[1:] if args.ext == 'auto' else args.ext)
        fname = f'{name}.{out_ext}' if args.suffix == '' else f'{name}_{args.suffix}.{out_ext}'
        save_path = os.path.join(args.output, fname)
        # 先查重再推理：输出已存在就没必要白跑一遍模型
        if os.path.exists(save_path) and not args.overwrite:
            print(f'[跳过] 已存在: {fname}（加 --overwrite 覆盖）')
            skipped += 1
            continue

        t0 = time.time()
        try:
            output, info = upscaler.enhance(img, outscale=args.outscale, force_level=args.force_level)
        except RuntimeError as err:
            print(f'[失败] {os.path.basename(path)}: {err}')
            print('       若 CUDA 显存不足，请调小 --tile（如 256）')
            failed += 1
            continue
        elapsed = time.time() - t0

        cv2.imwrite(save_path, output)
        done += 1
        if info['level'] is not None:
            level_counts[info['level']] += 1

        if args.report:
            level_cn = LEVEL_NAMES_CN[info['level']] if info['level'] else '不判级'
            score_str = f"{info['score']:.3f}" if info['score'] is not None else '  --  '
            print(f'{os.path.basename(path):<18}{output.shape[1]}x{output.shape[0]:<7}  '
                  f'{level_cn:<9}{score_str:>8}{info["dni_alpha"]:>8.2f}{elapsed:>7.2f}s')

    print(f'\n完成 {done} 张，跳过 {skipped} 张，失败 {failed} 张，'
          f'总耗时 {time.time() - t_start:.1f}s')
    if not args.no_adaptive:
        print('退化等级分布: ' + '  '.join(
            f'{LEVEL_NAMES_CN[level]}={level_counts[level]}' for level in LEVELS))
    print(f'输出目录: {args.output}')


if __name__ == '__main__':
    main()