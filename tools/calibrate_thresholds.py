"""
三档退化集标定的分级阈值与特征权重。
  1. 特征归一化区间：每个特征的 (lower, higher) 取全体样本的 (5%, 95%) 分位，
     让四个分项都落在 [0,1] 且铺满整个区间。
  2. 特征权重：在单调分项上做权重网格搜索，最大化**最弱的相邻档分离度**。
  3. 分级切分点：同时给出"准确率最高"与"干净图不越界"两种切法，打印对比。
  4. 验证：单调性、相邻档分离度（间距/σ）、各分项内容比，逐项标出达标与否。

    内容离散 C = mean(std_mild, std_mod, std_severe)  同一退化等级图像内容造成的特征抖动
    退化位移 D = (mean_severe - mean_mild) / 2        退化程度带来的总体特征偏移
    配对信噪比 SNR = D / C                            全局口径，只作诊断，不再决定权重
    界面分离度 S   = |mean_b - mean_a| / 合并档内 std   相邻两档的可分性，权重的搜索目标

结果写入 sr_improve/thresholds.json,推理时自动加载（优先级高于 config.py 的默认值）。

"""
import argparse
import os
import sys
import cv2
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sr_improve.adaptive import damage_score, extract_features, score_terms  # noqa: E402
from sr_improve.config import DEFAULT_DAMAGE_GATE, LEVELS, load_thresholds, save_calibration  # noqa: E402


# 参与标定的特征；把特征名映射到 score_terms() 输出分项名。
FEATURE_KEYS = ('gradient', 'noise', 'hf_ratio', 'var_dispersion')      #原始特征
TERMS_OF = {'gradient': 'blur', 'noise': 'noise', 'hf_ratio': 'hf', 'var_dispersion': 'var'}    #名称映射表
TERM_KEYS = ('blur', 'noise', 'hf', 'var')       # 简写特征
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')   #后缀特征

# 验证门槛
SNR_MIN = 1.0          # 全局配对信噪比下限：只用于打印"哪些分项有全局区分力"，不再决定权重
SIGMA_MIN = 3.0        # 相邻档分离度下限（类间距 / 合并档内 std）
WEIGHT_STEP = 0.05     # 权重网格搜索步长


    #遍历degraded_set提取blur、noise、hf、var四项特征，记录退化标签 level、原图名stems
def collect(root):
    feats_list, labels, stems = [], [], []
    for level in LEVELS:
        level_dir = os.path.join(root, level)
        if not os.path.isdir(level_dir):
            raise SystemExit(f'缺少目录 {level_dir}，请先运行 tools/make_degraded_set.py')
        names = sorted(f for f in os.listdir(level_dir) if os.path.splitext(f)[1].lower() in IMG_EXTS)
        for name in names:
            img = cv2.imread(os.path.join(level_dir, name), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            feats_list.append(extract_features(img))
            labels.append(level)
            stems.append(os.path.splitext(name)[0])
    if not feats_list:
        raise SystemExit(f'{root} 下没读到任何图片')
    return feats_list, labels, stems


    #每个特征的归一化区间取全体样本的 (5%, 95%)，且 higher > lower
def calibrate_ranges(feats_list, low_pct=5.0, high_pct=95.0):
    ranges = {}
    for key in FEATURE_KEYS:
        values = np.array([f[key] for f in feats_list], dtype=np.float64)
        lower = float(np.percentile(values, low_pct))
        higher = float(np.percentile(values, high_pct))
        if higher <= lower:               # 该特征在本数据集上无区分度，退化为微小区间
            higher = lower + max(abs(lower) * 0.1, 1e-4)
        ranges[key] = (lower, higher)
    return ranges


    #配对保留三档齐全的源图样本。
def pair_by_source(terms_list, labels, stems):
    by_src = {}
    for terms, level, stem in zip(terms_list, labels, stems):
        by_src.setdefault(stem, {})[level] = terms
    usable = sorted(s for s, d in by_src.items() if all(l in d for l in LEVELS))

    pairs = {}  #创建空字典
    for term in TERM_KEYS:
        pairs[term] = {
            level: np.array([by_src[s][level][term] for s in usable], dtype=np.float64)
            for level in LEVELS
        }
    return pairs, usable, len(by_src) - len(usable)


    #相邻两档的分离度 = 类间距 / 合并档内 std，与验收报告同一口径
def boundary_sep(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    gap = abs(float(np.mean(b) - np.mean(a)))
    sd = float(np.sqrt((np.var(a) + np.var(b)) / 2.0))
    return gap / sd if sd > 1e-9 else 0.0


def term_diagnostics(pairs, term):
    """单个分项的诊断量。返回 dict：
    within 内容离散 / between 退化位移 / snr 全局配对信噪比 /
    seps 各相邻界面分离度 / min_sep 其中最弱的一个 / monotonic 是否单调。
    只有 snr 与 min_sep 的区别值得强调：snr 要求"跨三档整体迁移"，会把
    到某档就饱和的特征（blur）判死；min_sep 只看最差的那个界面，正是要治的病。
    """
    mat = np.array([pairs[term][level] for level in LEVELS])   # (档数, 源图数)
    means = mat.mean(axis=1)
    within = float(np.mean(mat.std(axis=1)))                   # 内容抖动
    between = float((mat[-1].mean() - mat[0].mean()) / (len(LEVELS) - 1))   # 每档平均位移
    monotonic = bool(np.all(np.diff(means) >= 0))
    snr = between / within if within > 1e-9 else 0.0
    seps = [boundary_sep(mat[i], mat[i + 1]) for i in range(len(LEVELS) - 1)]
    return {
        'within': within, 'between': between, 'snr': snr,
        'seps': seps, 'min_sep': min(seps) if seps else 0.0,
        'monotonic': monotonic,
    }


def weight_grid(n, step=WEIGHT_STEP):
    """生成 n 个分项上权重和为 1 的网格组合（非负、步长 step）。"""
    unit = int(round(1.0 / step))
    out = []

    def rec(prefix, left):
        if len(prefix) == n - 1:
            out.append(prefix + [left])
            return
        for v in range(left + 1):
            rec(prefix + [v], left - v)

    rec([], unit)
    return [np.asarray(c, dtype=np.float64) / unit for c in out]


def calibrate_weights(terms_list, labels, stems):
    """分配特征权重，返回 (weights, diag, pairs, usable)。

    准则：在**单调**的分项上搜索权重组合，使最弱的相邻档分离度尽量大 ——
    与验收门槛（每对相邻档都要 ≥3σ）同源，而不是要求每个特征自己跨三档单调。
    非单调分项的线性组合给不出稳定排序，直接权重置 0，不参与搜索。
    """
    pairs, usable, _ = pair_by_source(terms_list, labels, stems)
    diag = {term: term_diagnostics(pairs, term) for term in TERM_KEYS}

    mono = [t for t in TERM_KEYS if diag[t]['monotonic']]
    if not mono:                       # 全都不单调：退回最弱界面最大的那个单分项
        mono = [max(TERM_KEYS, key=lambda t: diag[t]['min_sep'])]
    stack = {t: np.array([pairs[t][level] for level in LEVELS]) for t in mono}

    best = None
    for wv in weight_grid(len(mono)):
        score = sum(wi * stack[t] for wi, t in zip(wv, mono))
        key = min(boundary_sep(score[i], score[i + 1]) for i in range(len(LEVELS) - 1))
        if best is None or key > best[0]:
            best = (key, wv)

    weights = {t: 0.0 for t in TERM_KEYS}
    for wi, t in zip(best[1], mono):
        weights[t] = float(wi)
    return weights, diag, pairs, usable


    #按权重把四个分项折成综合退化分
def combine(terms, weights):
    total = sum(weights.values())
    return sum(weights[k] * terms[k] for k in weights) / total

    #在综合退化分上网格搜索两个切点，最大化三分类准确率
    #numpy广播，穷举区间候选阈值 计算准确率
def search_cut_points(scores, labels, grid=200):
    level_index = {lvl: i for i, lvl in enumerate(LEVELS)}
    y = np.array([level_index[l] for l in labels])
    cand = np.linspace(0.0, 1.0, grid)
    scores_3d = scores[None, None, :]
    c1 = cand[:, None, None]
    c2 = cand[None, :, None]
    pred = np.where(scores_3d < c1, 0, np.where(scores_3d < c2, 1, 2))
    acc = (pred == y[None, None, :]).mean(axis=2)
    valid = cand[None, :] >= cand[:, None]
    acc = np.where(valid, acc, -1.0)      # 要求 c1 <= c2
    i, j = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return float(cand[i]), float(cand[j]), float(acc[i, j])


def cut_points_clean_safe(scores, labels, margin=0.02):
    """稳健方式计算退化分级切点，避免样本过拟合。
    c1:轻度组样本最大退化分 + 安全余量,保证轻度样本基本落在c1以下
    c2:合并中度、重度样本,取中位数；满足 c1 < c2 ≤ 1.0
    """
    groups = {lvl: np.array([s for s, l in zip(scores, labels) if l == lvl]) for lvl in LEVELS}
    c1 = float(groups[LEVELS[0]].max()) + margin
    tail = np.concatenate([groups[LEVELS[1]], groups[LEVELS[2]]])
    c2 = float(np.clip(np.median(tail), c1 + 1e-3, 1.0))
    return c1, c2


def predict(scores, c1, c2):
    return np.where(scores < c1, 0, np.where(scores < c2, 1, 2))


    #打印混淆矩阵计算每一类召回率，评估退化分级效果。
def print_confusion(pred, labels):
    level_index = {lvl: i for i, lvl in enumerate(LEVELS)}
    matrix = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
    for p, l in zip(pred, labels):
        matrix[level_index[l], int(p)] += 1
    print('\n混淆矩阵（行=真实, 列=预测）:')
    print(f'{"":<12}' + ''.join(f'{lvl:>10}' for lvl in LEVELS))
    for i, lvl in enumerate(LEVELS):
        print(f'{lvl:<12}' + ''.join(f'{matrix[i, j]:>10}' for j in range(len(LEVELS))))
    per_level = matrix.diagonal() / np.maximum(matrix.sum(axis=1), 1)
    print('各等级召回率: ' + '  '.join(f'{lvl}={r * 100:.0f}%' for lvl, r in zip(LEVELS, per_level)))
    return per_level


    #打印相邻等级的分离度 = 类间距 / 合并档内 std，并标出是否达标
def report_separation(scores, labels, sigma_min=SIGMA_MIN):
    groups = {lvl: np.array([s for s, l in zip(scores, labels) if l == lvl]) for lvl in LEVELS}
    print(f'\n档间分离度（>= {sigma_min:.0f}σ 才认为可定阈值）:')
    print(f'{"档位对":<24}{"类间距":>10}{"档内std":>10}{"间距/σ":>10}')
    rows = []
    for a, b in zip(LEVELS[:-1], LEVELS[1:]):
        gap = abs(float(groups[b].mean() - groups[a].mean()))
        sd = float(np.sqrt((groups[a].var() + groups[b].var()) / 2.0))
        ratio = boundary_sep(groups[a], groups[b])
        rows.append((a, b, gap, sd, ratio))
        flag = 'OK' if ratio >= sigma_min else '** 不达标 **'
        print(f'{a + " - " + b:<24}{gap:>10.4f}{sd:>10.4f}{ratio:>10.2f}   {flag}')
    return rows


def main():
    parser = argparse.ArgumentParser(description='标定的退化分级阈值与特征权重')
    parser.add_argument('-d', '--dataset', type=str, default='degraded_set_div2k', help='退化样本集根目录')
    parser.add_argument('--clean-dir', type=str, default=None,
                        help='干净图目录（如 DIV2K 的下采样 LR),仅用于标定 DAMAGE_GATE')
    parser.add_argument('--gate-percentile', type=float, default=95.0,
                        help='DAMAGE_GATE 取干净样本 damage 的这个分位（默认 95)')
    parser.add_argument('--dry-run', action='store_true', help='只打印结果，不写 thresholds.json')
    args = parser.parse_args()

    feats_list, labels, stems = collect(args.dataset)
    print(f'样本数: {len(feats_list)}  ' + '  '.join(f'{lvl}={labels.count(lvl)}' for lvl in LEVELS))

    # ---- 1. 特征分布概览：确认每个特征能把三档分开 ----
    print('\n各特征按等级的分组均值:')
    print(f'{"特征":<16}' + ''.join(f'{lvl:>12}' for lvl in LEVELS))
    for key in FEATURE_KEYS:
        row = f'{key:<16}'
        means = []
        for lvl in LEVELS:
            vals = [f[key] for f, l in zip(feats_list, labels) if l == lvl]
            means.append(float(np.mean(vals)))
            row += f'{means[-1]:>12.4f}'
        print(row + ('   （注意：原始特征方向不一，单调性以分项为准）' if key == FEATURE_KEYS[0] else ''))

    # ---- 2. 归一化区间 ----
    th = load_thresholds()
    for key, (lower, higher) in calibrate_ranges(feats_list).items():
        th[key] = (lower, higher)
        print(f'区间 {key:<16} ({lower:.4f}, {higher:.4f})')

    # ---- 3. 特征权重（单调分项上最大化最弱界面分离度） ----
    terms_list = [score_terms(f, th) for f in feats_list]
    weights, diag, _, usable = calibrate_weights(terms_list, labels, stems)
    print(f'\n各分项的配对诊断（{len(usable)} 张源图参与配对）:')
    print(f'{"分项":<8}{"内容离散":>10}{"退化位移":>10}{"信噪比":>9}{"最弱界面":>10}'
          f'{"单调":>7}{"旧rho":>9}{"权重":>9}')
    level_index = {lvl: i for i, lvl in enumerate(LEVELS)}
    y = np.array([level_index[l] for l in labels], dtype=np.float64)
    for term in TERM_KEYS:
        d = diag[term]
        x = np.array([t[term] for t in terms_list], dtype=np.float64)
        rho = 0.0 if np.allclose(x, x[0]) else float(spearmanr(x, y).correlation or 0.0)
        flag = '' if d['monotonic'] else '  ← 非单调，权重归零'
        print(f'{term:<8}{d["within"]:>10.4f}{d["between"]:>10.4f}{d["snr"]:>9.2f}'
              f'{d["min_sep"]:>10.2f}{("是" if d["monotonic"] else "否"):>7}'
              f'{rho:>9.3f}{weights[term]:>9.3f}{flag}')

    # ---- 4. 切分点：两种切法对比 ----
    scores = np.array([combine(t, weights) for t in terms_list])
    a1, a2, aacc = search_cut_points(scores, labels)
    k1, k2 = cut_points_clean_safe(scores, labels)
    kacc = float((predict(scores, k1, k2) == np.array([level_index[l] for l in labels])).mean())

    print('\n切分点对比:')
    print(f'  准确率最高 : ({a1:.4f}, {a2:.4f})   准确率 {aacc * 100:.1f}%')
    print(f'  干净图优先 : ({k1:.4f}, {k2:.4f})   准确率 {kacc * 100:.1f}%')
    th['score'] = (round(a1, 4), round(a2, 4))      # 写盘仍用准确率最优，避免静默改变既有实验

    print_confusion(predict(scores, a1, a2), labels)
    rows = report_separation(scores, labels, SIGMA_MIN)

    # ---- 5. 可用性结论 ----
    weak = [f'{a}-{b}' for a, b, _, _, r in rows if r < SIGMA_MIN]
    weighted = [t for t in TERM_KEYS if weights[t] > 0]
    print(f'  参与加权的分项: '
          + (', '.join(f'{t}={weights[t]:.2f}' for t in weighted) if weighted else '无（兜底单分项）')
          + f'  / 共 {len(TERM_KEYS)} 项')
    if weak:
        print(f'  不可分档位对: {", ".join(weak)}')
        print(f'  → 这几档的重叠是结构性的，换任何切分点都无法消除；'
              f'要根治须改特征')
    else:
        print('  三档两两可分，标定可用。')

    # ---- 6. 损伤门槛标定 ----
    gate = None
    if args.clean_dir:
        if not os.path.isdir(args.clean_dir):
            raise SystemExit(f'--clean-dir 不是目录: {args.clean_dir}')
        clean = sorted(os.path.join(args.clean_dir, f) for f in os.listdir(args.clean_dir)
                       if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if not clean:
            raise SystemExit(f'{args.clean_dir} 下没读到图片')
        # 必须传 th：否则 damage_score 回退到模块级 THRESHOLDS（磁盘上的旧区间），
        # 算出的门槛与本次将要写入的新区间不是一套（实测同一批干净图差近一倍）。
        dmg = np.array([damage_score(extract_features(cv2.imread(p, cv2.IMREAD_UNCHANGED)), th)
                        for p in clean])
        gate = float(np.percentile(dmg, args.gate_percentile))
        print('\n损伤门槛 DAMAGE_GATE 标定（干净图上的 damage 分布）:')
        print(f'  样本 {len(dmg)} 张   min {dmg.min():.3f}  p50 {np.percentile(dmg, 50):.3f}  '
              f'p90 {np.percentile(dmg, 90):.3f}  p95 {np.percentile(dmg, 95):.3f}  max {dmg.max():.3f}')
        print(f'  DAMAGE_GATE = p{args.gate_percentile:g} = {gate:.3f}'
              f'   （先验默认 {DEFAULT_DAMAGE_GATE}；干净图误开闸约 {100 - args.gate_percentile:g}%）')
        if dmg.max() >= 0.99:
            print('     警告：有干净样本 damage 触顶 1.000。这类图门槛保护不了')
            print('     说明 damage 特征本身在内容丰富图上不可靠 —— 应当先修特征，而不是继续调门槛')

    # ---- 7. 写入 ----
    if args.dry_run:
        print('\n--dry-run：未写入 thresholds.json')
    else:
        print(f'\n已写入 {save_calibration(th, weights, gate)}')
        print('后续推理会自动加载；删除该文件即回退到默认值。')


if __name__ == '__main__':
    main()
