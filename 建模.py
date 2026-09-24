import argparse
import math
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import numpy as np


AVOGADRO = 6.02214076e23  # mol^-1

# 氧化物数据库：化学计量 + 摩尔质量(g/mol)
OXIDES = {
    # 网络形成体
    "SiO2":  {"atoms": {"Si": 1, "O": 2}, "mw": 60.0843},
    "B2O3":  {"atoms": {"B":  2, "O": 3}, "mw": 69.6202},
    "P2O5":  {"atoms": {"P":  2, "O": 5}, "mw": 141.9425},
    "GeO2":  {"atoms": {"Ge": 1, "O": 2}, "mw": 104.628},
    # 碱金属修饰体
    "Li2O":  {"atoms": {"Li": 2, "O": 1}, "mw": 29.8814},
    "Na2O":  {"atoms": {"Na": 2, "O": 1}, "mw": 61.9786},
    "K2O":   {"atoms": {"K":  2, "O": 1}, "mw": 94.1956},
    # 碱土金属修饰体
    "MgO":   {"atoms": {"Mg": 1, "O": 1}, "mw": 40.304},
    "CaO":   {"atoms": {"Ca": 1, "O": 1}, "mw": 56.077},
    "BaO":   {"atoms": {"Ba": 1, "O": 1}, "mw": 153.326},
    # 中间体
    "ZnO":   {"atoms": {"Zn": 1, "O": 1}, "mw": 81.38},
    "Al2O3": {"atoms": {"Al": 2, "O": 3}, "mw": 101.9601},
    "TiO2":  {"atoms": {"Ti": 1, "O": 2}, "mw": 79.865},
    "ZrO2":  {"atoms": {"Zr": 1, "O": 2}, "mw": 123.222},
    # 稀土
    "La2O3": {"atoms": {"La": 2, "O": 3}, "mw": 325.8092},
    "Y2O3":  {"atoms": {"Y":  2, "O": 3}, "mw": 225.8086},
    # 高价掺杂
    "Nb2O5": {"atoms": {"Nb": 2, "O": 5}, "mw": 265.81},
    "Ta2O5": {"atoms": {"Ta": 2, "O": 5}, "mw": 441.8908},
}

# 默认“反重叠半径”，不是键长
BASE_RADII = {
    "O":  1.05,
    "B":  1.20,
    "Si": 1.30,
    "P":  1.10,
    "Ge": 1.30,
    "Li": 1.20,
    "Na": 1.25,
    "K":  1.55,
    "Mg": 1.00,
    "Ca": 1.30,
    "Ba": 1.65,
    "Zn": 1.35,
    "Al": 1.25,
    "Ti": 1.40,
    "Zr": 1.55,
    "Nb": 1.60,
    "Ta": 1.60,
    "La": 1.85,
    "Y":  1.70,
}

# 关键对的人工覆盖：只是最小允许距离，不是目标键长
PAIR_MIN_DIST = {
    # O-O 和网络形成体-O
    tuple(sorted(("O",  "O"))):  1.80,
    tuple(sorted(("B",  "O"))):  1.20,
    tuple(sorted(("Si", "O"))):  1.35,
    tuple(sorted(("P",  "O"))):  1.25,
    tuple(sorted(("Ge", "O"))):  1.40,
    # 碱金属-O
    tuple(sorted(("Li", "O"))):  1.45,
    tuple(sorted(("Na", "O"))):  1.55,
    tuple(sorted(("K",  "O"))):  1.75,
    # 碱土金属-O
    tuple(sorted(("Mg", "O"))):  1.45,
    tuple(sorted(("Ca", "O"))):  1.55,
    tuple(sorted(("Ba", "O"))):  1.80,
    # 中间体-O
    tuple(sorted(("Zn", "O"))):  1.55,
    tuple(sorted(("Al", "O"))):  1.40,
    tuple(sorted(("Ti", "O"))):  1.50,
    tuple(sorted(("Zr", "O"))):  1.55,
    # 稀土-O
    tuple(sorted(("La", "O"))):  2.00,
    tuple(sorted(("Y",  "O"))):  1.75,
    # 高价掺杂-O
    tuple(sorted(("Nb", "O"))):  1.65,
    tuple(sorted(("Ta", "O"))):  1.60,
    # 阳离子-阳离子自对
    tuple(sorted(("Li", "Li"))): 1.80,
    tuple(sorted(("Na", "Na"))): 2.00,
    tuple(sorted(("K",  "K"))):  2.60,
    tuple(sorted(("Mg", "Mg"))): 1.90,
    tuple(sorted(("Ca", "Ca"))): 2.20,
    tuple(sorted(("Ba", "Ba"))): 2.70,
    tuple(sorted(("Si", "Si"))): 2.40,
    tuple(sorted(("B",  "B"))):  2.00,
    tuple(sorted(("P",  "P"))):  2.20,
    tuple(sorted(("Ge", "Ge"))): 2.40,
    tuple(sorted(("Zn", "Zn"))): 2.50,
    tuple(sorted(("Al", "Al"))): 2.30,
    tuple(sorted(("Ti", "Ti"))): 2.40,
    tuple(sorted(("Zr", "Zr"))): 2.60,
    tuple(sorted(("Nb", "Nb"))): 2.80,
    tuple(sorted(("Ta", "Ta"))): 2.60,
    tuple(sorted(("La", "La"))): 3.20,
    tuple(sorted(("Y",  "Y"))):  2.80,
}

# 放置顺序：先少量重元素/高价元素，再网络形成体，再 O，最后 Li
PLACEMENT_PRIORITY = {
    # 稀土：最重的先放
    "La": 0,
    "Y":  0,
    # 高价掺杂
    "Nb": 1,
    "Ta": 1,
    # 重中间体
    "Zr": 2,
    # 中间体
    "Zn": 3,
    "Ti": 3,
    # 网络形成体
    "Si": 4,
    "Ge": 4,
    "Al": 4,
    "B":  5,
    "P":  5,
    # 氧：在网络形成体之后，修饰体之前
    "O":  6,
    # 碱土金属修饰体
    "Mg": 7,
    "Ca": 7,
    "Ba": 7,
    # 碱金属修饰体：最轻最后放
    "Li": 8,
    "Na": 8,
    "K":  8,
}


def atoms_per_formula_unit(oxide: str) -> int:
    return sum(OXIDES[oxide]["atoms"].values())


def min_dist(sym1: str, sym2: str) -> float:
    key = tuple(sorted((sym1, sym2)))
    if key in PAIR_MIN_DIST:
        return PAIR_MIN_DIST[key]
    return 0.82 * (BASE_RADII[sym1] + BASE_RADII[sym2])


def _pair_table(symbols: List[str]) -> Tuple[np.ndarray, List[str]]:
    """按体系中实际出现的元素构建最小允许距离表 (K, K) 及元素列表。"""
    uniq = sorted(set(symbols))
    m = len(uniq)
    tab = np.empty((m, m), dtype=float)
    for a in range(m):
        for b in range(a, m):
            d = min_dist(uniq[a], uniq[b])
            tab[a, b] = d
            tab[b, a] = d
    return tab, uniq


def _symbol_index_array(symbols: List[str], uniq: List[str]) -> np.ndarray:
    """每个原子的元素在 uniq 表中的索引。"""
    idx = {s: i for i, s in enumerate(uniq)}
    return np.fromiter((idx[s] for s in symbols), dtype=np.intp, count=len(symbols))


def choose_formula_units(
    mol_pct: Dict[str, float],
    target_atoms: int,
    search_points: int = 4000
) -> Dict[str, int]:
    """
    根据氧化物摩尔百分比，搜索最接近 target_atoms 的整数公式单元数（向量化版）。
    """
    mol_pct = {k: v for k, v in mol_pct.items() if v > 0}
    for oxide in mol_pct:
        if oxide not in OXIDES:
            raise ValueError(f"未知氧化物: {oxide}")

    total_mol = sum(mol_pct.values())
    mol_frac = {k: v / total_mol for k, v in mol_pct.items()}

    avg_atoms_per_oxide = sum(
        mol_frac[k] * atoms_per_formula_unit(k) for k in mol_frac
    )
    s0 = target_atoms / avg_atoms_per_oxide

    s_arr = np.linspace(max(1.0, 0.5 * s0), 1.5 * s0, search_points)

    oxides = list(mol_frac.keys())
    fracs = np.array([mol_frac[o] for o in oxides], dtype=float)
    atoms_per = np.array([atoms_per_formula_unit(o) for o in oxides], dtype=float)

    # (search_points, K)：np.rint 与 int(round(...)) 同为银行家舍入，结果一致
    fu = np.maximum(1, np.rint(fracs[None, :] * s_arr[:, None])).astype(np.int64)
    total_atoms = (fu * atoms_per[None, :]).sum(axis=1)  # 乘加求和，避免触发 BLAS/MKL
    fu_total = fu.sum(axis=1)
    comp_err = np.abs(fu / fu_total[:, None] - fracs[None, :]).sum(axis=1)
    score = np.abs(total_atoms - target_atoms) + 30.0 * comp_err

    best = int(np.argmin(score))
    return {ox: int(fu[best, k]) for k, ox in enumerate(oxides)}


def formula_units_to_atom_counts(formula_units: Dict[str, int]) -> Dict[str, int]:
    counts = Counter()
    for oxide, n_fu in formula_units.items():
        for element, n_atom in OXIDES[oxide]["atoms"].items():
            counts[element] += n_fu * n_atom
    return dict(counts)


def formula_units_to_mass_gram(formula_units: Dict[str, int]) -> float:
    """
    返回体系总质量，单位 g
    """
    total_mass = 0.0
    for oxide, n_fu in formula_units.items():
        total_mass += n_fu * OXIDES[oxide]["mw"] / AVOGADRO
    return total_mass


def cubic_box_length_from_density(formula_units: Dict[str, int], density_g_cm3: float) -> float:
    """
    根据密度计算立方盒边长，单位 Å
    """
    if density_g_cm3 <= 0:
        raise ValueError("密度必须大于 0")
    mass_g = formula_units_to_mass_gram(formula_units)
    volume_cm3 = mass_g / density_g_cm3
    volume_a3 = volume_cm3 * 1.0e24
    return volume_a3 ** (1.0 / 3.0)


def expand_symbols(atom_counts: Dict[str, int]) -> List[str]:
    symbols = []
    for sym, n in atom_counts.items():
        symbols.extend([sym] * n)
    symbols.sort(key=lambda s: PLACEMENT_PRIORITY.get(s, 99))
    return symbols


def pbc_distance(r1: np.ndarray, r2: np.ndarray, box: float) -> float:
    dr = r1 - r2
    dr -= box * np.round(dr / box)
    return float(np.linalg.norm(dr))


def place_atoms_randomly(
    symbols: List[str],
    box: float,
    seed: int = 42,
    max_trials_per_atom: int = 20000,
    best_of: int = 150,
) -> np.ndarray:
    """
    随机放置 + 最小距离约束（cell-list 邻域批量版）。

    放置语义与逐对循环版一致：先在 best_of 次尝试中挑"离已放置原子最近距离"
    最大的合法位置，失败再回退到"第一个合法即接受"。区别在于邻域查询用
    cell list（cell 边长 = 最大最小距离，27 邻域即完整覆盖），每批 trial 与其
    邻域的距离用 numpy 广播一次算完，数万原子也可在数十秒内完成。

    注意：随机数按批消耗，同 seed 的具体坐标与逐对循环版不逐位相同，
    但结构的统计质量与可复现性（同 seed 同结果）不变。
    """
    n = len(symbols)
    if n == 0:
        return np.empty((0, 3), dtype=float)

    rng = np.random.default_rng(seed)
    tab, uniq = _pair_table(symbols)
    sidx = _symbol_index_array(symbols, uniq)
    uids = np.unique(sidx)
    # 邻域半径 = 现有元素对的最大最小距离（cell 边长），保证 27 邻域覆盖完整
    cell = max(float(tab[np.ix_(uids, uids)].max()), 1e-6)
    nc = max(1, int(math.floor(box / cell)))
    positions = np.empty((n, 3), dtype=float)

    cells: Dict[int, List[int]] = {}
    nb_cache: Dict[int, Tuple[object, object]] = {}

    def _decode(key: int) -> Tuple[int, int, int]:
        cx = key // (nc * nc)
        rem = key % (nc * nc)
        return cx, rem // nc, rem % nc

    def _encode(cx: int, cy: int, cz: int) -> int:
        return (cx * nc + cy) * nc + cz

    def _neighbors(key: int):
        got = nb_cache.get(key)
        if got is not None:
            return got
        cx, cy, cz = _decode(key)
        mem: List[int] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    mem.extend(cells.get(_encode((cx + dx) % nc, (cy + dy) % nc, (cz + dz) % nc), ()))
        if mem:
            mem_arr = np.fromiter(mem, dtype=np.intp, count=len(mem))
            got = (positions[mem_arr], sidx[mem_arr])
        else:
            got = (None, None)
        nb_cache[key] = got
        return got

    def _check_trials(trials: np.ndarray, si: int, first_ok_only: bool):
        """一批 trials 的邻域检查（全向量化：一次 ragged 展开 + 单批距离计算）。

        返回 (best_pos, best_mind)：first_ok_only=False 时返回合法 trials 中
        mindist 最大者；True 时返回第一个合法 trial。
        """
        m = len(trials)
        tcell = np.floor(trials / cell).astype(np.int64) % nc
        tkey = (tcell[:, 0] * nc + tcell[:, 1]) * nc + tcell[:, 2]
        uniq_keys, inv = np.unique(tkey, return_inverse=True)
        n_groups = len(uniq_keys)
        order = np.argsort(inv, kind="stable")  # trials 按 cell 分组排序
        bounds = np.searchsorted(inv[order], np.arange(n_groups + 1))

        nb_pos_list: List[np.ndarray] = []
        nb_sidx_list: List[np.ndarray] = []
        k_arr = np.empty(n_groups, dtype=np.int64)
        first_empty = -1
        for g in range(n_groups):
            nb_pos, nb_sidx = _neighbors(int(uniq_keys[g]))
            if nb_pos is None:
                k_arr[g] = 0
                if first_empty < 0:
                    first_empty = g
            else:
                nb_pos_list.append(nb_pos)
                nb_sidx_list.append(nb_sidx)
                k_arr[g] = len(nb_pos)

        # 存在空邻域组：其 trials 必合法且 mindist=inf，取该组第一个 trial
        if first_empty >= 0:
            return trials[order[bounds[first_empty]]].copy(), float("inf")

        t_arr = np.bincount(inv, minlength=n_groups)          # 每组 trial 数
        p_arr = t_arr * k_arr                                  # 每组 (trial, 邻居) 对数
        total = int(p_arr.sum())
        if total == 0:
            return trials[0].copy(), float("inf")

        nb_all = np.concatenate(nb_pos_list, axis=0)
        nb_sidx_all = np.concatenate(nb_sidx_list, axis=0)
        nb_start = np.concatenate([[0], np.cumsum(k_arr)[:-1]])

        # (trial, 邻居) 对的 ragged 展开：order 是按 cell 分组排序后的 trial 序列，
        # 重复计数必须与 order 对齐（k_sorted），否则 trial 与邻居集合错配
        k_sorted = k_arr[inv[order]]                           # (m,) 与 order 对齐
        trial_of_pair = np.repeat(order, k_sorted)             # (total,)
        g_of_pair = np.repeat(inv[order], k_sorted)
        ordinal = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(p_arr) - p_arr, p_arr)
        nb_idx = nb_start[g_of_pair] + ordinal % k_arr[g_of_pair]

        diff = trials[trial_of_pair] - nb_all[nb_idx]
        diff -= box * np.round(diff / box)
        d2 = np.einsum("pi,pi->p", diff, diff)
        illegal = d2 < tab[si, nb_sidx_all[nb_idx]] ** 2

        ill_per_trial = np.bincount(trial_of_pair, weights=illegal, minlength=m)
        legal = ill_per_trial == 0                             # 按 trial 原始索引
        if not legal.any():
            return None, -1.0
        # 对合法 trial 全部对均合法，per-trial 最小 d2 即 mindist^2（按 order 排列）
        d2f = np.where(illegal, np.inf, d2)
        starts = np.cumsum(k_sorted) - k_sorted
        mind_sorted = np.sqrt(np.minimum.reduceat(d2f, starts))
        legal_sorted = legal[order]
        if first_ok_only:
            js = int(np.argmax(legal_sorted))
        else:
            cand = np.nonzero(legal_sorted)[0]
            js = int(cand[np.argmax(mind_sorted[cand])])
        j = int(order[js])
        return trials[j].copy(), float(mind_sorted[js])

    for i in range(n):
        si = int(sidx[i])
        # 阶段 1：best_of 次尝试选最优合法位置
        trials = rng.random((best_of, 3)) * box
        best_pos, _best_mind = _check_trials(trials, si, first_ok_only=False)
        # 阶段 2：回退"第一个合法即接受"
        if best_pos is None:
            remaining = max_trials_per_atom
            while best_pos is None and remaining > 0:
                b = min(256, remaining)
                remaining -= b
                trials = rng.random((b, 3)) * box
                best_pos, _ = _check_trials(trials, si, first_ok_only=True)
        if best_pos is None:
            raise RuntimeError(
                f"放置原子失败：{symbols[i]} (第 {i+1}/{n} 个)。"
                " 你可以尝试：\n"
                "1) 略微增大盒长；\n"
                "2) 降低某些最小距离；\n"
                "3) 调整放置顺序；\n"
                "4) 先做更小体系。"
            )
        positions[i] = best_pos
        # 加入 cell，并使其 27 邻域缓存失效
        cxyz = np.floor(best_pos / cell).astype(np.int64) % nc
        key = _encode(int(cxyz[0]), int(cxyz[1]), int(cxyz[2]))
        cells.setdefault(key, []).append(i)
        cx, cy, cz = _decode(key)
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb_cache.pop(_encode((cx + dx) % nc, (cy + dy) % nc, (cz + dz) % nc), None)

    return positions


def _candidate_pairs(pos: np.ndarray, box: float, rc: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回候选原子对 (i, j)（i < j，无重复），保证覆盖所有距离 < rc 的对。
    盒子每维 >= 3*rc 时用 cell list（cell 边长 = rc）；否则回退全对组合。
    """
    n = len(pos)
    nc = int(box // rc) if rc > 0 else 0
    if nc < 3:
        iu, ju = np.triu_indices(n, k=1)
        return iu.astype(np.int64), ju.astype(np.int64)
    c = np.floor(pos / rc).astype(np.int64) % nc
    cid = (c[:, 0] * nc + c[:, 1]) * nc + c[:, 2]
    order = np.argsort(cid, kind="stable")
    ncells = nc ** 3
    cell_start = np.searchsorted(cid[order], np.arange(ncells + 1))
    counts = np.diff(cell_start).astype(np.int64)

    i_out: List[np.ndarray] = []
    j_out: List[np.ndarray] = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nb_c = (c + np.array([dx, dy, dz])) % nc
                nb_cid = (nb_c[:, 0] * nc + nb_c[:, 1]) * nc + nb_c[:, 2]
                k = counts[nb_cid]
                total = int(k.sum())
                if total == 0:
                    continue
                i_arr = np.repeat(np.arange(n, dtype=np.int64), k)
                within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(k) - k, k)
                j_arr = order[np.repeat(cell_start[nb_cid], k) + within]
                keep = i_arr < j_arr  # 27 全偏移双向展开，i<j 恰保留一次
                if keep.any():
                    i_out.append(i_arr[keep])
                    j_out.append(j_arr[keep])
    if not i_out:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(i_out), np.concatenate(j_out)


def relax_positions(
    symbols: List[str],
    positions: np.ndarray,
    box: float,
    steps: int = 200,
    dt: float = 0.03,
    seed: int = 42
) -> np.ndarray:
    """
    梯度下降软球弛豫（Verlet 邻居表 + 向量化版）。
    双层势：硬核区 (r < d_min) 二次排斥 + 软尾区 (d_min <= r < 1.25*d_min) 线性排斥。
    势能在 1.25*d_min 处精确为 0：以 rc+skin 建邻居表，任一原子累计位移超过
    skin/2 时才重建，力的作用对集合与"每步全量重建"完全一致，不引入任何近似。
    数万原子 x 200 步亦可在数秒内完成。
    """
    rng = np.random.default_rng(seed)
    n = len(symbols)
    pos = positions.copy().astype(np.float64)
    if n < 2:
        return pos

    tab, uniq = _pair_table(symbols)
    sidx = _symbol_index_array(symbols, uniq)
    uids = np.unique(sidx)
    rc = 1.25 * float(tab[np.ix_(uids, uids)].max())  # 最大软尾截断半径
    skin = 1.5                                        # 邻居表皮肤厚度 (Å)
    rc_build = rc + skin

    i_arr = np.empty(0, dtype=np.int64)
    j_arr = np.empty(0, dtype=np.int64)
    pos_build = None

    for _step in range(steps):
        # 建表后任一原子位移超过 skin/2 才重建邻居表（保证力集合精确不变）；
        # 位移取最小镜像，避免跨周期边界回绕造成假触发
        if pos_build is not None:
            disp = pos - pos_build
            disp -= box * np.round(disp / box)
            need_rebuild = float(np.abs(disp).max()) > 0.5 * skin
        else:
            need_rebuild = True
        if need_rebuild:
            i_arr, j_arr = _candidate_pairs(pos, box, rc_build)
            if i_arr.size:
                drb = pos[i_arr] - pos[j_arr]
                drb -= box * np.round(drb / box)
                keep = np.einsum("pi,pi->p", drb, drb) < rc_build ** 2
                i_arr, j_arr = i_arr[keep], j_arr[keep]
            pos_build = pos.copy()
        if i_arr.size == 0:
            break
        dr = pos[i_arr] - pos[j_arr]
        dr -= box * np.round(dr / box)
        d2 = np.einsum("pi,pi->p", dr, dr)
        dmin_p = tab[sidx[i_arr], sidx[j_arr]]
        sel = d2 < (1.25 * dmin_p) ** 2  # 软尾之外无相互作用
        if not sel.any():
            break
        i_s, j_s = i_arr[sel], j_arr[sel]
        dr_s = dr[sel]
        dmin_s = dmin_p[sel]
        dist = np.sqrt(d2[sel])

        # 重合对：用微小随机方向踢开
        collide = dist < 1e-12
        if collide.any():
            kick = (rng.random((int(collide.sum()), 3)) - 0.5) * 1e-6
            dr_s[collide] = kick
            dist[collide] = np.linalg.norm(kick, axis=1)

        dist_safe = np.where(dist > 0.0, dist, 1.0)
        overlap_h = np.where(dist < dmin_s, dmin_s - dist, 0.0)
        overlap_s = np.where((dist >= dmin_s) & (dist < 1.25 * dmin_s),
                             1.25 * dmin_s - dist, 0.0)

        # i<j 对无重复，能量不除 2
        energy = (overlap_h * overlap_h).sum() + (overlap_s * 0.5 * dmin_s).sum()
        if energy < 1e-12:
            break

        mag = np.where(overlap_h > 0.0, 2.0 * overlap_h / dist_safe, 0.0)
        mag += np.where(overlap_s > 0.0,
                        overlap_s / (dist_safe * dmin_s + 1e-12) * 0.3, 0.0)
        force = np.empty((n, 3), dtype=float)
        for ax in range(3):
            w = mag * dr_s[:, ax]
            force[:, ax] = (np.bincount(i_s, weights=w, minlength=n)
                            - np.bincount(j_s, weights=w, minlength=n))

        # 限制最大位移防止 overshoot
        max_f = float(np.linalg.norm(force, axis=1).max())
        step_scale = min(dt, 0.08 / max_f) if max_f > 1e-12 else dt
        pos = (pos + force * step_scale) % box

    return pos


def _distribute_atoms(atom_counts: Dict[str, int], parts: int,
                      rng: np.random.Generator) -> List[Dict[str, int]]:
    """把各元素原子数尽量均匀地随机分配到 parts 个区块（整体配比不变）。"""
    keys = sorted(atom_counts)
    blocks: List[Dict[str, int]] = [{el: atom_counts[el] // parts for el in keys} for _ in range(parts)]
    for el in keys:
        rem = atom_counts[el] % parts
        if rem:
            for b in rng.choice(parts, size=rem, replace=False):
                blocks[int(b)][el] += 1
    return blocks


def _run_subprocess_workers(tasks: List[Dict[str, Any]], jobs: int) -> List[Any]:
    """用独立子进程并行执行任务（subprocess + 线程等待）。

    不依赖 multiprocessing 的信号量/管道，在受限环境（沙箱、杀软拦截）
    下也能工作；子进程通过隐藏子命令 --_worker 复用本脚本，任务与结果
    用 pickle 文件交换。
    """
    tmpdir = os.path.join(os.getcwd(), f"_blocks_tmp_{os.getpid()}")
    os.makedirs(tmpdir, exist_ok=True)
    try:
        script = os.path.abspath(__file__)

        def _run_one(idx_task):
            idx, task = idx_task
            task_path = os.path.join(tmpdir, f"task_{idx}.pkl")
            result_path = os.path.join(tmpdir, f"result_{idx}.pkl")
            with open(task_path, "wb") as f:
                pickle.dump(dict(task, result_path=result_path), f)
            proc = subprocess.run(
                [sys.executable, script, "--_worker", task_path],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"子进程失败(exit {proc.returncode})：{(proc.stderr or '')[-400:]}"
                )
            with open(result_path, "rb") as f:
                return pickle.load(f)

        results: List[Any] = [None] * len(tasks)
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {pool.submit(_run_one, it): it[0] for it in enumerate(tasks)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _worker_main(task_path: str) -> None:
    """子进程入口：执行单个区块放置或单行建模任务，结果写回 pickle。"""
    with open(task_path, "rb") as f:
        task = pickle.load(f)
    if task.get("kind", "row") == "block":
        symbols = expand_symbols(task["atom_counts"])
        pos = place_atoms_randomly(symbols, box=task["box"], seed=task["seed"])
        result = {"symbols": symbols, "positions": pos}
    else:
        label = task.get("label", "row")
        try:
            symbols, positions, box, _fu, _ac = build_glass_model(
                mol_pct=task["mol_pct"],
                density_g_cm3=task["density"],
                target_atoms=task["atoms"],
                seed=task["seed"],
                box_inflate=task["box_inflate"],
                relax=task["relax"],
                relax_steps=task["relax_steps"],
                jobs=1,  # 行并行模式行内串行，避免嵌套并行
            )
            write_extended_xyz(task["output"], symbols, positions, box)
            result = {"label": label, "ok": True, "output": task["output"],
                      "natoms": len(symbols), "error": None}
        except Exception as e:
            result = {"label": label, "ok": False, "output": task.get("output", ""),
                      "natoms": 0, "error": str(e)}
    with open(task["result_path"], "wb") as f:
        pickle.dump(result, f)


def _parallel_block_placement(atom_counts: Dict[str, int], box: float,
                              seed: int, jobs: int):
    """把盒子按 P×P×P 个立方块分解，多进程并行放置后拼接回全局盒子。

    每块独立做周期性放置（密度与全局一致，复用同一放置算法），
    块间界面可能存在少量违反最小距离的原子对，由随后的全局弛豫消除。
    同 seed 结果可复现。不满足并行条件时返回 None。
    """
    n_total = sum(atom_counts.values())
    p = 0
    for cand in (3, 2):  # 27 块 / 8 块
        if jobs >= cand ** 3 and n_total // (cand ** 3) >= 600:
            p = cand
            break
    if p == 0:
        return None
    parts = p ** 3
    print(f"[并行放置] {p}x{p}x{p} 分块，共 {parts} 个子进程，每块约 {n_total // parts} 原子")
    blocks = _distribute_atoms(atom_counts, parts, np.random.default_rng(seed + 7))
    sub = box / p
    tasks = []
    for b, counts_b in enumerate(blocks):
        ix, iy, iz = b // (p * p), (b // p) % p, b % p
        tasks.append({
            "kind": "block",
            "origin": (ix * sub, iy * sub, iz * sub),
            "atom_counts": counts_b,
            "box": sub,
            "seed": seed + 1009 * (b + 1),
        })
    symbols_all: List[str] = []
    pos_all = np.empty((n_total, 3), dtype=float)
    filled = 0
    try:
        results = _run_subprocess_workers(tasks, min(parts, jobs))
    except Exception as e:
        print(f"[警告] 并行放置失败({e})，回退串行放置")
        return None
    for task, res in zip(tasks, results):
        symbols_b, pos_b = res["symbols"], res["positions"]
        symbols_all.extend(symbols_b)
        nb = len(symbols_b)
        pos_all[filled:filled + nb] = np.asarray(pos_b) + np.array(task["origin"])
        filled += nb
    return symbols_all, pos_all[:filled]


def build_glass_model(
    mol_pct: Dict[str, float],
    density_g_cm3: float,
    target_atoms: int,
    seed: int = 42,
    box_inflate: float = 1.00,
    relax: bool = True,
    relax_steps: int = 150,
    jobs: int = 0,
) -> Tuple[List[str], np.ndarray, float, Dict[str, int], Dict[str, int]]:
    """
    返回:
    symbols, positions(Å), box_length(Å), formula_units, atom_counts

    jobs: 并行进程数。0=自动(CPU 核数)；1=强制串行；>=8 且原子数足够时，
          按 2x2x2（或 3x3x3）立方块分块并行放置，再全局弛豫。
    """
    formula_units = choose_formula_units(mol_pct, target_atoms=target_atoms)
    atom_counts = formula_units_to_atom_counts(formula_units)
    box = cubic_box_length_from_density(formula_units, density_g_cm3) * box_inflate

    symbols = None
    relaxed_parallel = False
    if jobs != 1:
        eff = jobs if jobs > 0 else (os.cpu_count() or 1)
        par = _parallel_block_placement(atom_counts, box, seed, eff)
        if par is not None:
            symbols, positions = par
            relaxed_parallel = True  # 分块界面有少量浅层接触，多弛豫几步清干净
    if symbols is None:
        symbols = expand_symbols(atom_counts)
        positions = place_atoms_randomly(symbols, box=box, seed=seed)
    if relax:
        steps = relax_steps * 2 if relaxed_parallel else relax_steps
        positions = relax_positions(symbols, positions, box,
                                    steps=steps, seed=seed + 1)
    return symbols, positions, box, formula_units, atom_counts


def write_extended_xyz(filename: str, symbols: List[str], positions: np.ndarray, box: float) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{len(symbols)}\n")
        f.write(
            f'Lattice="{box:.6f} 0 0 0 {box:.6f} 0 0 0 {box:.6f}" '
            f'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
        )
        for sym, xyz in zip(symbols, positions):
            f.write(f"{sym:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")


def print_summary(formula_units: Dict[str, int], atom_counts: Dict[str, int], box: float) -> None:
    total_atoms = sum(atom_counts.values())
    print("=== 公式单元数 ===")
    for k, v in formula_units.items():
        print(f"{k:6s}: {v}")
    print("\n=== 原子数 ===")
    for k, v in atom_counts.items():
        print(f"{k:2s}: {v}")
    print(f"\n总原子数: {total_atoms}")
    print(f"立方盒边长: {box:.4f} Angstrom")
    print("\nCP2K &CELL 可直接写成：")
    print("&CELL")
    print(f"  ABC {box:.6f} {box:.6f} {box:.6f}")
    print("  PERIODIC XYZ")
    print("&END CELL")


def parse_composition(s: str) -> Dict[str, float]:
    """解析配方字符串 'SiO2:32,Li2O:7,ZnO:39,B2O3:22' -> {oxide: mol_pct}"""
    mol_pct = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        oxide, val = part.rsplit(":", 1)
        oxide = oxide.strip()
        mol_pct[oxide] = float(val.strip())
    return mol_pct


def _row_val(row: Dict[str, Any], name: str, default):
    """从 CSV 行读取参数值，行级优先于默认值"""
    v = row.get(name)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "true", "1"):
            return True
        if s in ("no", "false", "0"):
            return False
    try:
        if isinstance(default, bool):
            return v
        if isinstance(default, float):
            return float(v)
        if isinstance(default, int):
            return int(float(v))
        return v
    except (ValueError, TypeError):
        return default


def _build_one(task: Dict[str, Any]) -> Dict[str, Any]:
    """单个配方建模 + 写出 XYZ（供多进程调用）"""
    label = task["label"]
    try:
        symbols, positions, box, _fu, _ac = build_glass_model(
            mol_pct=task["mol_pct"],
            density_g_cm3=task["density"],
            target_atoms=task["atoms"],
            seed=task["seed"],
            box_inflate=task["box_inflate"],
            relax=task["relax"],
            relax_steps=task["relax_steps"],
            jobs=1,  # CSV 多行模式按行并行，行内串行
        )
        write_extended_xyz(task["output"], symbols, positions, box)
        return {"label": label, "ok": True, "output": task["output"], "natoms": len(symbols), "error": None}
    except Exception as e:
        return {"label": label, "ok": False, "output": task["output"], "natoms": 0, "error": str(e)}


def _process_csv(filepath: str, args) -> None:
    """从 CSV 文件批量读取配方并生成结构"""
    try:
        import pandas as pd
        df = pd.read_csv(filepath, comment="#")
        rows = df.to_dict("records")
    except ImportError:
        # 无 pandas 时用标准库 csv
        import csv as csv_mod
        with open(filepath, "r", encoding="utf-8-sig") as fh:
            reader = csv_mod.DictReader(fh)
            rows = list(reader)

    if not rows:
        print("错误: CSV 文件中没有数据行")
        sys.exit(1)

    oxide_names = [k for k in rows[0].keys() if k in OXIDES]
    if not oxide_names:
        print("错误: CSV 中未找到任何氧化物列名，请用 --list 查看支持的氧化物")
        sys.exit(1)

    # 构建任务列表
    tasks = []
    for idx, row in enumerate(rows):
        label = row.get("label", "").strip() or f"batch_{idx+1}"

        mol_pct = {}
        for ox in oxide_names:
            val = row.get(ox)
            if isinstance(val, str):
                val = val.strip()
            if val is None or val == "":
                continue
            try:
                v = float(val)
            except (ValueError, TypeError):
                continue
            if v > 0:
                mol_pct[ox] = v

        if not mol_pct:
            print(f"[{label}] 跳过: 配方为空")
            continue

        tasks.append({
            "label": label,
            "mol_pct": mol_pct,
            "density": _row_val(row, "density", args.density),
            "atoms": int(_row_val(row, "atoms", args.atoms)),
            "seed": int(_row_val(row, "seed", args.seed)),
            "output": str(row.get("output", "")).strip() or f"{label}.xyz",
            "relax": _row_val(row, "relax", not args.no_relax),
            "relax_steps": int(_row_val(row, "relax_steps", args.relax_steps)),
            "box_inflate": float(_row_val(row, "box_inflate", args.box_inflate)),
        })

    njobs = args.jobs if args.jobs > 0 else None
    print(f"从 {filepath} 读取 {len(rows)} 个配方 -> {len(tasks)} 个有效任务")
    print(f"并行进程数: {njobs or '自动 (CPU 核数)'}")
    print()

    if len(tasks) == 1 and njobs != 1:
        # 单任务：单个结构内部按 2x2x2（或 3x3x3）分块并行放置
        task = tasks[0]
        comp = ", ".join(f"{k}={v:.1f}" for k, v in task["mol_pct"].items())
        print(f"[{task['label']}] {comp}")
        try:
            symbols, positions, box, _fu, _ac = build_glass_model(
                mol_pct=task["mol_pct"],
                density_g_cm3=task["density"],
                target_atoms=task["atoms"],
                seed=task["seed"],
                box_inflate=task["box_inflate"],
                relax=task["relax"],
                relax_steps=task["relax_steps"],
                jobs=args.jobs,
            )
            write_extended_xyz(task["output"], symbols, positions, box)
            print(f"  -> 已写出 {task['output']} ({len(symbols)} atoms)")
            print("\n完成: 1/1 个配方成功")
        except Exception as e:
            print(f"  -> 失败: {e}")
            print("\n完成: 0/1 个配方成功")
        return

    if njobs == 1:
        # 单进程模式：直接循环
        success = 0
        for task in tasks:
            info = f"[{task['label']}] {', '.join(f'{k}={v:.1f}' for k, v in task['mol_pct'].items())}"
            print(info)
            result = _build_one(task)
            if result["ok"]:
                print(f"  -> 已写出 {result['output']} ({result['natoms']} atoms)")
                success += 1
            else:
                print(f"  -> 失败: {result['error']}")
    else:
        success = 0
        try:
            results = _run_subprocess_workers(
                [dict(t, kind="row") for t in tasks], njobs)
        except Exception as e:
            print(f"[警告] 行并行失败({e})，改用串行")
            results = None
        if results is not None:
            for task, result in zip(tasks, results):
                comp = ", ".join(f"{k}={v:.1f}" for k, v in task["mol_pct"].items())
                print(f"[{task['label']}] {comp}")
                if result["ok"]:
                    print(f"  -> 已写出 {result['output']} ({result['natoms']} atoms)")
                    success += 1
                else:
                    print(f"  -> 失败: {result['error']}")
        else:
            for task in tasks:
                result = _build_one(task)
                comp = ", ".join(f"{k}={v:.1f}" for k, v in task["mol_pct"].items())
                print(f"[{task['label']}] {comp}")
                if result["ok"]:
                    print(f"  -> 已写出 {result['output']} ({result['natoms']} atoms)")
                    success += 1
                else:
                    print(f"  -> 失败: {result['error']}")

    print(f"\n完成: {success}/{len(tasks)} 个配方成功")


def _generate_csv_template(filepath: str) -> None:
    """生成 CSV 模板文件"""
    header = ["label", "output"] + list(OXIDES.keys()) + ["density", "atoms", "seed", "relax"]
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        f.write(",".join(header) + "\n")
        # 示例行
        example = ["G1", "glass_G1.xyz", "32", "22", "", "", "", "7", "", "", "39", "", "", "", "", "", "", ""] + ["2.955", "220", "42", "1"]
        f.write(",".join(example) + "\n")
    print(f"模板文件已生成: {filepath}")
    print("编辑此文件，每行一个配方（空列=不添加该氧化物）")


def _cli() -> None:
    p = argparse.ArgumentParser(
        prog="python 建模.py",
        description="氧化物玻璃初始结构建模工具 —— 生成 extended XYZ 格式用于 CP2K/ABACUS 等 DFT 软件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单配方模式
  python 建模.py "SiO2:32,Li2O:7,ZnO:39,B2O3:22"
  python 建模.py "SiO2:40,Na2O:10,CaO:15,Al2O3:10,B2O3:25" -d 2.65 -n 200

  # CSV 批量模式
  python 建模.py batch.csv
  python 建模.py batch.csv -j 4 --no-relax

  # 其他
  python 建模.py --template batch.csv
  python 建模.py --list
""",
    )
    p.add_argument(
        "input", nargs="?", default=None,
        help="配方字符串 'SiO2:32,Li2O:7,...' 或 CSV 文件路径（以 .csv 结尾）"
    )
    p.add_argument("-d", "--density", type=float, default=2.955,
                   help="目标密度 g/cm^3 (默认: 2.955)")
    p.add_argument("-n", "--atoms", type=int, default=220,
                   help="目标总原子数 (默认: 220)")
    p.add_argument("-s", "--seed", type=int, default=42,
                   help="随机种子 (默认: 42)")
    p.add_argument("-o", "--output", type=str, default="glass_init.xyz",
                   help="输出文件名，单配方模式用 (默认: glass_init.xyz)")
    p.add_argument("--no-relax", action="store_true",
                   help="跳过弛豫步骤")
    p.add_argument("--relax-steps", type=int, default=150,
                   help="弛豫步数 (默认: 150)")
    p.add_argument("--box-inflate", type=float, default=1.02,
                   help="盒膨胀系数，>1 放大盒长便于放置 (默认: 1.02)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="静默模式，单配方模式下不打印摘要")
    p.add_argument("--list", dest="list_oxides", action="store_true",
                   help="列出所有支持的氧化物并退出")
    p.add_argument("--template", type=str, default=None, metavar="FILE",
                   help="生成 CSV 模板文件并退出")
    p.add_argument("-j", "--jobs", type=int, default=0, metavar="N",
                   help="并行进程数，0=自动 (默认: 0)。CSV 多行按行并行；"
                        "单配方/单行任务在 >=8 进程时按 2x2x2 分块并行放置")

    args = p.parse_args()

    if args.list_oxides:
        print("支持的氧化物:")
        for name, info in OXIDES.items():
            atoms_str = " ".join(f"{el}{n}" for el, n in info["atoms"].items())
            print(f"  {name:6s}  ({atoms_str})  MW={info['mw']:.4f} g/mol")
        print(f"\n共 {len(OXIDES)} 种")
        return

    if args.template:
        _generate_csv_template(args.template)
        return

    if args.input is None:
        p.print_help()
        print("\n错误: 请提供配方字符串或 CSV 文件路径")
        print("  单配方: python 建模.py \"SiO2:32,Li2O:7,ZnO:39,B2O3:22\"")
        print("  批量:   python 建模.py batch.csv")
        print("  模板:   python 建模.py --template batch.csv")
        sys.exit(1)

    # CSV 批量模式
    if args.input.endswith(".csv"):
        _process_csv(args.input, args)
        return

    # 单配方模式
    mol_pct = parse_composition(args.input)
    mol_pct = {k: v for k, v in mol_pct.items() if v > 0}
    if not mol_pct:
        print("错误: 配方中至少需要一个氧化物且摩尔比 > 0")
        sys.exit(1)

    unknown = [k for k in mol_pct if k not in OXIDES]
    if unknown:
        print(f"错误: 未知氧化物 {unknown}")
        print("使用 --list 查看支持的氧化物列表")
        sys.exit(1)

    if not args.quiet:
        print(f"配方: {', '.join(f'{k}={v:.1f}%' for k, v in mol_pct.items())}")
        print(f"密度: {args.density} g/cm^3, 目标原子数: {args.atoms}")
        print(f"弛豫: {'关' if args.no_relax else f'{args.relax_steps} 步'},  种子: {args.seed}")
        print()

    symbols, positions, box, formula_units, atom_counts = build_glass_model(
        mol_pct=mol_pct,
        density_g_cm3=args.density,
        target_atoms=args.atoms,
        seed=args.seed,
        box_inflate=args.box_inflate,
        relax=not args.no_relax,
        relax_steps=args.relax_steps,
        jobs=args.jobs,
    )

    if not args.quiet:
        print_summary(formula_units, atom_counts, box)

    write_extended_xyz(args.output, symbols, positions, box)
    print(f"已写出: {args.output}  ({len(symbols)} atoms)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--_worker":
        _worker_main(sys.argv[2])
    else:
        _cli()