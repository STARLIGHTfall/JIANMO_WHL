import argparse
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def choose_formula_units(
    mol_pct: Dict[str, float],
    target_atoms: int,
    search_points: int = 4000
) -> Dict[str, int]:
    """
    根据氧化物摩尔百分比，搜索最接近 target_atoms 的整数公式单元数。
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

    best_fu = None
    best_score = float("inf")

    s_min = max(1.0, 0.5 * s0)
    s_max = 1.5 * s0

    for s in np.linspace(s_min, s_max, search_points):
        fu = {}
        for oxide, frac in mol_frac.items():
            n = int(round(frac * s))
            fu[oxide] = max(1, n)

        total_atoms_now = sum(
            fu[ox] * atoms_per_formula_unit(ox) for ox in fu
        )

        # 配比误差
        fu_total = sum(fu.values())
        comp_err = sum(abs(fu[ox] / fu_total - mol_frac[ox]) for ox in fu)

        score = abs(total_atoms_now - target_atoms) + 30.0 * comp_err

        if score < best_score:
            best_score = score
            best_fu = fu

    if best_fu is None:
        raise RuntimeError("未能找到合适的整数公式单元数。")

    return best_fu


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
    rng = np.random.default_rng(seed)
    positions = []
    placed_symbols = []

    for i, sym in enumerate(symbols):
        best_pos = None
        best_mindist = -1.0

        # 阶段 1：在 best_of 次尝试中选最优合法位置
        for _ in range(best_of):
            trial = rng.random(3) * box
            ok = True
            trial_mindist = float("inf")
            for prev_sym, prev_pos in zip(placed_symbols, positions):
                d = pbc_distance(trial, prev_pos, box)
                if d < trial_mindist:
                    trial_mindist = d
                if d < min_dist(sym, prev_sym):
                    ok = False
                    break
            if ok:
                if trial_mindist > best_mindist:
                    best_mindist = trial_mindist
                    best_pos = trial.copy()
                if best_mindist > 2.5:
                    break

        # 阶段 2：若未找到，回退到"第一个合法即接受"
        if best_pos is None:
            for _ in range(max_trials_per_atom):
                trial = rng.random(3) * box
                ok = True
                for prev_sym, prev_pos in zip(placed_symbols, positions):
                    if pbc_distance(trial, prev_pos, box) < min_dist(sym, prev_sym):
                        ok = False
                        break
                if ok:
                    best_pos = trial.copy()
                    break

        if best_pos is None:
            raise RuntimeError(
                f"放置原子失败：{sym} (第 {i+1}/{len(symbols)} 个)。"
                " 你可以尝试：\n"
                "1) 略微增大盒长；\n"
                "2) 降低某些最小距离；\n"
                "3) 调整放置顺序；\n"
                "4) 先做更小体系。"
            )

        positions.append(best_pos)
        placed_symbols.append(sym)

    return np.array(positions, dtype=float)


def relax_positions(
    symbols: List[str],
    positions: np.ndarray,
    box: float,
    steps: int = 200,
    dt: float = 0.03,
    seed: int = 42
) -> np.ndarray:
    """
    梯度下降软球弛豫。
    双层势：硬核区 (r < d_min) 二次排斥 + 软尾区 (d_min <= r < 1.25*d_min) 线性排斥。
    200 原子 x 200 步约 2~4 秒。
    """
    rng = np.random.default_rng(seed)
    n = len(symbols)
    pos = positions.copy().astype(np.float64)

    # 预计算对距离矩阵
    dmin = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = min_dist(symbols[i], symbols[j])
            dmin[i, j] = d
            dmin[j, i] = d

    for step in range(steps):
        force = np.zeros_like(pos)
        energy = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                dr = pos[i] - pos[j]
                dr -= box * np.round(dr / box)
                dist = np.linalg.norm(dr)
                if dist < 1e-12:
                    dr = (rng.random(3) - 0.5) * 1e-6
                    dist = np.linalg.norm(dr)

                d0 = dmin[i, j]
                d_soft = 1.25 * d0

                if dist < d0:
                    # 硬核区：二次排斥  E = (d0 - r)^2,  F = 2*(d0 - r)/r * dr
                    overlap = d0 - dist
                    energy += overlap * overlap
                    mag = 2.0 * overlap / dist
                    force[i] += mag * dr
                    force[j] -= mag * dr
                elif dist < d_soft:
                    # 软尾区：线性排斥  E = (d_soft - r),  F = (1/r) * dr
                    overlap = d_soft - dist
                    energy += overlap * 0.5 * d0  # 权重低于硬核
                    mag = overlap / (dist * d0 + 1e-12)
                    force[i] += mag * dr * 0.3
                    force[j] -= mag * dr * 0.3

        if energy < 1e-12:
            break

        # 限制最大位移防止 overshoot
        max_f = 0.0
        for fi in force:
            nf = np.linalg.norm(fi)
            if nf > max_f:
                max_f = nf
        if max_f > 1e-12:
            step_scale = min(dt, 0.08 / max_f)

        for i in range(n):
            pos[i] += force[i] * step_scale
            pos[i] %= box

    return pos


def build_glass_model(
    mol_pct: Dict[str, float],
    density_g_cm3: float,
    target_atoms: int,
    seed: int = 42,
    box_inflate: float = 1.00,
    relax: bool = True,
    relax_steps: int = 150,
) -> Tuple[List[str], np.ndarray, float, Dict[str, int], Dict[str, int]]:
    """
    返回:
    symbols, positions(Å), box_length(Å), formula_units, atom_counts
    """
    formula_units = choose_formula_units(mol_pct, target_atoms=target_atoms)
    atom_counts = formula_units_to_atom_counts(formula_units)
    box = cubic_box_length_from_density(formula_units, density_g_cm3) * box_inflate
    symbols = expand_symbols(atom_counts)
    positions = place_atoms_randomly(symbols, box=box, seed=seed)
    if relax:
        positions = relax_positions(symbols, positions, box,
                                    steps=relax_steps, seed=seed + 1)
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
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            futures = {executor.submit(_build_one, t): t for t in tasks}
            for future in as_completed(futures):
                task = futures[future]
                label = task["label"]
                comp = ", ".join(f"{k}={v:.1f}" for k, v in task["mol_pct"].items())
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[{label}] {comp}")
                    print(f"  -> 进程异常: {e}")
                    continue
                if result["ok"]:
                    print(f"[{label}] {comp}")
                    print(f"  -> 已写出 {result['output']} ({result['natoms']} atoms)")
                    success += 1
                else:
                    print(f"[{label}] {comp}")
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
                   help="并行进程数，0=自动 (默认: 0)")

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
    )

    if not args.quiet:
        print_summary(formula_units, atom_counts, box)

    write_extended_xyz(args.output, symbols, positions, box)
    print(f"已写出: {args.output}  ({len(symbols)} atoms)")


if __name__ == "__main__":
    _cli()