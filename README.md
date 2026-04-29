# 氧化物玻璃初始结构建模工具

将氧化物摩尔百分比配方转换为可用于 DFT（CP2K / ABACUS / VASP）计算的初始三维结构，输出 extended XYZ 格式。

## 快速开始

```bash
# 安装依赖（仅需 numpy）
pip install numpy

# 生成一个玻璃结构
python 建模.py "SiO2:32,Li2O:7,ZnO:39,B2O3:22"
```

输出 `glass_init.xyz`，可直接导入 CP2K 或 ASE/OVITO 查看。

---

## 命令行参数

```
python 建模.py <输入> [选项]
```

### 输入（二选一）

| 形式 | 示例 |
|------|------|
| 配方字符串 | `"SiO2:32,Li2O:7,ZnO:39,B2O3:22"` |
| CSV 批量文件 | `batch.csv` |

### 选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-d, --density` | float | 2.955 | 目标密度 (g/cm³) |
| `-n, --atoms` | int | 220 | 目标总原子数 |
| `-s, --seed` | int | 42 | 随机种子 |
| `-o, --output` | str | glass_init.xyz | 输出文件名（单配方模式） |
| `--no-relax` | flag | - | 跳过弛豫（更快但质量稍低） |
| `--relax-steps` | int | 150 | 弛豫迭代步数 |
| `--box-inflate` | float | 1.02 | 盒膨胀系数 |
| `-q, --quiet` | flag | - | 静默模式，不打印摘要 |
| `-j, --jobs` | int | 0 | 并行进程数，0=自动（仅 CSV 模式） |
| `--list` | - | - | 列出所有支持的氧化物 |
| `--template FILE` | str | - | 生成 CSV 模板 |

---

## 使用场景

### 场景一：单配方快速建模

```bash
# 基本用法
python 建模.py "SiO2:32,Li2O:7,ZnO:39,B2O3:22"

# 指定密度和原子数
python 建模.py "SiO2:40,Al2O3:15,CaO:20,MgO:10,B2O3:15" -d 2.70 -n 300

# 关闭弛豫加速（初筛用）
python 建模.py "SiO2:50,B2O3:30,Na2O:20" --no-relax -q

# 高分掺杂玻璃
python 建模.py "SiO2:25,ZnO:35,B2O3:20,La2O3:15,Nb2O5:5" -d 3.80 -n 250
```

### 场景二：CSV 批量建模

**步骤 1 — 生成模板**

```bash
python 建模.py --template my_compositions.csv
```

生成内容：

| label | output | SiO2 | B2O3 | Li2O | Na2O | CaO | Al2O3 | ... | density | atoms | seed | relax |
|-------|--------|------|------|------|------|-----|-------|-----|---------|-------|------|-------|
| G1 | glass_G1.xyz | 32 | 22 | 7 | | | | | 2.955 | 220 | 42 | 1 |
| G2 | glass_G2.xyz | 30 | 20 | 5 | | 15 | 10 | | 3.10 | 200 | 42 | 0 |

**步骤 2 — 编辑 CSV**

- 列名 = 氧化物名称 → 对应 mol% 值
- 不需要的氧化物留空或填 0
- `output`：输出文件名（必填，或填 `label` 列自动生成）
- `relax`：`1` = 弛豫，`0` = 跳过
- `density`/`atoms`/`seed`：不填则用 CLI 默认值

**步骤 3 — 运行**

```bash
# 自动并行
python 建模.py my_compositions.csv

# 指定 4 核 + 全部关闭弛豫
python 建模.py my_compositions.csv --no-relax -j 4

# 单进程（调试用）
python 建模.py my_compositions.csv -j 1
```

### 场景三：密度/成分扫描

在同一 CSV 中写不同密度的同一配方，实现密度扫描：

```csv
label,output,SiO2,B2O3,Li2O,ZnO,density,atoms,seed,relax
G_d2.8,scan_d2.8.xyz,32,22,7,39,2.80,220,42,1
G_d3.0,scan_d3.0.xyz,32,22,7,39,3.00,220,42,1
G_d3.2,scan_d3.2.xyz,32,22,7,39,3.20,220,42,1
G_d3.4,scan_d3.4.xyz,32,22,7,39,3.40,220,42,1
```

```bash
python 建模.py scan.csv -j 4
```

---

## 输出格式

Extended XYZ 格式（ASE/OVITO/CP2K 均可读取）：

```
220
Lattice="14.319704 0 0 0 14.319704 0 0 0 14.319704" Properties=species:S:1:pos:R:3 pbc="T T T"
Zn     2.54774201     9.15744185     6.70772051
Si     5.32482554     5.04520469    11.31429554
O     12.96139214     2.53964527     9.34768538
...
```

包含完整立方晶胞信息，可直接用作 CP2K `&COORD` 输入。

---

## 支持的氧化物（18 种）

| 类别 | 氧化物 |
|------|--------|
| 网络形成体 | SiO₂, B₂O₃, P₂O₅, GeO₂ |
| 碱金属修饰体 | Li₂O, Na₂O, K₂O |
| 碱土金属修饰体 | MgO, CaO, BaO |
| 中间体 | ZnO, Al₂O₃, TiO₂, ZrO₂ |
| 稀土 | La₂O₃, Y₂O₃ |
| 高价掺杂 | Nb₂O₅, Ta₂O₅ |

运行 `python 建模.py --list` 查看完整列表及摩尔质量。

---

## Python API

可在脚本中调用：

```python
from 建模 import build_glass_model, write_extended_xyz, print_summary

symbols, positions, box, formula_units, atom_counts = build_glass_model(
    mol_pct={"SiO2": 32, "Li2O": 7, "ZnO": 39, "B2O3": 22},
    density_g_cm3=2.955,
    target_atoms=220,
    seed=42,
    box_inflate=1.02,
    relax=True,        # 是否弛豫
    relax_steps=150,   # 弛豫步数
)

write_extended_xyz("output.xyz", symbols, positions, box)
```

---

## 常见问题

**Q: 配方百分比不需要归一化到 100？**
不需要。`SiO2:32,Li2O:16` 和 `SiO2:66.7,Li2O:33.3` 效果相同，内部会自动归一化。

**Q: 弛豫开关怎么选？**
- 需要高质量初始结构 → 开弛豫（默认），耗时约 30s（220 原子）
- 批量初筛 / 快速迭代 → `--no-relax`，耗时约 10s

**Q: 放置失败怎么办？**
提示 "放置原子失败" 时尝试：
1. 增大 `--box-inflate`（如 1.05）
2. 减少 `--atoms`（如 180）
3. 调大密度降低值若干

**Q: 输出文件可在哪些软件中使用？**
- CP2K：直接作为 `&COORD` 输入，`&CELL` 参数已打印到控制台
- ABACUS：使用 extended XYZ 格式
- VASP：可通过 ASE `read_xyz` + `write_poscar` 转换
- OVITO / VESTA：直接拖入查看结构
