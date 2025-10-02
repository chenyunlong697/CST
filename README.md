# CST
标题：从翼型坐标生成“CST厚度/弯度 + 14维特征 + Sobol敏感性 + 后续优化变量清单”的Python工具

【目标/范围】
- 输入：单个翼型坐标文件（Selig .dat/.txt 或 CSV）。仓库在 `data/` 目录下提供了示例 `NACA4412.dat` / `NACA4412.txt`，可直接用于体验。
- 处理：归一化（弦长=1）、余弦聚点重采样、CST（厚度/弯度分解，N1=0.5,N2=1.0，nt=5,nc=3）拟合。
- 输出：
  1) 基线的 CST 系数（b∈R^6, c∈R^4, dz_te）
  2) 14 维设计要素（供后续Sobol与优化）
  3) 一次 Sobol 敏感性报告（支持 SALib，缺失则回退LHS+近似）
  4) “后续优化变量清单” 与建议取值范围（预设 + 按敏感性Top-K）
  5) 预览图与CSV/JSON

【依赖/要求】
- Python 3.10+；允许使用：numpy, scipy, pandas, matplotlib；可选 SALib（存在则用，不存在要优雅回退并给提示）。
- 结构模块化，函数具备类型注解与docstring；提供命令行入口 `aflow.py`。
- 不实现任何优化算法；仅生成“后续优化变量清单”。

【几何与CST表达（方案A：厚度/弯度分解）】
- 类函数固定：C(x)=x^0.5*(1-x)^1.0
- 厚度：t(x)=C(x)*Σ_{i=0..5} b_i*B_i^5(x) + x*dz_te
- 弯度：z(x)=C(x)*Σ_{j=0..3} c_j*B_j^3(x)
- 上/下表面：y_u=z+0.5 t，y_l=z-0.5 t
- 实现Bernstein及导数：dB_i^n/dx = n*(B_{i-1}^{n-1} - B_i^{n-1})

【I/O与归一化】
- 支持 Selig .dat（上表面后缘→前缘→下表面前缘→后缘）与 CSV（x,y 或 xu,yu,xl,yl）。
- 归一化：自动平移/旋转使TE在x=1, LE在x=0；弦长=1；保证x单调。
- 重采样：余弦聚点网格 x∈(0,1)，默认N=200（避开0与1）。

【拟合与派生量】
- 由重采样曲线得到 t(x)=yu-yl, z(x)=(yu+yl)/2。
- 用带微弱Tikhonov正则的最小二乘拟合 b,c,dz_te。
- 计算派生几何量：极值/位置、导数、曲率等。

【14维设计要素（全部导出）】
核心7维：
1) t_max（最大厚度）
2) x_t  （最大厚度位置）
3) f_max（最大弯度）
4) x_f  （最大弯度位置）
5) r_le_hat（前缘半径估计：在 x∈{0.003,0.007,0.015,0.03} 拟合 t/2≈a√x+bx，r_le_hat=a^2/2）
6) dz_te = t(1)（尾缘实厚）
7) s_rec = t(0.6) - t(0.9)（压力恢复强度；越小越温和）

新增7维（位置化/斜率化）：
8)  t_015 = t(0.15)
9)  t_050 = t(0.50)
10) t_075 = t(0.75)
11) dt_080 = t'(0.80)
12) dz_005 = z'(0.05)
13) dz_090 = z'(0.90)
14) r_fx = z(0.90) - [ z(0.70) + 0.20*(z(0.95)-z(0.70)) ]

【敏感性分析（必须包含）】
- 若检测到 SALib：用Saltelli采样+Sobol一阶/总效应指数；否则回退到LHS采样+方差分解近似（并在日志中明确提示）。
- 评估函数：先内置“几何代理得分” geom_score(features)（无需XFoil）：
  geom_score = - w_rec*|s_rec - s_rec_base|
               - w_le*|r_le_hat - r_le_base|
               - w_mono*penalty_mono_backhalf
               - w_tmax*|t_max - t_max_base|
  其中 base 来自输入翼型的14维特征，penalty_mono_backhalf 对 x≥0.5 的 dt/dx>0 加罚。
- 允许通过 `--evaluator geom`（默认）或 `--evaluator plugin:my_eval.py` 注入自定义评估器（为后续接入XFoil/代理预留钩子）。
- 参数空间（特征变量的范围）：以“基线特征±带宽”的方式自动生成（默认带宽见下）；也支持从 `--ranges ranges.json` 读取覆盖。
  默认带宽（可写到代码常量里）：
  t_max: ±10%；x_t: ±0.05；f_max: ±40%；x_f: ±0.08；
  r_le_hat: ×[0.9,1.2]；dz_te: [0.002,0.004]（绝对）
  s_rec: ±0.03；t_015: ±0.015；t_050: ±0.02；t_075: ±0.02；
  dt_080: [+0.00, -0.20]（保持负值，给区间[-0.30,-0.05]内收敛）
  dz_005: [0.00, 0.12]；dz_090: [-0.10, 0.05]；r_fx: ±0.004
- 命令行样例：`python aflow.py sens --in airfoil.dat --n_base 200 --evaluator geom`
- 输出：`sensitivity.csv`（Si, ST, conf区间）、`sensitivity.png`（条形图）。

【后续优化变量清单（本次仅生成配置，不做优化）】
- 生成 `opt_config.json`，包含：
  - "opt_vars_recommended": 预设变量列表：
    ["t_max","x_t","f_max","x_f","r_le_hat","dz_te","s_rec","t_050","dt_080","dz_090"]
  - "opt_vars_by_sensitivity": 运行敏感性后，自动选取总效应指数 ST Top-K（默认K=8）的变量名数组（保存两版：Top-K与阈值法 ST>0.05）。
  - "bounds": 各变量的上下界（即上面“参数空间”最终生效的范围）。
  - "notes": 说明本文件供后续优化器直接读取。

【健康检查（失败打印警告但不中断导出）】
- t(x) ≥ dz_te 全弦；x≥0.5 区域 dt/dx ≤ 0（允许小阈值）
- r_le_hat > 0；x_t,x_f ∈ (0,1) 且能被稳健求出

【命令行接口】
- `python aflow.py extract --in file.dat --out outdir`   # 读入→归一→CST拟合→导出14维+系数+预览图
- `python aflow.py sens    --in file.dat --out outdir --n_base 200 [--evaluator geom|plugin:xxx.py] [--ranges ranges.json] [--topk 8]`

【快速开始】
1. 创建并激活虚拟环境：`python -m venv .venv && source .venv/bin/activate`（Windows 使用 `.\.venv\Scripts\activate`）。
2. 安装依赖：`pip install -r requirements.txt`。SALib 为可选依赖，安装后可获得 Sobol 指数；未安装时程序会自动降级到 LHS 近似。
3. 运行提取流程：`python aflow.py extract --in data/NACA4412.txt --out outputs/extract`（或使用 `.dat` 版本）。
4. 在同一输出目录执行敏感性分析：`python aflow.py sens --in data/NACA4412.txt --out outputs/sens`。

【导出文件】
- `features.json`（14维+派生量）/ `cst_coeffs.json`（b0..b5,c0..c3,dz_te）
- `coords_resampled.csv`（x,yu,yl,t,z）
- `preview.png`（翼型、厚度/弯度、LE/TE放大、特征位置标注）
- `sensitivity.csv`, `sensitivity.png`
- `opt_config.json`

【函数/模块建议】
- io_utils.py: load_airfoil(), normalize_chord(), resample_cosine()
- cst.py: bernstein(), class_fn(), cst_thickness(), cst_camber(), airfoil_coords()
- fit.py: fit_cst_thickness(), fit_cst_camber()
- features.py: compute_features_14(), find_extrema(), le_radius_fit(), monotonicity_penalty()
- sensitivity.py: build_ranges(), sample_saltelli_or_lhs(), run_sobol(), plot_bars()
- aflow.py: CLI入口（extract/sens），组织调用并写出文件

【验收自检】
1) 读取对称翼型应得到 f_max≈0，且 x_f 合理；重建误差(RMSE) < 1e-3 量级。
2) features.json 含齐全14项；cst_coeffs.json 含 b0..b5,c0..c3,dz_te。
3) 无 SALib 时能回退；有 SALib 时输出一阶/总效应指数。
4) 生成 opt_config.json，且 `opt_vars_by_sensitivity` 与图表一致（Top-K变量名称匹配）。
5) CLI 两个子命令均可运行；输出文件齐备。

【不要做】
- 不要把 N1,N2 当变量；不要实现任何优化算法；不要访问网络；不要引入额外第三方库（SALib可选）。

