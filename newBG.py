#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STS line scan (log |LI X| plateau band-edge extraction only)

功能：
- 读取一条 STS line（同一文件夹下的一系列 Omicron .dat 文件）
- 对于每一个数据点，计算带边
- 返回 dI/dV map 和带边结果 DataFrame，方便在你自己的 GUI 里调用。

核心：
- 统一调用核心算法，直接计算 Ev/Ec/Eg
"""

from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import make_interp_spline
import re
from band_edge_core import detect_band_edges, apply_gn_normalization, apply_vertical_shift


# ---------------------------------------------------------------------
# 绘图辅助工具：使用 Spline 拟合曲线
# ---------------------------------------------------------------------
def plot_spline(ax, x, y, *args, **kwargs):
    """
    使用 spline 拟合曲线并绘制。
    如果点数不足，退化为普通 plot。
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 去除 NaN 和无穷大
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 4:
        return ax.plot(x, y, *args, **kwargs)
    
    # 确保 x 是升序
    idx = np.argsort(x_clean)
    x_sorted = x_clean[idx]
    y_sorted = y_clean[idx]
    
    # 确保 x 的唯一性
    x_unique, unique_indices = np.unique(x_sorted, return_index=True)
    y_unique = y_sorted[unique_indices]
    
    if len(x_unique) < 4:
        return ax.plot(x, y, *args, **kwargs)
    
    try:
        # 生成平滑曲线点
        x_new = np.linspace(x_unique.min(), x_unique.max(), 300)
        spl = make_interp_spline(x_unique, y_unique, k=3)
        y_smooth = spl(x_new)
        return ax.plot(x_new, y_smooth, *args, **kwargs)
    except Exception:
        # 如果拟合失败，回退到原始绘图
        return ax.plot(x, y, *args, **kwargs)


# ---------------------------------------------------------------------
# 读单个 .dat 文件
# ---------------------------------------------------------------------
def read_sts_dat(path: Path, bias_col="Bias calc (V)", curr_col="Current (A)", lix_col="LI Demod 1 X (A)"):
    """
    读取 Omicron STS .dat 文件，返回 (df, meta)。

    df: pandas.DataFrame，包含 Bias, Current, LI Demod 等列
    meta: dict，包含 x, y, z (以 m 为单位) 以及文件名
    """
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        lines = f.read().splitlines()

    x = y = z = None
    data_start = None

    for i, s in enumerate(lines):
        t = s.strip()
        if t == "[DATA]":
            data_start = i + 1
            break

        # header 中的坐标行：X (m) / Y (m) / Z (m)
        if t.startswith("X (m)") or t.startswith("Y (m)") or (
            t.startswith("Z (m)") and "offset" not in t
        ):
            parts = t.split()
            if len(parts) >= 2:
                token = parts[-1]
                try:
                    val = float(token.replace(",", "."))
                except ValueError:
                    continue
                if t.startswith("X (m)"):
                    x = val
                elif t.startswith("Y (m)"):
                    y = val
                elif t.startswith("Z (m)"):
                    z = val

    if data_start is None:
        raise ValueError(f"No [DATA] section in {path}")

    # 数据真正开始的地方是在 [DATA] 行之后
    # 第一行通常就是列标题
    data_lines = lines[data_start:]
    # 过滤掉可能的空行，找到第一行作为表头
    header_idx = 0
    for i, line in enumerate(data_lines):
        if line.strip():
            header_idx = i
            break
            
    data_str = "\n".join(data_lines[header_idx:])
    # 对于 Omicron 文件，优先使用 Tab 分隔符，因为列名本身包含空格
    try:
        # 先尝试 Tab 分隔
        df = pd.read_csv(StringIO(data_str), sep="\t")
        # 如果只读出了一列且列名中包含空格，说明可能不是 Tab 分隔，再尝试自动检测
        if len(df.columns) <= 1:
            df = pd.read_csv(StringIO(data_str), sep=None, engine='python')
    except Exception:
        # 最后的回退逻辑
        df = pd.read_csv(StringIO(data_str), sep=None, engine='python')

    meta = {"x": x, "y": y, "z": z, "filename": path.name}
    return df, meta


# ---------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------
def smooth_curve(y, window: int = 11):
    """简单 moving-average 平滑。window 必须为奇数。"""
    if window <= 1:
        return y.copy()
    kernel = np.ones(window) / float(window)
    return np.convolve(y, kernel, mode="same")


# ---------------------------------------------------------------------
# 主函数：处理整条 line 的文件夹
# ---------------------------------------------------------------------
def process_line_folder(
    folder: Path,
    pattern: str = "*.dat",
    method: str = "Adaptive Thresholding",
    eg_min: float = 0.3,
    eg_max: float = 3.5,
    v_center_min=None,
    v_center_max=None,
    delta_E=None,
    min_gap=None,
    sigma_level=2.0,
    fit_window=0.05,
    smooth_window=11,
    use_auto_params: bool = True,
    slope_threshold: float = 0.5,
    bias_col="Bias calc (V)",
    curr_col="Current (A)",
    lix_col="LI Demod 1 X (A)",
    sigma_map_bias: float = 1.0,
    sigma_map_pos: float = 1.0,
    gn_norm: bool = False,
    lam_factor: float = 5.0,
    sigma_multiplier: float = 0.0,
    sigma_v_min: float = -1.0,
    sigma_v_max: float = 0.2,
    **kwargs
):
    """
    处理给定文件夹中的所有 STS .dat 文件，返回：
        bias      : 1D numpy array, Bias 轴 (V)
        pos_nm    : 1D numpy array, 位置 (nm)，以起点为 0
        M         : 2D numpy array, shape=(Nbias, Npoints)，dI/dV map
        df_res    : pandas.DataFrame，包含 Ev / Ec / Eg 等结果
    """
    folder = Path(folder)
    files = sorted(folder.glob(pattern), key=lambda x: int(re.search(r'(\d+)\.dat$', x.name).group(1)) if re.search(r'(\d+)\.dat$', x.name) else 0)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {folder}")

    # ------ 先读入所有曲线，构建 map 用的 g_list 和坐标 ------
    V_list = []
    g_list = []
    I_list = []
    x_nm = []
    y_nm = []
    z_nm = []
    names = []

    for p in files:
        df, meta = read_sts_dat(p, bias_col=bias_col, curr_col=curr_col, lix_col=lix_col)
        V = df[bias_col].to_numpy()
        g = df[lix_col].to_numpy()
        I = df[curr_col].to_numpy() if curr_col in df.columns else None

        # 按电压排序，确保 V 单调
        idx = np.argsort(V)
        V = V[idx]
        g = g[idx]
        if I is not None:
            I = I[idx]

        # 快速归一化: gN = (dI/dV) / (I/V)
        if gn_norm and I is not None:
            g = apply_gn_normalization(V, I, g, lam_factor=lam_factor)

        V_list.append(V)
        g_list.append(g)
        I_list.append(I)

        x_nm.append(meta["x"] * 1e9 if meta["x"] is not None else np.nan)
        y_nm.append(meta["y"] * 1e9 if meta["y"] is not None else np.nan)
        z_nm.append(meta["z"] * 1e9 if meta["z"] is not None else np.nan)
        names.append(meta["filename"])

    V_list = np.array(V_list, dtype=object)
    g_list = np.array(g_list, dtype=object)
    I_list = np.array(I_list, dtype=object)
    x_nm = np.array(x_nm, dtype=float)
    y_nm = np.array(y_nm, dtype=float)
    z_nm = np.array(z_nm, dtype=float)

    # 用第一条曲线作为统一 Bias 轴，其他条插值到这条上
    bias = V_list[0]
    Nb = len(bias)
    Np = len(files)

    G_matrix = np.zeros((Np, Nb), dtype=float)
    for i in range(Np):
        v_array = np.array(V_list[i], dtype=float)
        bias_array = np.array(bias, dtype=float)
        g_array = np.array(g_list[i], dtype=float)
        
        if len(v_array) == 0 or len(bias_array) == 0:
            continue
            
        if len(v_array) == Nb and np.allclose(v_array, bias_array):
            G_matrix[i, :] = g_array
        else:
            G_matrix[i, :] = np.interp(bias_array, v_array, g_array)

    # 位置：使用 Ec_log 的累计距离算法 (基于 X/Y/Z)
    coords = np.column_stack([x_nm, y_nm, z_nm])
    if np.isfinite(coords).any() and len(coords) > 0:
        # 如果至少有一个点是有限的，我们尝试处理
        # 先处理可能的 NaN (用第一个有限点填充)
        if np.isnan(coords).any():
            # 简单处理：如果 X/Y/Z 确实没读到，退化为索引
            pos_nm = np.arange(Np, dtype=float)
        else:
            if len(coords) > 1:
                steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
                pos_nm = np.concatenate(([0.0], np.cumsum(steps)))
            else:
                pos_nm = np.array([0.0], dtype=float)
    else:
        # 完全没有坐标信息，退化为 0, 1, 2, ...
        pos_nm = np.arange(Np, dtype=float)

    # ------ 计算 band edges（log 方法, 每点自动微调） ------
    Ev_list = []
    Ec_list = []
    Eg_list = []

    for i in range(Np):
        V_curve = V_list[i]
        g_curve = g_list[i]  # LI 数据
        try:
            from band_edge_core import detect_band_edges
            Ev, Ec, Eg, _ = detect_band_edges(
                V_curve,
                g_curve,
                method=method,
                mode="both",
                min_gap=float(eg_min),
                v_center_min=v_center_min if v_center_min is not None else -0.3,
                v_center_max=v_center_max if v_center_max is not None else 0.3,
                delta_E=delta_E if delta_E is not None else 0.15,
                sigma_level=sigma_level,
                fit_window=fit_window,
                smooth_window=smooth_window,
                slope_threshold=slope_threshold,
                sigma_multiplier=sigma_multiplier,
                sigma_v_min=sigma_v_min,
                sigma_v_max=sigma_v_max,
                **kwargs
            )
        except Exception:
            Ev, Ec, Eg = np.nan, np.nan, np.nan
        Ev_list.append(Ev)
        Ec_list.append(Ec)
        Eg_list.append(Eg)

    Ev_arr = np.array(Ev_list, dtype=float)
    Ec_arr = np.array(Ec_list, dtype=float)
    Eg_arr = np.array(Eg_list, dtype=float)

    # 保存原始数据用于绘图
    Ev_raw = Ev_arr.copy()
    Ec_raw = Ec_arr.copy()
    Eg_raw = Eg_arr.copy()

    # 物理范围过滤 + 空间平滑
    good = (~np.isnan(Eg_arr)) & (Eg_arr > eg_min) & (Eg_arr < eg_max)

    Ev_ref = Ev_arr.copy()
    Ec_ref = Ec_arr.copy()
    Eg_ref = Eg_arr.copy()
    for i in range(Np):
        if not good[i]:
            lo = max(0, i - 2)
            hi = min(Np - 1, i + 2)
            neigh = np.where(good[lo : hi + 1])[0]
            if len(neigh) > 0:
                Ev_ref[i] = np.median(Ev_arr[lo + neigh])
                Ec_ref[i] = np.median(Ec_arr[lo + neigh])
                Eg_ref[i] = np.median(Eg_arr[lo + neigh])
    Ev_arr = Ev_ref
    Ec_arr = Ec_ref
    Eg_arr = Eg_ref

    # 再做一次高斯平滑
    valid = ~np.isnan(Ev_arr)
    if valid.sum() > 3:
        Ev_arr[valid] = gaussian_filter1d(Ev_arr[valid], sigma=1.0, mode="nearest")
        Ec_arr[valid] = gaussian_filter1d(Ec_arr[valid], sigma=1.0, mode="nearest")
        Eg_arr[valid] = gaussian_filter1d(Eg_arr[valid], sigma=1.0, mode="nearest")

    # dI/dV map 按照 Ec_log 的逻辑，使用 2D 高斯平滑
    imap = G_matrix.T  # shape (Nbias, Npos)
    gmap = gaussian_filter(imap, sigma=(sigma_map_bias, sigma_map_pos))

    # 构造结果 DataFrame
    df_res = pd.DataFrame(
        {
            "filename": names,
            "x_nm": x_nm,
            "y_nm": y_nm,
            "z_nm": z_nm,
            "pos_nm": pos_nm,
            "Ev": Ev_arr,
            "Ec": Ec_arr,
            "Eg": Eg_arr,
            "Ev_raw": Ev_raw,
            "Ec_raw": Ec_raw,
            "Eg_raw": Eg_raw,
        }
    )

    # I map
    I_matrix = np.full((Np, Nb), np.nan)
    for i, I_vec in enumerate(I_list):
        if I_vec is None: continue
        v_array = np.array(V_list[i], dtype=float)
        bias_array = np.array(bias, dtype=float)
        i_array = np.array(I_vec, dtype=float)
        
        if len(v_array) == Nb and np.allclose(v_array, bias_array):
            I_matrix[i, :] = i_array
        else:
            I_matrix[i, :] = np.interp(bias_array, v_array, i_array)
    imap_i = I_matrix.T

    return bias, pos_nm, gmap, df_res, imap, imap_i  # 返回 imap (原始), gmap (平滑), imap_i (I_raw)


# ---------------------------------------------------------------------
# 画图函数（可选）
# ---------------------------------------------------------------------
def plot_final(bias, pos_nm, M, df_res, title_prefix="", show_gap_curve=True, log_li=False, gn_norm=False, lam_factor=5.0, sigma_multiplier=0.0, sigma_v_min=-1.0, sigma_v_max=0.2, v_display_min=None, v_display_max=None, vmin=None, vmax=None):
    """
    绘图：返回 [fig1, fig2] 两个独立的 Figure 对象，尺寸固定为用于发表的 6x4.5 英寸 (4:3)
    """
    # --- 图1: dI/dV Map ---
    fig1 = plt.figure(figsize=(6, 4.5))
    ax1 = fig1.add_subplot(111)
    extent = [pos_nm.min(), pos_nm.max(), bias.min(), bias.max()]

    # 应用 log 转换（如果选中）
    M_plot = M
    cb_label = "gN (arb. u.)" if gn_norm else "dI/dV (A/V)"
    if log_li:
        # 使用垂直平移优化：基于 bias 的指定范围
        M_shifted, _ = apply_vertical_shift(bias, M, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
        M_plot = np.log10(np.maximum(M_shifted, 1e-20))
        cb_label = f"log10({cb_label} + offset)"

    im = ax1.imshow(
        M_plot,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="jet",
        interpolation="bilinear",
        vmin=vmin,
        vmax=vmax,
    )

    # 自动适配 Y 轴范围
    if v_display_min is not None and v_display_max is not None:
        ax1.set_ylim(v_display_min, v_display_max)

    title_suffix = f" (gN, λ factor={lam_factor})" if gn_norm else ""
    ax1.set_title(f"{title_prefix}dI/dV Map{title_suffix}")
    ax1.set_xlabel("Position (nm)")
    ax1.set_ylabel("Bias (V)")
    plt.colorbar(im, ax=ax1, label=cb_label)
    fig1.tight_layout()

    # --- 图2: Ev/Ec/Eg Curves ---
    fig2 = plt.figure(figsize=(6, 4.5))
    ax2 = fig2.add_subplot(111)
    
    # Ev: 土黄色 (darkgoldenrod), 原始数据用点, 拟合用线
    ax2.plot(df_res["pos_nm"], df_res["Ev_raw"], "o", color="darkgoldenrod", alpha=0.3, markersize=3)
    plot_spline(ax2, df_res["pos_nm"], df_res["Ev"], "-", color="darkgoldenrod", label="Ev", linewidth=1.5)
    
    # Ec: 蓝色 (blue), 原始数据用点, 拟合用线
    ax2.plot(df_res["pos_nm"], df_res["Ec_raw"], "bo", alpha=0.3, markersize=3)
    plot_spline(ax2, df_res["pos_nm"], df_res["Ec"], "b-", label="Ec", linewidth=1.5)

    # 自动适配 Y 轴范围
    if v_display_min is not None and v_display_max is not None:
        ax2.set_ylim(v_display_min, v_display_max)
    
    if show_gap_curve:
        ax2_twin = ax2.twinx()
        # Eg: 红色 (red), 原始数据用实心三角, 拟合用线
        ax2_twin.plot(df_res["pos_nm"], df_res["Eg_raw"], "^", color="red", alpha=0.3, markersize=4)
        plot_spline(ax2_twin, df_res["pos_nm"], df_res["Eg"], "-", color="red", label="Eg", linewidth=1.5)
        ax2_twin.set_ylabel("Band Gap (eV)", color="red")
        ax2_twin.tick_params(axis='y', labelcolor='red')

    ax2.set_title(f"{title_prefix} Band Edges vs Position")
    ax2.set_xlabel("Position (nm)")
    ax2.set_ylabel("Energy (eV)")
    
    # 合并两个坐标轴的图例
    lines, labels = ax2.get_legend_handles_labels()
    if show_gap_curve:
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper left")
    else:
        ax2.legend(loc="upper left")
        
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    return [fig1, fig2]
