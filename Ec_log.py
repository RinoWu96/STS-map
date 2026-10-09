#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ec_0to1V_log.py

专门用于只有 0–1 V 范围 STS 数据的导带底 Ec 提取：
- 使用 log10(|I|) plateau + (base + Δdecade) 阈值
- 只找正电压侧的导带底 Ec，不尝试计算完整带隙 Eg
- 适合 *.dat 扫描（Omicron STS .dat 格式）

使用方式：
    python Ec_0to1V_log.py
然后选择包含 *.dat 的文件夹。

输出：
- 左图：Lock-in X (dI/dV) map
- 右图：Ec_raw / Ec_smoothed 随 position (nm) 的变化
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_interp_spline
import tkinter as tk
from tkinter import filedialog
import re
from band_edge_core import detect_band_edges, apply_gn_normalization, apply_vertical_shift
from newBG import read_sts_dat, plot_spline


# ================================================================
# log(|I|) 方法：在 0–1 V 范围内找到 Ec
# ================================================================
def ec_from_IV_log(
    V,
    I,
    li_data=None,
    v_center_min=0.0,
    v_center_max=0.2,
    delta_decade=0.95,
    min_pos_voltage=0.02,
    sigma_multiplier=0.0,
    sigma_v_min=-1.0,
    sigma_v_max=0.2,
):
    V = np.asarray(V, dtype=float)
    I = np.asarray(I, dtype=float)

    idx = np.argsort(V)
    V = V[idx]
    I = I[idx]
    if li_data is not None:
        li_data = np.asarray(li_data, dtype=float)[idx]

    # 计算 log10(|I|) 或 log10(LI)
    if li_data is not None:
        LI_shifted, _ = apply_vertical_shift(V, li_data, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
        logI = np.log10(np.maximum(LI_shifted, 1e-20))
    else:
        I_shifted, _ = apply_vertical_shift(V, I, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
        logI = np.log10(np.maximum(I_shifted, 1e-20))

    # plateau 在靠近 0 V 的小窗口
    center_mask = (V >= v_center_min) & (V <= v_center_max)
    if center_mask.sum() < 5:
        # 若点太少，就用中间 20% 兜底
        n = len(V)
        i1 = int(n * 0.4)
        i2 = int(n * 0.6)
        center_mask = np.zeros_like(V, dtype=bool)
        center_mask[i1:i2] = True

    base = np.median(logI[center_mask])
    threshold = base + delta_decade

    # 只在正电压、且大于某个 Vmin 的范围内找 Ec
    mask_cb = (V > min_pos_voltage) & (logI > threshold)
    if mask_cb.any():
        idx_cb_all = np.where(mask_cb)[0]

        # 连续三个点都高于阈值
        potential_cb_indices = []
        for idx0 in idx_cb_all:
            if idx0 >= 1 and idx0 < (len(logI) - 1):
                if (logI[idx0 - 1] > threshold) and (logI[idx0] > threshold) and (logI[idx0 + 1] > threshold):
                    potential_cb_indices.append(idx0)

        if potential_cb_indices:
            diffs = [abs(logI[i] - threshold) for i in potential_cb_indices]
            idx_cb = potential_cb_indices[int(np.argmin(diffs))]
        else:
            idx_cb = int(idx_cb_all.min())

        Ec = float(V[idx_cb])
    else:
        Ec = np.nan

    return Ec, base, threshold, V, logI


# ================================================================
# 简单的 1D 邻域平滑
# ================================================================
def smooth_line_robust(values, half_window=3):
    """对 1D 数组做简单邻域平滑（忽略 NaN）。"""
    values = np.asarray(values, dtype=float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n - 1, i + half_window)
        seg = values[lo:hi+1]
        seg = seg[np.isfinite(seg)]
        if seg.size > 0:
            out[i] = seg.mean()

    # 没有平滑值的点保持原始值
    mask_nan = np.isnan(out)
    out[mask_nan] = values[mask_nan]
    return out


# ================================================================
# 处理 *.dat 扫描
# ================================================================
def process_line_folder_0to1V(
    folder,
    pattern="*.dat",
    method="Adaptive Thresholding",
    sigma_level=2.0,
    fit_window=0.05,
    smooth_window=11,
    v_center_min=None,
    v_center_max=None,
    delta_E=0.15,
    sigma_map_bias=1.0,
    sigma_map_pos=1.0,
    smooth_half_window=3,
    use_auto_params=True,
    slope_threshold=0.5,
    bias_col="Bias calc (V)",
    curr_col="Current (A)",
    lix_col="LI Demod 1 X (A)",
    log_li=False,
    gn_norm=False,
    lam_factor=5.0,
    sigma_multiplier=0.0,
    sigma_v_min=-1.0,
    sigma_v_max=0.2,
    v_display_min=None,
    v_display_max=None,
    **kwargs
):
    """在指定 folder 中寻找 *.dat 文件，按多种方法提取 Ec(x)。
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)), key=lambda x: int(re.search(r'(\d+)\.dat$', os.path.basename(x)).group(1)) if re.search(r'(\d+)\.dat$', os.path.basename(x)) else 0)
    if not paths:
        raise RuntimeError(f"No files found in {folder} with pattern {pattern!r}")

    coords = []
    Vs_list = []
    I_list = []
    LIx_list = []

    for p in paths:
        df, meta = read_sts_dat(Path(p), bias_col=bias_col, curr_col=curr_col, lix_col=lix_col)
        # 即使缺失坐标也继续，后面会退化为索引
        coords.append((meta["x"], meta["y"], meta["z"]))

        V = df[bias_col].to_numpy()
        I = df[curr_col].to_numpy() if curr_col in df.columns else np.zeros_like(V)
        LIx = df[lix_col].to_numpy()

        s = np.argsort(V)
        V = V[s]
        I = I[s]
        LIx = LIx[s]

        # 快速归一化: gN = (dI/dV) / (I/V)
        if gn_norm:
            # 使用 Soft Regularization (软正则)
            LIx = apply_gn_normalization(V, I, LIx, lam_factor=lam_factor)

        Vs_list.append(V)
        I_list.append(I)
        LIx_list.append(LIx)

    # Bias 网格
    Vgrid = Vs_list[0]
    Nb = len(Vgrid)
    Np = len(paths)

    # 构造均匀的 dI/dV 矩阵 (imap)
    imap = np.zeros((Nb, Np), dtype=float)
    for i in range(Np):
        v_arr = np.array(Vs_list[i], dtype=float)
        lix_arr = np.array(LIx_list[i], dtype=float)
        if len(v_arr) == Nb and np.allclose(v_arr, Vgrid):
            imap[:, i] = lix_arr
        else:
            imap[:, i] = np.interp(Vgrid, v_arr, lix_arr)

    # 构造均匀的 I 矩阵 (I_map)
    I_map = np.zeros((Nb, Np), dtype=float)
    for i in range(Np):
        v_arr = np.array(Vs_list[i], dtype=float)
        i_arr = np.array(I_list[i], dtype=float)
        if len(v_arr) == Nb and np.allclose(v_arr, Vgrid):
            I_map[:, i] = i_arr
        else:
            I_map[:, i] = np.interp(Vgrid, v_arr, i_arr)

    # 对均匀矩阵进行高斯平滑
    imap_s = gaussian_filter(imap, sigma=(sigma_map_bias, sigma_map_pos))

    # 位置 (nm)
    coords_arr = np.array(coords, dtype=float)
    if np.isfinite(coords_arr).any() and len(coords_arr) > 0:
        if np.isnan(coords_arr).any():
            dists = np.arange(len(paths), dtype=float)
        else:
            if len(coords_arr) > 1:
                steps = np.linalg.norm(np.diff(coords_arr, axis=0), axis=1)
                dists = np.concatenate(([0.0], np.cumsum(steps)))
            else:
                dists = np.array([0.0], dtype=float)
    else:
        dists = np.arange(len(paths), dtype=float)
    pos_nm = dists * 1e9 if not np.isnan(coords_arr).any() else dists

    Ec_raw = []
    for V, I, LIx in zip(Vs_list, I_list, LIx_list):
        try:
            Ev, Ec, Eg, _ = detect_band_edges(
                V, LIx, 
                method=method,
                mode="Ec", 
                min_gap=0.05,
                sigma_level=sigma_level,
                fit_window=fit_window,
                smooth_window=smooth_window,
                v_center_min=v_center_min if v_center_min is not None else -0.3,
                v_center_max=v_center_max if v_center_max is not None else 0.3,
                delta_E=delta_E,
                slope_threshold=slope_threshold,
                sigma_multiplier=sigma_multiplier,
                sigma_v_min=sigma_v_min,
                sigma_v_max=sigma_v_max,
                **kwargs
            )
            if np.isnan(Ec) and (method == "Adaptive Thresholding" or method == "Dynamic Threshold"):
                Ev, Ec, Eg, _ = detect_band_edges(
                    V, LIx, 
                    method=method,
                    mode="Ec", 
                    min_gap=0.05, 
                    slope_percentile=70, 
                    edge_consistency_points=2,
                    v_center_min=v_center_min if v_center_min is not None else -0.3,
                    v_center_max=v_center_max if v_center_max is not None else 0.3,
                    delta_E=delta_E,
                    slope_threshold=slope_threshold,
                    sigma_multiplier=sigma_multiplier,
                    sigma_v_min=sigma_v_min,
                    sigma_v_max=sigma_v_max,
                    **kwargs
                )
            
            # 如果还是 NaN，且是 Adaptive Thresholding，才尝试最后的兜底
            if np.isnan(Ec) and (method == "Adaptive Thresholding" or method == "Dynamic Threshold"):
                Ec2, *_ = ec_from_IV_log(V, I, li_data=LIx, v_center_min=0.0, v_center_max=0.2, delta_decade=0.95, min_pos_voltage=0.02, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
                if np.isfinite(Ec2):
                    Ec = Ec2
            Ec_raw.append(Ec if Ec is not None else np.nan)
        except Exception:
            Ec_raw.append(np.nan)

    Ec_raw = np.asarray(Ec_raw, dtype=float)
    Ec_s = smooth_line_robust(Ec_raw, half_window=smooth_half_window)

    # 画图
    # --- 图1: dI/dV Map ---
    fig1 = plt.figure(figsize=(6, 4.5))
    ax1 = fig1.add_subplot(111)
    extent = [pos_nm.min(), pos_nm.max(), Vgrid.min(), Vgrid.max()]
    
    # 应用 log 转换（如果选中）
    imap_plot = imap_s
    cb_label = "gN (arb. u.)" if gn_norm else "LI Demod 1 X (A)"
    if log_li:
        # 使用垂直平移优化：基于 Vgrid 的指定范围
        imap_shifted, _ = apply_vertical_shift(Vgrid, imap_s, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
        imap_plot = np.log10(np.maximum(imap_shifted, 1e-20))
        cb_label = f"log10({cb_label} + offset)"
        
    im = ax1.imshow(
        imap_plot,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="jet",
        interpolation="bilinear",
    )
    title_suffix = f" (gN, λ factor={lam_factor})" if gn_norm else ""
    ax1.set_title(f"Line STS Lock-in X Map{title_suffix}")
    ax1.set_xlabel("Position (nm)")
    ax1.set_ylabel("Bias (V)")
    
    # 自动适配 Y 轴范围
    if v_display_min is not None and v_display_max is not None:
        ax1.set_ylim(v_display_min, v_display_max)
        
    plt.colorbar(im, ax=ax1, label=cb_label)
    fig1.tight_layout()

    # --- 图2: Ec vs position ---
    fig2 = plt.figure(figsize=(6, 4.5))
    ax2 = fig2.add_subplot(111)
    ax2.plot(pos_nm, Ec_raw, "o", alpha=0.3, label="Ec raw", markersize=3)
    plot_spline(ax2, pos_nm, Ec_s, "-", linewidth=2, label="Ec smoothed")
    ax2.set_title("Conduction band onset Ec from log10(|I|)")
    ax2.set_xlabel("Position (nm)")
    ax2.set_ylabel("Ec (V)")
    
    # 自动适配 Y 轴范围
    if v_display_min is not None and v_display_max is not None:
        ax2.set_ylim(v_display_min, v_display_max)
        
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig2.tight_layout()

    return {
        "Vgrid": Vgrid,
        "pos_nm": pos_nm,
        "imap": imap,      # 返回原始 map
        "imap_s": imap_s,  # 返回平滑后的 map
        "Ec_raw": Ec_raw,
        "Ec_s": Ec_s,
        "paths": paths,
        "I_map": I_map,
        "figs": [fig1, fig2],
    }


# ================================================================
# 脚本入口
# ================================================================
def main():
    root = tk.Tk()
    root.withdraw()

    folder = filedialog.askdirectory(title="选择包含 *.dat 的文件夹")
    if not folder:
        print("未选择文件夹，退出。")
        return

    try:
        _ = process_line_folder_0to1V(
            folder=folder,
            pattern="*.dat",
            method="Dynamic Threshold",
            sigma_level=2.0,
            fit_window=0.05,
            smooth_window=11,
            v_center_min=None,
            v_center_max=None,
            delta_E=0.15,
            sigma_map_bias=1.0,
            sigma_map_pos=1.0,
            smooth_half_window=3,
            use_auto_params=True,
        )
        plt.show()
    except Exception as e:
        print("处理过程中出现错误:", e)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
