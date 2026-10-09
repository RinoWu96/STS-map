#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Line_CBM_VBM.py - 线扫描STM数据导带底(CBM)和价带顶(VBM)自动识别工具

功能：
- 读取 line*.dat 扫描数据（Omicron STS 格式）
- 对每条线自动识别导带底和价带顶位置
- 使用改进的自动参数估算法（仿照GUI_log.py中的方法）
- 输出带导带底和价带顶标记的可视化图表
- 保存导带底和价带顶位置数据到文本文件
- 在图表中用虚线绘制导带底和价带顶位置
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
from tkinter import messagebox
import newBG as newbg_mod
from newBG import plot_spline
from band_edge_core import detect_band_edges, apply_vertical_shift
from Line_Log import read_dat_file, process_all_files

def linear_fit_with_R2(x, y):
    """线性拟合并计算R²值"""
    if len(x) < 3:
        return None, None, None
    p = np.polyfit(x, y, 1)
    y_pred = np.polyval(p, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return p[0], p[1], float(R2)


def save_band_edge_results(cbm_bias_list, vbm_bias_list, file_names, save_directory='.', use_filtered=False):
    """把每条线的导带底和价带顶结果写入一个 txt 文件"""
    filename_suffix = "_filtered" if use_filtered else ""
    out_path = os.path.join(save_directory, f"band_edge_results{filename_suffix}.txt")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# Band edge detection results\n")
        f.write("# LineIndex\tFileName\tCBM_Bias(V)\tVBM_Bias(V)\tBandGap(V)\n")
        for i, (cbm_val, vbm_val, fname) in enumerate(zip(cbm_bias_list, vbm_bias_list, file_names), start=1):
            if cbm_val is None or vbm_val is None:
                bg_val = None
            else:
                bg_val = cbm_val - vbm_val
                
            if cbm_val is None:
                cbm_str = "NaN"
            else:
                cbm_str = f"{cbm_val:.6f}"
                
            if vbm_val is None:
                vbm_str = "NaN"
            else:
                vbm_str = f"{vbm_val:.6f}"
                
            if bg_val is None:
                bg_str = "NaN"
            else:
                bg_str = f"{bg_val:.6f}"
                
            f.write(f"{i}\t{fname}\t{cbm_str}\t{vbm_str}\t{bg_str}\n")
    
    print(f"能带边结果已保存到: {out_path}")

def process_all_files(directory, use_filtered=False, bias_col='Bias calc (V)', fwd_col='LI Demod 1 X (A)', bwd_col='LI Demod 1 X [bwd] (A)', curr_col='Current (A)', gn_norm=False, lam_factor=5.0):
    import Line_Log
    return Line_Log.process_all_files(directory, use_filtered, bias_col, fwd_col, bwd_col, curr_col, gn_norm, lam_factor)

def plot_lix_with_offset_log(lix_fwd_lines, lix_bwd_lines, bias_values, file_names, save_directory, use_filtered=False, 
                            show_plot=True, save_txt=True, save_svg=True, gn_norm=False, lam_factor=5.0, **kwargs):
    """
    绘制带有偏移的LI Demod 1 X对数图，并标注导带底和价带顶
    """
    # 提取参数
    method = kwargs.get('method', 'Adaptive Thresholding')
    sigma_level = kwargs.get('sigma_level', 2.0)
    fit_window = kwargs.get('fit_window', 0.05)
    smooth_window = kwargs.get('smooth_window', 11)
    v_center_min = kwargs.get('v_center_min', -0.3)
    v_center_max = kwargs.get('v_center_max', 0.3)
    delta_E = kwargs.get('delta_E', 0.15)
    min_gap = kwargs.get('min_gap', 0.05)
    slope_threshold = kwargs.get('slope_threshold', 0.5)
    sigma_multiplier = kwargs.get('sigma_multiplier', 0.0)
    sigma_v_min = kwargs.get('sigma_v_min', -1.0)
    sigma_v_max = kwargs.get('sigma_v_max', 0.2)
    stack_offset_k = kwargs.get('stack_offset_k', 1.2)
    v_display_min = kwargs.get('v_display_min')
    v_display_max = kwargs.get('v_display_max')
    
    # 裁剪电压范围
    if v_display_min is not None and v_display_max is not None:
        mask = (bias_values >= v_display_min) & (bias_values <= v_display_max)
        if np.any(mask):
            bias_values = bias_values[mask]
            lix_fwd_lines = [line[mask] for line in lix_fwd_lines]
            lix_bwd_lines = [line[mask] for line in lix_bwd_lines]
            
    # 颜色列表
    colors = plt.cm.jet(np.linspace(0, 1, len(lix_fwd_lines)))
    
    # 创建保存目录
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # 处理数据
    if len(lix_fwd_lines) == 0 or len(lix_bwd_lines) == 0:
        print("错误: 没有数据用于计算偏移量")
        return None

    # 合并所有数据估计偏移量
    all_lix_data = np.concatenate(lix_fwd_lines + lix_bwd_lines)
    all_abs = np.abs(all_lix_data)

    finite_mask = np.isfinite(all_abs) & (all_abs > 0)
    if not np.any(finite_mask):
        print("错误: 没有有效的数据用于计算偏移量 (全部为0或非数)")
        return None

    log_all = np.log10(all_abs[finite_mask])
    std_dev = np.std(log_all)
    offset_value = std_dev * stack_offset_k # 使用传入的系数

    data_type_suffix = " (Filtered)" if use_filtered else ""
    title_suffix = f" (gN, λ factor={lam_factor})" if gn_norm else ""
    filename_suffix = "_filtered" if use_filtered else ""
    forward_filename = os.path.join(save_directory, f'lix_x_forward_with_offset_log_band_edges{filename_suffix}.svg')
    backward_filename = os.path.join(save_directory, f'lix_x_backward_with_offset_log_band_edges{filename_suffix}.svg')

    # ===================== Separate Figures =====================
    fig1 = plt.figure(figsize=(6, 10))
    ax_fwd = fig1.add_subplot(111)
    
    fig2 = plt.figure(figsize=(6, 10))
    ax_bwd = fig2.add_subplot(111)

    # --- Forward logic ---
    cbm_bias_list = []
    cbm_plot_x = []
    cbm_plot_y = []
    vbm_bias_list = []
    vbm_plot_x = []
    vbm_plot_y = []

    for i, (line_data, color) in enumerate(zip(lix_fwd_lines, colors)):
        # 使用垂直平移优化
        line_shifted, _ = apply_vertical_shift(bias_values, line_data, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
        log_line = np.log10(np.maximum(line_shifted, 1e-20))
        
        offset_line_data = log_line + i * offset_value
        ax_fwd.plot(bias_values, offset_line_data, color=color, alpha=0.8, linewidth=1)

        try:
            Ev, Ec, Eg, _ = detect_band_edges(
                bias_values, line_data, 
                method=method, mode="both", min_gap=min_gap,
                sigma_level=sigma_level, fit_window=fit_window,
                smooth_window=smooth_window, v_center_min=v_center_min,
                v_center_max=v_center_max, delta_E=delta_E,
                slope_threshold=slope_threshold,
                sigma_multiplier=sigma_multiplier,
                sigma_v_min=sigma_v_min,
                sigma_v_max=sigma_v_max,
                **kwargs
            )
            if Ec is not None and not np.isnan(Ec):
                idx_cbm = int(np.argmin(np.abs(bias_values - Ec)))
                cbm_bias = float(bias_values[idx_cbm])
                cbm_y = float(offset_line_data[idx_cbm])
                cbm_bias_list.append(cbm_bias)
                cbm_plot_x.append(cbm_bias)
                cbm_plot_y.append(cbm_y)
                ax_fwd.scatter(cbm_bias, cbm_y, marker='o', facecolors='none', edgecolors='blue', linewidths=1.2, zorder=5)
            else:
                cbm_bias_list.append(None)
            if Ev is not None and not np.isnan(Ev):
                idx_vbm = int(np.argmin(np.abs(bias_values - Ev)))
                vbm_bias = float(bias_values[idx_vbm])
                vbm_y = float(offset_line_data[idx_vbm])
                vbm_bias_list.append(vbm_bias)
                vbm_plot_x.append(vbm_bias)
                vbm_plot_y.append(vbm_y)
                ax_fwd.scatter(vbm_bias, vbm_y, marker='o', facecolors='none', edgecolors='red', linewidths=1.2, zorder=5)
            else:
                vbm_bias_list.append(None)
        except Exception:
            cbm_bias_list.append(None)
            vbm_bias_list.append(None)

    # 连接导带底和价带顶 (使用 spline 拟合)
    for plot_x, plot_y, color_str in [(cbm_plot_x, cbm_plot_y, 'blue'), (vbm_plot_x, vbm_plot_y, 'red')]:
        if len(plot_x) >= 2:
            # 排序以便 spline 拟合
            points = sorted(zip(plot_x, plot_y), key=lambda x: x[1])
            px, py = zip(*points)
            plot_spline(ax_fwd, px, py, linestyle='--', linewidth=2.0, color=color_str, dashes=[6, 4], zorder=10, alpha=0.8)
            ax_fwd.scatter(px, py, marker='o', s=50, facecolors='none', edgecolors=color_str, linewidths=2.0, zorder=11)

    ax_fwd.set_title(f'LI Demod 1 X Forward{data_type_suffix}{title_suffix}')
    ax_fwd.set_xlabel('Bias Voltage (V)')
    
    base_label = 'gN' if gn_norm else 'LI'
    ax_fwd.set_ylabel(f'log10({base_label} + offset) + stack_offset')
    ax_fwd.grid(True, alpha=0.3)

    # --- Backward logic ---
    for i, (line_data, color) in enumerate(zip(lix_bwd_lines, colors)):
        # 使用垂直平移优化
        line_shifted, _ = apply_vertical_shift(bias_values, line_data, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
        log_line = np.log10(np.maximum(line_shifted, 1e-20))
        
        offset_line_data = log_line + i * offset_value
        ax_bwd.plot(bias_values, offset_line_data, color=color, alpha=0.8, linewidth=1)
        
        try:
            Ev, Ec, Eg, _ = detect_band_edges(
                bias_values, line_data, 
                method=method, mode="both", min_gap=min_gap,
                sigma_level=sigma_level, fit_window=fit_window,
                smooth_window=smooth_window, v_center_min=v_center_min,
                v_center_max=v_center_max, delta_E=delta_E,
                slope_threshold=slope_threshold,
                sigma_multiplier=sigma_multiplier,
                sigma_v_min=sigma_v_min,
                sigma_v_max=sigma_v_max,
                **kwargs
            )
            if Ec is not None and not np.isnan(Ec):
                idx_cbm = int(np.argmin(np.abs(bias_values - Ec)))
                ax_bwd.scatter(bias_values[idx_cbm], offset_line_data[idx_cbm], marker='o', facecolors='none', edgecolors='blue', linewidths=1.2, zorder=5)
            if Ev is not None and not np.isnan(Ev):
                idx_vbm = int(np.argmin(np.abs(bias_values - Ev)))
                ax_bwd.scatter(bias_values[idx_vbm], offset_line_data[idx_vbm], marker='o', facecolors='none', edgecolors='red', linewidths=1.2, zorder=5)
        except Exception: pass

    ax_bwd.set_title(f'LI Demod 1 X Backward{data_type_suffix}{title_suffix}')
    ax_bwd.set_xlabel('Bias Voltage (V)')
    ax_bwd.set_ylabel(f'log10({base_label} + offset) + stack_offset')
    ax_bwd.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig2.tight_layout()
    
    if save_svg:
        fig1.savefig(forward_filename, dpi=300, bbox_inches='tight')
        fig2.savefig(backward_filename, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    if save_txt:
        save_band_edge_results(cbm_bias_list, vbm_bias_list, file_names, save_directory, use_filtered)

    return (fig1, fig2), cbm_bias_list, vbm_bias_list

def main():
    """主函数"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        tk_root = tk.Tk()
        tk_root.withdraw()
        
        directory = filedialog.askdirectory(title="请选择包含.dat文件的文件夹")
        tk_root.destroy()
        
        if not directory:
            print("未选择文件夹，程序退出。")
            return
    except ImportError:
        print("注意: 无法导入GUI库，将使用当前目录。")
        directory = '.'
    
    use_filtered = False  # 如需使用 [filt] 列可改为 True
    
    print(f"正在处理文件夹: {directory}")
    print("正在处理数据文件...")
    lix_x_fwd_lines, lix_x_bwd_lines, bias_values, file_names = process_all_files(directory, use_filtered)
    
    if len(lix_x_fwd_lines) == 0:
        print("错误: 没有找到数据文件或数据文件格式不正确。")
        return
    
    if bias_values is None:
        print("错误: 无法从数据文件中提取Bias值。")
        return
    
    print(f"处理完成:")
    print(f"  线数: {len(lix_x_fwd_lines)}")
    print(f"  每条线数据点数: {len(bias_values)}")
    
    print("正在生成 LI Demod 1 X 带偏移线图 (对数坐标) 并标记导带底和价带顶...")
    plot_lix_with_offset_log(lix_x_fwd_lines, lix_x_bwd_lines, bias_values,
                             file_names, directory, use_filtered,
                             v_center_min=-0.5, v_center_max=0.3, delta_E=0.15, min_gap=0.1)
    
    filename_suffix = "_filtered" if use_filtered else ""
    print("图像已保存到:")
    print(f"  - {os.path.join(directory, f'lix_x_forward_with_offset_log_band_edges{filename_suffix}.svg')}")
    print(f"  - {os.path.join(directory, f'lix_x_backward_with_offset_log_band_edges{filename_suffix}.svg')}")

if __name__ == "__main__":
    main()
