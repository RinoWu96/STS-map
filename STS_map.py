#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# 显式导入矢量图后端，防止 PyInstaller 打包时遗漏
try:
    import matplotlib.backends.backend_svg
    import matplotlib.backends.backend_pdf
except ImportError:
    pass

# PyInstaller 资源路径处理
def get_resource_path(relative_path):
    """获取程序资源的绝对路径，兼容 PyInstaller 打包后的路径"""
    if hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS) / relative_path
    return Path(os.path.abspath(".")) / relative_path

# ==== 导入所需的模块 ====
# 确保模块导入路径正确
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import Ec_log as ecmod
import newBG as newbg_mod
import Line_CBM_VBM as cbm_vbm_mod
import Line_Log as line_log_mod
from newBG import plot_spline
from band_edge_core import apply_vertical_shift

# ==== 全局参数变量 ====

# ==== 语言配置 ====
LANG_CONFIG = {
    'zh': {
        'title': "STS Line Scan 简化分析工具",
        'folder_label': "数据文件夹:",
        'browse_btn': "浏览",
        'tool_label': "分析类型:",
        'display_options': "显示选项:",
        'save_txt': "保存带隙结果",
        'param_btn': "参数设置",
        'run_btn': "Run 分析",
        'update_btn': "更新绘图",
        'table_title': "结果表格 (双击 Ec/Ev 编辑)",
        'id_col': "编号",
        'file_col': "文件名",
        'ec_col': "Ec (V)",
        'ev_col': "Ev (V)",
        'eg_col': "Eg (eV)",
        'log_label': "日志:",
        'placeholder': "结果图将在此处显示",
        'msg_hint': "提示",
        'msg_err': "错误",
        'select_folder': "请先选择数据文件夹。",
        'run_analysis_hint': "请先运行分析获取数据。",
        'manual_update_log': "正在根据手动修改的数据更新绘图...",
        'smooth_log': "已对手动修改的数据重新执行了高斯平滑。",
        'updated_txt_log': "已保存结果到 TXT: ",
        'updated_plot_log': "绘图已更新。",
        'processing_log': "正在处理数据文件...",
        'complete_log': "处理完成: 线数: {n_lines}, 每点数: {n_points}",
        'generating_plot_log': "正在生成图像...",
        'save_folder_log': "图像已保存到文件夹: ",
        'batch_start_log': "开始处理: ",
        'batch_complete_log': "完成: ",
        'param_window_title': "参数设置 - {tool}",
        'pattern': "文件模式 (pattern):",
        'method': "分析方法 (method):",
        'method_adaptive': "自适应阈值法 (Adaptive Thresholding)",
        'method_extrapolation': "线性外推法 (Linear Extrapolation)",
        'method_slope': "斜率判定法 (Slope Detection)",
        'sigma': "Sigma Level:",
        'slope_threshold': "斜率阈值 (slope_thresh):",
        'fit_window': "Fit Window (eV):",
        'smooth_window': "Smooth Window:",
        'v_center_min': "V_center_min:",
        'v_center_max': "V_center_max:",
        'delta_E': "ΔE (eV):",
        'eg_min': "最小带隙 (eg_min):",
        'eg_max': "最大带隙 (eg_max):",
        'min_gap': "最小带隙 (eV):",
        'save_params': "保存参数",
        'lang_label': "语言:",
        'ec_center_min': "V_center_min (Ec):",
        'ec_center_max': "V_center_max (Ec):",
        'invalid_tool_plot': "目前工具 '{tool}' 暂不支持手动更新后的重新绘图。",
        'no_params': "该工具暂无参数设置",
        'bias_col': "电压列 (Bias):",
        'curr_col': "电流列 (Current):",
        'lix_col': "LI Demod列:",
        'scan_err': "扫描文件列名失败: {err}",
        'save_fig1': "保存图1",
        'save_fig2': "保存图2",
        'save_results': "保存带隙结果",
        'save_line_plot_data': "保存Line绘图数据",
        'save_success': "图片已保存: {path}",
        'line_plot_data_saved': "已保存Line绘图数据到: ",
        'log_li_label': "LI Demod 对数显示",
        'gn_norm_label': "快速归一化 (gN)",
        'lam_factor_label': "gN 软正则因子 (λ factor):",
        'fit_order_label': "Map 平滑量级:",
        'run_len': "VB 连续点数 (run_len):",
        'run_len_cb': "CB 连续点数 (run_len_cb):",
        'baseline_k': "基准系数 (baseline_k):",
        'amp_k': "幅度系数 (amp_k):",
        'r2_min': "最小 R2 (r2_min):",
        'curv_k': "曲率系数 (curv_k):",
        'curv_percentile': "曲率分位数:",
        'run_len_curv': "曲率连续点数:",
        'vb_q': "VB 分位数 (vb_q):",
        'sigma_multiplier': "对数平移倍数 (sigma):",
        'gn_norm': "应用 gN 归一化:",
        'lam_factor': "gN λ 因子:",
        'map_smooth_sigma': "Map 平滑量级 (sigma):",
        'v_display_min': "电压显示下限 (V):",
        'v_display_max': "电压显示上限 (V):",
        'sigma_v_min': "噪声采样下限 (V):",
        'sigma_v_max': "噪声采样上限 (V):",
        'stack_offset_k': "堆叠偏移系数 (k):",
    },
    'en': {
        'title': "STS Line Scan Analysis Tool",
        'folder_label': "Data Folder:",
        'browse_btn': "Browse",
        'tool_label': "Analysis Type:",
        'display_options': "Display Options:",
        'save_txt': "Save Gap Results",
        'param_btn': "Parameters",
        'run_btn': "Run Analysis",
        'update_btn': "Update Plot",
        'table_title': "Results Table (Double-click Ec/Ev to edit)",
        'id_col': "ID",
        'file_col': "Filename",
        'ec_col': "Ec (V)",
        'ev_col': "Ev (V)",
        'eg_col': "Eg (eV)",
        'log_label': "Log:",
        'placeholder': "Result plot will be displayed here",
        'msg_hint': "Hint",
        'msg_err': "Error",
        'select_folder': "Please select data folder first.",
        'run_analysis_hint': "Please run analysis to get data first.",
        'manual_update_log': "Updating plot based on manual modifications...",
        'smooth_log': "Gaussian smoothing re-applied to manual data.",
        'updated_txt_log': "Saved results to TXT: ",
        'updated_plot_log': "Plot updated.",
        'processing_log': "Processing data files...",
        'complete_log': "Completed: Lines: {n_lines}, Points: {n_points}",
        'generating_plot_log': "Generating plot...",
        'save_folder_log': "Images saved to: ",
        'batch_start_log': "Processing: ",
        'batch_complete_log': "Done: ",
        'param_window_title': "Params - {tool}",
        'pattern': "File Pattern:",
        'method': "Method:",
        'method_adaptive': "Adaptive Thresholding",
        'method_extrapolation': "Linear Extrapolation",
        'method_slope': "Slope Detection",
        'sigma': "Sigma Level:",
        'slope_threshold': "Slope Threshold:",
        'fit_window': "Fit Window (eV):",
        'smooth_window': "Smooth Window:",
        'v_center_min': "V_center_min:",
        'v_center_max': "V_center_max:",
        'delta_E': "ΔE (eV):",
        'eg_min': "Min Gap (eg_min):",
        'eg_max': "Max Gap (eg_max):",
        'min_gap': "Min Gap (eV):",
        'save_params': "Save Parameters",
        'lang_label': "Language:",
        'ec_center_min': "V_center_min (Ec):",
        'ec_center_max': "V_center_max (Ec):",
        'invalid_tool_plot': "Tool '{tool}' currently doesn't support re-plotting after manual update.",
        'no_params': "This tool has no parameters.",
        'bias_col': "Bias Column:",
        'curr_col': "Current Column:",
        'lix_col': "LI Demod Column:",
        'scan_err': "Failed to scan file columns: {err}",
        'save_fig1': "Save Fig1",
        'save_fig2': "Save Fig2",
        'save_results': "Save Gap Results",
        'save_line_plot_data': "Save Line Plot Data",
        'save_success': "Figure saved: {path}",
        'line_plot_data_saved': "Line plot data saved to: ",
        'log_li_label': "Log LI Demod Map",
        'gn_norm_label': "Quick Normalization (gN)",
        'lam_factor_label': "gN λ factor:",
        'fit_order_label': "Map Smooth Sigma:",
        'run_len': "VB Run Length:",
        'run_len_cb': "CB Run Length:",
        'baseline_k': "Baseline K:",
        'amp_k': "Amplitude K:",
        'r2_min': "Min R2:",
        'curv_k': "Curvature K:",
        'curv_percentile': "Curvature %:",
        'run_len_curv': "Curvature Run Len:",
        'vb_q': "VB Quantile (vb_q):",
        'sigma_multiplier': "Log Shift Multiplier (sigma):",
        'gn_norm': "Apply gN Norm:",
        'lam_factor': "gN λ Factor:",
        'map_smooth_sigma': "Map Smooth Sigma:",
        'v_display_min': "Display Min (V):",
        'v_display_max': "Display Max (V):",
        'sigma_v_min': "Noise Sampling Min (V):",
        'sigma_v_max': "Noise Sampling Max (V):",
        'stack_offset_k': "Stack Offset Factor (k):",
    }
}

# 当前语言
CURRENT_LANG = 'zh'

def get_text(key, **kwargs):
    """获取当前语言的文本"""
    text = LANG_CONFIG[CURRENT_LANG].get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text

# Ec_log 参数 (0-1V)
EC_PARAMS = {
    'pattern': '*.dat',
    'method': 'Adaptive Thresholding',
    'sigma_level': 2.0,
    'fit_window': 0.05,
    'smooth_window': 5,
    'v_center_min': 0.0,
    'v_center_max': 0.2,
    'delta_E': 0.15,
    'slope_threshold': 0.5,
    'sigma_map_bias': 1.0,
    'sigma_map_pos': 1.0,
    'run_len': 8,
    'run_len_cb': 5,
    'baseline_k': 0.0,
    'amp_k': 2.0,
    'r2_min': 0.90,
    'curv_k': 2.0,
    'curv_percentile': 95,
    'run_len_curv': 3,
    'vb_q': 0.80,
    'sigma_multiplier': 0.0,
    'stack_offset_k': 1.2,
    'gn_norm': False,
    'lam_factor': 5.0,
    'map_smooth_sigma': 1.0,
    'v_display_min': None,
    'v_display_max': None,
    'sigma_v_min': -1.0,
    'sigma_v_max': 0.2,
}

# Line CBM VBM 参数
CBM_VBM_PARAMS = {
    'pattern': '*.dat',
    'method': 'Adaptive Thresholding',
    'sigma_level': 2.0,
    'fit_window': 0.05,
    'smooth_window': 5,
    'v_center_min': -0.3,
    'v_center_max': 0.3,
    'delta_E': 0.15,
    'min_gap': 0.05,
    'slope_threshold': 0.5,
    'run_len': 8,
    'run_len_cb': 5,
    'baseline_k': 0.0,
    'amp_k': 2.0,
    'r2_min': 0.90,
    'curv_k': 2.0,
    'curv_percentile': 95,
    'run_len_curv': 3,
    'vb_q': 0.80,
    'sigma_multiplier': 0.0,
    'stack_offset_k': 1.2,
    'gn_norm': False,
    'lam_factor': 5.0,
    'map_smooth_sigma': 1.0,
    'v_display_min': None,
    'v_display_max': None,
    'sigma_v_min': -1.0,
    'sigma_v_max': 0.2,
}

# newBG 参数
NEWBG_PARAMS = {
    'pattern': '*.dat',
    'method': 'Adaptive Thresholding',
    'eg_min': 0.3,
    'eg_max': 3.5,
    'sigma_level': 2.0,
    'fit_window': 0.05,
    'smooth_window': 5,
    'v_center_min': -0.3,
    'v_center_max': 0.3,
    'delta_E': 0.15,
    'min_gap': 0.05,
    'slope_threshold': 0.5,
    'run_len': 8,
    'run_len_cb': 5,
    'baseline_k': 0.0,
    'amp_k': 2.0,
    'r2_min': 0.90,
    'curv_k': 2.0,
    'curv_percentile': 95,
    'run_len_curv': 3,
    'vb_q': 0.80,
    'sigma_multiplier': 0.0,
    'stack_offset_k': 1.2,
    'gn_norm': False,
    'lam_factor': 5.0,
    'map_smooth_sigma': 1.0,
    'v_display_min': None,
    'v_display_max': None,
    'sigma_v_min': -1.0,
    'sigma_v_max': 0.2,
}
# Line CBM VBM 无需额外参数


# ==== 全局数据变量 ====
CURRENT_DATA = {
    'df_res': None,
    'bias': None,
    'pos_nm': None,
    'M': None,
    'tool': None,
    'folder': None,
    'filename_prefix': None,
    'line_plot_df': None,
    'line_plot_meta': None,
    'figs': [None, None] # 存储当前生成的两个 Figure
}

# 数据列选择变量
bias_col_var = None
curr_col_var = None
lix_col_var = None
log_li_var = None
gn_norm_var = None
folder_var = None
tool_var = None
stats_var = None
log_text = None
root = None
canvas_frame = None
canvas_frames = None
result_tree = None

# ========== GUI回调函数 ==========

def display_figure(fig):
    """在右侧区域显示 Matplotlib 图像 (支持单个图或图列表，并强制 4:3 比例缩放)"""
    global canvas_frames
    
    # 清空两个容器
    for frame in canvas_frames:
        for widget in frame.winfo_children():
            widget.destroy()
    
    # 将 fig 转为列表统一处理
    figs = fig if isinstance(fig, (list, tuple)) else [fig]
    
    for i, f in enumerate(figs):
        if f is None or i >= len(canvas_frames): continue
        
        target_frame = canvas_frames[i]
        
        # 将生成的 Figure 存入全局，供手动保存使用
        if 'figs' not in CURRENT_DATA: CURRENT_DATA['figs'] = [None, None]
        if i < len(CURRENT_DATA['figs']):
            CURRENT_DATA['figs'][i] = f
            
        # 创建一个内部容器用于控制比例
        inner_container = tk.Frame(target_frame, bg="white")
        inner_container.pack(expand=True, fill="both")
        
        # 创建画布
        canvas = FigureCanvasTkAgg(f, master=inner_container)
        canvas_widget = canvas.get_tk_widget()
        
        # 核心：监听容器尺寸变化，动态调整画布以保持原始比例
        def make_resize_handler(cv_widget, figure):
            # 获取原始 Figure 的比例
            fw, fh = figure.get_size_inches()
            fig_ratio = fw / fh
            
            def on_resize(event):
                w, h = event.width, event.height
                if w <= 0 or h <= 0: return
                
                # 目标比例使用 Figure 自身的比例
                target_ratio = fig_ratio
                current_ratio = w / h
                
                if current_ratio > target_ratio:
                    # 太宽了，以高度为基准
                    new_h = h
                    new_w = h * target_ratio
                else:
                    # 太高了，以宽度为基准
                    new_w = w
                    new_h = w / target_ratio
                
                # 居中放置
                cv_widget.place(relx=0.5, rely=0.5, anchor="center", width=new_w, height=new_h)
            return on_resize

        inner_container.bind("<Configure>", make_resize_handler(canvas_widget, f))
        
        # 添加工具栏 (放在容器顶部)
        toolbar_frame = tk.Frame(target_frame)
        toolbar_frame.pack(side="top", fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()


def update_table_view(df_res):
    """更新表格显示内容 (显示平滑前的 raw 数据)"""
    # 清空表格
    for item in result_tree.get_children():
        result_tree.delete(item)
    
    if df_res is None:
        return
        
    for i, row in df_res.iterrows():
        # 优先显示 raw 原始数据
        ec = row.get("Ec_raw", row.get("Ec", np.nan))
        ev = row.get("Ev_raw", row.get("Ev", np.nan))
        eg = row.get("Eg_raw", row.get("Eg", np.nan))
        filename = row.get("filename", f"Point {i+1}")
        
        result_tree.insert("", "end", values=(
            i + 1,
            filename,
            f"{ec:.4f}" if not np.isnan(ec) else "NaN",
            f"{ev:.4f}" if not np.isnan(ev) else "NaN",
            f"{eg:.4f}" if not np.isnan(eg) else "NaN"
        ))


def on_tree_double_click(event):
    """双击表格单元格进行编辑"""
    region = result_tree.identify_region(event.x, event.y)
    if region != "cell":
        return
        
    column = result_tree.identify_column(event.x) # "#1", "#2", ...
    item_id = result_tree.identify_row(event.y)
    
    # 只允许编辑 Ec (#3) 和 Ev (#4)
    if column not in ("#3", "#4"):
        return
        
    # 获取单元格位置和大小
    x, y, width, height = result_tree.bbox(item_id, column)
    
    # 创建临时 Entry
    entry = tk.Entry(result_tree)
    entry.insert(0, result_tree.item(item_id, "values")[int(column[1:])-1])
    entry.select_range(0, tk.END)
    entry.focus_set()
    
    def save_edit(event=None):
        new_val = entry.get()
        old_values = list(result_tree.item(item_id, "values"))
        col_idx = int(column[1:]) - 1
        
        try:
            # 验证输入是否为数字
            if new_val.lower() != "nan":
                float(new_val)
            
            old_values[col_idx] = new_val
            
            # 自动计算 Eg (col_idx 2 is Ec, 3 is Ev, 4 is Eg)
            try:
                ec_val = float(old_values[2])
                ev_val = float(old_values[3])
                eg_val = ec_val - ev_val
                old_values[4] = f"{eg_val:.4f}"
            except ValueError:
                old_values[4] = "NaN"
                
            result_tree.item(item_id, values=old_values)
        except ValueError:
            pass # 忽略无效输入
            
        entry.destroy()
        
    entry.bind("<Return>", save_edit)
    entry.bind("<FocusOut>", save_edit)
    entry.place(x=x, y=y, width=width, height=height)


def update_from_table():
    """从表格读取修改后的数据并重新绘图"""
    global log_li_var
    if CURRENT_DATA['df_res'] is None:
        messagebox.showwarning(get_text('msg_hint'), get_text('run_analysis_hint'))
        return
        
    log(get_text('manual_update_log'))
    
    df = CURRENT_DATA['df_res'].copy()
    
    # 从表格读取新值
    for item in result_tree.get_children():
        values = result_tree.item(item, "values")
        idx = int(values[0]) - 1 # 编号从 1 开始
        
        try:
            ec = float(values[2]) if values[2].lower() != "nan" else np.nan
            ev = float(values[3]) if values[3].lower() != "nan" else np.nan
            eg = float(values[4]) if values[4].lower() != "nan" else np.nan
            
            df.loc[idx, "Ec"] = ec
            df.loc[idx, "Ev"] = ev
            df.loc[idx, "Eg"] = eg
            
            # 同时更新 raw 字段，以便绘图函数同步显示
            if "Ec_raw" in df.columns: df.loc[idx, "Ec_raw"] = ec
            if "Ev_raw" in df.columns: df.loc[idx, "Ev_raw"] = ev
            if "Eg_raw" in df.columns: df.loc[idx, "Eg_raw"] = eg
        except ValueError:
            continue
            
    # 更新全局数据
    CURRENT_DATA['df_res'] = df
    
    # 如果用户希望在手动修改后重新执行一次平滑 (refitting)
    # 可以通过 scipy 的 gaussian_filter1d 模拟 newBG 的逻辑
    try:
        from scipy.ndimage import gaussian_filter1d
        valid = ~np.isnan(df["Ev"])
        if valid.sum() > 3:
            df.loc[valid, "Ev"] = gaussian_filter1d(df.loc[valid, "Ev"], sigma=1.0, mode="nearest")
            df.loc[valid, "Ec"] = gaussian_filter1d(df.loc[valid, "Ec"], sigma=1.0, mode="nearest")
            df["Eg"] = df["Ec"] - df["Ev"]
            log(get_text('smooth_log'))
    except Exception:
        pass
    
    # 更新表格显示 (反映平滑后的值)
    update_table_view(df)
    
    # 重新计算统计量并更新标签
    stats = compute_eg_stats(df)
    update_stats_label(stats, CURRENT_DATA['folder'].name)
    
    # 重新绘图
    tool = CURRENT_DATA['tool']
    bias = CURRENT_DATA['bias']
    pos_nm = CURRENT_DATA['pos_nm']
    M = CURRENT_DATA['M']
    log_li = log_li_var.get()
    gn_norm = gn_norm_var.get()
    
    # 获取当前工具的参数字典
    params = {
        "Ec_log": EC_PARAMS,
        "newBG": NEWBG_PARAMS,
        "Line CBM VBM": CBM_VBM_PARAMS,
        "line": EC_PARAMS
    }.get(tool, EC_PARAMS)

    lam_factor = params.get('lam_factor', 5.0)
    m_sigma = params.get('map_smooth_sigma', 1.0)
    s_mult = params.get('sigma_multiplier', 0.0)
    v_min = params.get('v_display_min')
    v_max = params.get('v_display_max')
    s_v_min = params.get('sigma_v_min', -1.0)
    s_v_max = params.get('sigma_v_max', 0.2)

    # --- 新增：从当前图中提取用户手动调整后的量程 ---
    vmin, vmax = None, None
    if 'figs' in CURRENT_DATA and CURRENT_DATA['figs'][0] is not None:
        try:
            # 查找旧图中的 imshow 对象并获取其当前的 clim
            for ax in CURRENT_DATA['figs'][0].get_axes():
                for im in ax.get_images():
                    vmin, vmax = im.get_clim()
                    break
                if vmin is not None: break
        except Exception:
            pass

    # 如果 M_raw 存在且平滑量级或归一化状态发生了变化，重新应用
    if 'M_raw' in CURRENT_DATA and CURRENT_DATA['M_raw'] is not None:
        from scipy.ndimage import gaussian_filter
        
        # 重新应用归一化 (如果选中)
        M_to_smooth = CURRENT_DATA['M_raw'].copy()
        if gn_norm and 'I_raw' in CURRENT_DATA and CURRENT_DATA['I_raw'] is not None:
            V = CURRENT_DATA['bias']
            if len(V.shape) == 1:
                V = V[:, np.newaxis] # 广播到 (Nbias, Npos)
            I = CURRENT_DATA['I_raw']
            
            from band_edge_core import apply_gn_normalization
            M_to_smooth = apply_gn_normalization(V, I, M_to_smooth, lam_factor=lam_factor)
            log(f"gN normalization (Soft Regularization) applied (λ factor={lam_factor}).")

        # 重新平滑
        CURRENT_DATA['M'] = gaussian_filter(M_to_smooth, sigma=(m_sigma, m_sigma))
        CURRENT_DATA['m_sigma'] = m_sigma
        log(f"Map processing updated (sigma={m_sigma}, gn={gn_norm}).")

    fig = None
    if tool == "newBG":
        fig = newbg_mod.plot_final(
            bias, pos_nm, CURRENT_DATA['M'], df,
            title_prefix=f"STS line (Updated)",
            show_gap_curve=True,
            log_li=log_li,
            gn_norm=gn_norm,
            lam_factor=lam_factor,
            sigma_multiplier=s_mult,
            sigma_v_min=s_v_min, sigma_v_max=s_v_max,
            v_display_min=v_min, v_display_max=v_max,
            vmin=vmin, vmax=vmax
        )
    elif tool == "Ec_log":
        # 手动更新 Ec_log 绘图，保持两个独立 Figure 分左右显示
        # --- 图1: dI/dV Map ---
        fig1 = plt.figure(figsize=(6, 4.5))
        ax1 = fig1.add_subplot(111)
        extent = [pos_nm.min(), pos_nm.max(), bias.min(), bias.max()]
        
        M_plot = CURRENT_DATA['M']
        cb_label = "gN (arb. u.)" if gn_norm else "LI Demod 1 X (A)"
        if log_li:
            # 使用垂直平移优化
            M_shifted, _ = apply_vertical_shift(bias, M_plot, sigma_multiplier=s_mult, sigma_v_min=s_v_min, sigma_v_max=s_v_max)
            M_plot = np.log10(np.maximum(M_shifted, 1e-20))
            cb_label = f"log10({cb_label} + offset)"
            
        im = ax1.imshow(M_plot, extent=extent, origin="lower", aspect="auto", cmap="jet", vmin=vmin, vmax=vmax)
        title_suffix = f" (gN, λ factor={lam_factor})" if gn_norm else ""
        ax1.set_title(f"Line STS Lock-in X Map{title_suffix}")
        ax1.set_xlabel("Position (nm)")
        ax1.set_ylabel("Bias (V)")
        
        # 自动适配 Y 轴两程
        if v_min is not None and v_max is not None:
            ax1.set_ylim(v_min, v_max)
            
        plt.colorbar(im, ax=ax1, label=cb_label)
        fig1.tight_layout()
        
        # --- 图2: Ec vs position ---
        fig2 = plt.figure(figsize=(6, 4.5))
        ax2 = fig2.add_subplot(111)
        ax2.plot(pos_nm, df["Ec_raw"], "o", alpha=0.3, label="Ec (raw)", markersize=3)
        plot_spline(ax2, pos_nm, df["Ec"], "-", alpha=0.8, label="Ec (spline)", linewidth=1.5)
        ax2.set_title("Conduction band onset Ec (Manual Updated)")
        ax2.set_xlabel("Position (nm)")
        ax2.set_ylabel("Ec (V)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        fig2.tight_layout()
        
        fig = [fig1, fig2]
    elif tool == "Line CBM VBM":
        # ... (rest of logic)
        # 重新生成 Line CBM VBM 的图
        # 需要获取原始数据
        # 这里我们可以直接模拟 plot_lix_with_offset_log 的绘图逻辑，但使用更新后的 Ec/Ev
        
        # 为了保持一致性，我们在这里简单实现一个更新后的可视化
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        
        # 假设 M 是堆叠的 forward 数据
        # 我们需要从 CURRENT_DATA 中获取更多信息，或者简化显示
        # 实际上 Line CBM VBM 的图很复杂，包含 offset。
        # 最好的办法是修改 Line_CBM_VBM.py 增加一个根据给定 Ec/Ev 绘图的函数
        
        log(get_text('invalid_tool_plot', tool=tool))
        return
    else:
        log(get_text('invalid_tool_plot', tool=tool))
        return

    if fig:
        display_figure(fig)
            
    log(get_text('updated_plot_log'))


def choose_folder():
    folder = filedialog.askdirectory(title=get_text('folder_label'))
    if folder:
        folder_var.set(folder)
        # 重置电压显示范围为自动模式 (None)
        for p_dict in [EC_PARAMS, NEWBG_PARAMS, CBM_VBM_PARAMS]:
            p_dict['v_display_min'] = None
            p_dict['v_display_max'] = None
        log(f"Selected folder: {folder}. Voltage range reset to auto.")
        scan_sample_columns(folder)


def scan_sample_columns(folder_path):
    """扫描文件夹中的第一个 .dat 文件以获取列名列表"""
    global bias_col_var, curr_col_var, lix_col_var
    try:
        folder = Path(folder_path)
        dat_files = list(folder.glob("*.dat"))
        if not dat_files:
            return
            
        # 使用 newBG 模块中的读取函数
        sample_file = dat_files[0]
        # 尝试使用不同编码
        try:
            df, meta = newbg_mod.read_sts_dat(sample_file)
        except Exception:
            # 如果 utf-8 失败，尝试 latin-1 (Omicron 某些旧版常用)
            with open(sample_file, "r", encoding="latin-1", errors="ignore") as f:
                lines = f.read().splitlines()
                # 简单的解析逻辑，找包含 [DATA] 的行
                data_start = -1
                for i, line in enumerate(lines):
                    if "[DATA]" in line:
                        data_start = i + 1
                        break
                if data_start != -1:
                    # [DATA] 后的第一行即为表头
                    data_lines = lines[data_start:]
                    header_line = ""
                    header_idx = 0
                    for i, line in enumerate(data_lines):
                        if line.strip():
                            header_line = line
                            header_idx = i
                            break
                    if header_line:
                        import io
                        # 优先尝试 Tab 分隔
                        try:
                            df = pd.read_csv(io.StringIO("\n".join(data_lines[header_idx:])), sep="\t")
                            if len(df.columns) <= 1:
                                df = pd.read_csv(io.StringIO("\n".join(data_lines[header_idx:])), sep=None, engine='python')
                        except Exception:
                            df = pd.read_csv(io.StringIO("\n".join(data_lines[header_idx:])), sep=None, engine='python')
                    else:
                        raise ValueError("No header line found after [DATA]")
                else:
                    raise ValueError("No [DATA] section found")
                    
        cols = list(df.columns)
        
        # 更新下拉列表
        for combo in [UI_WIDGETS['bias_combo'], UI_WIDGETS['curr_combo'], UI_WIDGETS['lix_combo']]:
            combo['values'] = cols
            
        # 尝试自动匹配
        def auto_set(var, patterns):
            for c in cols:
                if any(p.lower() in c.lower() for p in patterns):
                    var.set(c)
                    break
        
        auto_set(bias_col_var, ["bias", "volt"])
        auto_set(curr_col_var, ["current", "curr"])
        auto_set(lix_col_var, ["LI Demod 1 X", "LI Demod", "Lock-in"])
        
    except Exception as e:
        log(get_text('scan_err', err=str(e)))


def log(msg: str):
    """简单日志输出到 Text 框"""
    log_text.configure(state="normal")
    log_text.insert("end", msg + "\n")
    log_text.see("end")
    log_text.configure(state="disabled")


def compute_eg_stats(df_res):
    """计算 Eg 的统计量（mean, std, min, max, N_valid）"""
    if df_res is None or "Eg" not in df_res.columns:
        return None

    eg = df_res["Eg"].dropna()
    eg = eg[eg > 1e-6]  # 过滤掉 ≈0 的值
    if len(eg) == 0:
        return None

    return {
        "mean": float(eg.mean()),
        "std": float(eg.std()),
        "min": float(eg.min()),
        "max": float(eg.max()),
        "N": int(len(eg)),
    }


def safe_stack_offset_k(stack_offset_k, default=1.2):
    try:
        stack_offset_k = float(stack_offset_k)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(stack_offset_k) or stack_offset_k <= 0:
        return default
    return stack_offset_k


def build_line_plot_dataframe(lix_fwd_lines, lix_bwd_lines, bias_values, log_li=False,
                              sigma_multiplier=0.0, v_display_min=None, v_display_max=None,
                              sigma_v_min=-1.0, sigma_v_max=0.2, stack_offset_k=1.2,
                              gn_norm=False):
    """把 line 模式中真正绘到图上的曲线整理成一列 x、多列 y 的表格。"""
    if bias_values is None or len(lix_fwd_lines) == 0:
        return None, None
    stack_offset_k = safe_stack_offset_k(stack_offset_k)

    bias_values = np.asarray(bias_values, dtype=float)
    fwd_lines = [np.asarray(line, dtype=float) for line in lix_fwd_lines]
    bwd_lines = [np.asarray(line, dtype=float) for line in lix_bwd_lines]

    if v_display_min is not None and v_display_max is not None:
        mask = (bias_values >= v_display_min) & (bias_values <= v_display_max)
        if np.any(mask):
            bias_values = bias_values[mask]
            fwd_lines = [line[mask] for line in fwd_lines]
            bwd_lines = [line[mask] for line in bwd_lines]

    all_lines = fwd_lines + bwd_lines
    if not all_lines:
        return None, None

    all_lix_data = np.concatenate(all_lines)
    all_abs = np.abs(all_lix_data)
    finite_mask = np.isfinite(all_abs) & (all_abs > 0)

    if log_li:
        log_all = np.log10(all_abs[finite_mask]) if np.any(finite_mask) else np.array([0.0])
        std_dev = float(np.std(log_all))
        if not np.isfinite(std_dev) or std_dev <= 0:
            std_dev = 1.0
        offset_value = float(std_dev * stack_offset_k)
        y_mode = "log10_shifted_plus_stack_offset"
    else:
        max_val = np.nanmax(all_abs) if np.any(np.isfinite(all_abs)) else 1e-9
        offset_value = float(max_val * 0.8 * stack_offset_k)
        y_mode = "linear_plus_stack_offset"

    data = {"Bias_V": bias_values}

    def add_plot_columns(prefix, lines):
        for i, line_data in enumerate(lines, start=1):
            if log_li:
                line_shifted, _ = apply_vertical_shift(
                    bias_values, line_data,
                    sigma_multiplier=sigma_multiplier,
                    sigma_v_min=sigma_v_min,
                    sigma_v_max=sigma_v_max
                )
                plot_data = np.log10(np.maximum(line_shifted, 1e-20))
            else:
                plot_data = line_data

            data[f"{prefix}_Line_{i:03d}_Y"] = plot_data + (i - 1) * offset_value

    add_plot_columns("Fwd", fwd_lines)
    add_plot_columns("Bwd", bwd_lines)

    meta = {
        "offset_value": offset_value,
        "spacing_multiplier": float(stack_offset_k),
        "y_mode": y_mode,
        "n_x": int(len(bias_values)),
        "n_y": int(len(data) - 1),
    }
    return pd.DataFrame(data), meta


def update_stats_label(stats, folder_name):
    """在 GUI 上显示 Eg 统计结果"""
    if stats is None:
        if CURRENT_LANG == 'zh':
            stats_var.set(f"[{folder_name}] 无有效 Eg 数据（全为 0 或 NaN）")
        else:
            stats_var.set(f"[{folder_name}] No valid Eg data (all 0 or NaN)")
    else:
        stats_var.set(
            f"[{folder_name}] Eg (eV): "
            f"mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"min={stats['min']:.3f}, max={stats['max']:.3f}, N={stats['N']}"
        )


def run_line_analysis(folder: Path, tool_name: str, save_png: bool, is_batch: bool, bias_col="Bias calc (V)", lix_col="LI Demod 1 X (A)", log_li=False, gn_norm=False, lam_factor=5.0, sigma_multiplier=0.0, v_display_min=None, v_display_max=None, sigma_v_min=-1.0, sigma_v_max=0.2, stack_offset_k=1.2, filename_prefix=None):
    """运行通用的 Line 分析"""
    try:
        log(get_text('processing_log'))
        # 尝试猜测后向列名
        bwd_col = lix_col.replace("Forward", "Backward").replace(" (A)", " [bwd] (A)") if "Forward" in lix_col else "LI Demod 1 X [bwd] (A)"
        
        lix_x_fwd_lines, lix_x_bwd_lines, bias_values, valid_files, metadata = line_log_mod.process_all_files(
            str(folder), False, bias_col=bias_col, fwd_col=lix_col, bwd_col=bwd_col, curr_col=curr_col_var.get(), gn_norm=gn_norm, lam_factor=lam_factor
        )
        
        if not lix_x_fwd_lines:
            log("Warning: No valid data files found.")
            return None
            
        if bias_values is None:
            log("Error: Could not extract bias values.")
            return None

        # 自动适配显示范围 (如果尚未设定)
        # 注意：这里我们使用传入的 v_display_min/max。
        # 如果调用者传的是 None，我们会计算并更新 it
        if v_display_min is None: v_display_min = float(np.min(bias_values))
        if v_display_max is None: v_display_max = float(np.max(bias_values))

        log(get_text('complete_log', n_lines=len(lix_x_fwd_lines), n_points=len(bias_values)))
        log(get_text('generating_plot_log'))
        
        # 使用 Line_Log.py 中的绘图函数，传递 log_li 参数和 metadata
        figs = line_log_mod.plot_lix_with_offset_log(
            lix_x_fwd_lines, lix_x_bwd_lines, bias_values, str(folder), 
            False, show_plot=False, log_li=log_li, gn_norm=gn_norm, lam_factor=lam_factor, sigma_multiplier=sigma_multiplier, 
            v_display_min=v_display_min, v_display_max=v_display_max, 
            sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max, stack_offset_k=stack_offset_k, metadata=metadata
        )

        line_plot_df, line_plot_meta = build_line_plot_dataframe(
            lix_x_fwd_lines, lix_x_bwd_lines, bias_values,
            log_li=log_li,
            sigma_multiplier=sigma_multiplier,
            v_display_min=v_display_min,
            v_display_max=v_display_max,
            sigma_v_min=sigma_v_min,
            sigma_v_max=sigma_v_max,
            stack_offset_k=stack_offset_k,
            gn_norm=gn_norm
        )
        CURRENT_DATA.update({
            'df_res': None,
            'bias': bias_values,
            'pos_nm': np.arange(len(valid_files)),
            'M': None,
            'M_raw': None,
            'I_raw': None,
            'tool': tool_name,
            'folder': folder,
            'filename_prefix': filename_prefix or folder.name,
            'line_plot_df': line_plot_df,
            'line_plot_meta': line_plot_meta,
            'log_li': log_li,
            'gn_norm': gn_norm
        })
        if line_plot_meta is not None:
            log(f"Line plot data prepared for Origin: x={line_plot_meta['n_x']}, y columns={line_plot_meta['n_y']}.")
        
        if save_png:
            log(get_text('save_folder_log') + str(folder))
        
        return figs
            
    except Exception as e:
        import traceback
        log(f"Error running {tool_name}: {str(e)}")
        log(traceback.format_exc())
        return None

def run_single_folder(folder: Path, tool: str, save_png: bool, is_batch: bool, filename_prefix: str = None):
    """对单个文件夹执行分析"""
    global bias_col_var, curr_col_var, lix_col_var, log_li_var, gn_norm_var
    log(get_text('batch_start_log') + str(folder))
    out_dir = folder
    filename_prefix = filename_prefix or folder.name

    fig = None
    
    # 获取当前工具的参数字典
    params = {
        "Ec_log": EC_PARAMS,
        "newBG": NEWBG_PARAMS,
        "Line CBM VBM": CBM_VBM_PARAMS,
        "line": EC_PARAMS
    }.get(tool, EC_PARAMS)

    # 获取用户选择的列名和显示选项
    b_col = bias_col_var.get()
    c_col = curr_col_var.get()
    l_col = lix_col_var.get()
    log_li = log_li_var.get()
    gn_norm = gn_norm_var.get()
    
    lam_factor = params.get('lam_factor', 5.0)
    m_sigma = params.get('map_smooth_sigma', 1.0)
    s_mult = params.get('sigma_multiplier', 0.0)
    v_min = params.get('v_display_min')
    v_max = params.get('v_display_max')
    s_v_min = params.get('sigma_v_min', -1.0)
    s_v_max = params.get('sigma_v_max', 0.2)
    st_k = safe_stack_offset_k(params.get('stack_offset_k', 1.2))
    CURRENT_DATA.update({'line_plot_df': None, 'line_plot_meta': None})

    if tool == "Ec_log":
        valid_keys_ec = [
            'pattern', 'method', 'sigma_level', 'fit_window', 'smooth_window', 
            'v_center_min', 'v_center_max', 'delta_E', 'sigma_map_bias', 'sigma_map_pos', 
            'smooth_half_window', 'use_auto_params', 'slope_threshold',
            'run_len', 'run_len_cb', 'baseline_k', 'amp_k', 'r2_min', 
            'curv_k', 'curv_percentile', 'run_len_curv', 'vb_q', 'sigma_multiplier',
            'v_display_min', 'v_display_max', 'sigma_v_min', 'sigma_v_max', 'stack_offset_k'
        ]
        filtered_ec_params = {k: v for k, v in params.items() if k in valid_keys_ec}
        # 传入自定义参数
        filtered_ec_params.update({'bias_col': b_col, 'curr_col': c_col, 'lix_col': l_col, 'log_li': log_li, 'gn_norm': gn_norm, 'lam_factor': lam_factor,
                                   'sigma_map_bias': m_sigma, 'sigma_map_pos': m_sigma, 'sigma_v_min': s_v_min, 'sigma_v_max': s_v_max})
        result = ecmod.process_line_folder_0to1V(folder, **filtered_ec_params)
        figs = result["figs"] # 现在返回的是 figs 列表
        
        # 自动适配显示范围 (如果尚未设定)
        bias_vals = result["Vgrid"]
        if params.get('v_display_min') is None: params['v_display_min'] = float(np.min(bias_vals))
        if params.get('v_display_max') is None: params['v_display_max'] = float(np.max(bias_vals))
        
        df_res = pd.DataFrame({
            "filename": [Path(p).name for p in result["paths"]],
            "pos_nm": result["pos_nm"],
            "Ec": result["Ec_raw"],
            "Ev": np.nan, "Eg": np.nan,
            "Ec_raw": result["Ec_raw"],
            "Ev_raw": np.nan, "Eg_raw": np.nan
        })

        CURRENT_DATA.update({
            'df_res': df_res, 'bias': result["Vgrid"], 'pos_nm': result["pos_nm"],
            'M': result["imap_s"], 'M_raw': result["imap"], 'I_raw': result.get("I_map"), 'tool': tool, 'folder': folder, 'filename_prefix': filename_prefix, 'log_li': log_li, 'gn_norm': gn_norm,
            'm_sigma': m_sigma # 记录平滑量级
        })
        update_table_view(df_res)
        display_figure(figs) # 显示两个独立的图

    elif tool == "newBG":
        valid_keys_newbg = [
            'pattern', 'method', 'eg_min', 'eg_max', 'v_center_min', 'v_center_max', 'delta_E', 'min_gap', 'sigma_level', 'fit_window', 'smooth_window', 
            'use_auto_params', 'sigma_map_bias', 'sigma_map_pos', 'slope_threshold',
            'run_len', 'run_len_cb', 'baseline_k', 'amp_k', 'r2_min', 
            'curv_k', 'curv_percentile', 'run_len_curv', 'vb_q', 'sigma_multiplier',
            'v_display_min', 'v_display_max', 'sigma_v_min', 'sigma_v_max', 'stack_offset_k'
        ]
        filtered_newbg_params = {k: v for k, v in params.items() if k in valid_keys_newbg}
        # 传入自定义参数
        filtered_newbg_params.update({'bias_col': b_col, 'curr_col': c_col, 'lix_col': l_col, 
                                     'sigma_map_bias': m_sigma, 'sigma_map_pos': m_sigma, 'gn_norm': gn_norm, 'lam_factor': lam_factor,
                                     'sigma_v_min': s_v_min, 'sigma_v_max': s_v_max})
        bias, pos_nm, M, df_res, M_raw, I_raw = newbg_mod.process_line_folder(folder, **filtered_newbg_params)

        # 自动适配显示范围 (如果尚未设定)
        if params.get('v_display_min') is None: params['v_display_min'] = float(np.min(bias))
        if params.get('v_display_max') is None: params['v_display_max'] = float(np.max(bias))
        v_min, v_max = params.get('v_display_min'), params.get('v_display_max')

        CURRENT_DATA.update({
            'df_res': df_res, 'bias': bias, 'pos_nm': pos_nm, 'M': M, 'M_raw': M_raw, 'I_raw': I_raw,
            'tool': tool, 'folder': folder, 'filename_prefix': filename_prefix, 'log_li': log_li, 'gn_norm': gn_norm,
            'm_sigma': m_sigma # 记录平滑量级
        })
        update_table_view(df_res)
        figs = newbg_mod.plot_final(bias, pos_nm, M, df_res, title_prefix=f"STS line", show_gap_curve=True, 
                                    log_li=log_li, gn_norm=gn_norm, lam_factor=lam_factor, sigma_multiplier=s_mult,
                                    sigma_v_min=s_v_min, sigma_v_max=s_v_max,
                                    v_display_min=v_min, v_display_max=v_max)
        display_figure(figs) # 显示两个独立的图
        update_stats_label(compute_eg_stats(df_res), folder.name)

    elif tool == "line":
        figs = run_line_analysis(folder, tool, save_png, is_batch, bias_col=b_col, lix_col=l_col, 
                                 log_li=log_li, gn_norm=gn_norm, lam_factor=lam_factor, sigma_multiplier=s_mult,
                                 v_display_min=v_min, v_display_max=v_max,
                                 sigma_v_min=s_v_min, sigma_v_max=s_v_max,
                                 stack_offset_k=st_k, filename_prefix=filename_prefix)
        update_table_view(None)
        stats_var.set("")
        display_figure(figs)
        
    elif tool == "Line CBM VBM":
        try:
            log(get_text('processing_log'))
            # 尝试猜测后向列名
            b_bwd_col = l_col.replace("Forward", "Backward").replace(" (A)", " [bwd] (A)") if "Forward" in l_col else "LI Demod 1 X [bwd] (A)"
            
            lix_fwd, lix_bwd, bias_vals, file_names, metadata = cbm_vbm_mod.process_all_files(
                str(folder), False, bias_col=b_col, fwd_col=l_col, bwd_col=b_bwd_col, curr_col=c_col, gn_norm=gn_norm, lam_factor=lam_factor
            )
            if not lix_fwd: return
            log(get_text('complete_log', n_lines=len(lix_fwd), n_points=len(bias_vals)))
            log(get_text('generating_plot_log'))
            
            # 自动适配显示范围 (如果尚未设定)
            if params.get('v_display_min') is None: params['v_display_min'] = float(np.min(bias_vals))
            if params.get('v_display_max') is None: params['v_display_max'] = float(np.max(bias_vals))
            v_min, v_max = params.get('v_display_min'), params.get('v_display_max')

            # 过滤参数，只传递函数接受的参数
            valid_keys = [
                'method', 'sigma_level', 'fit_window', 'smooth_window', 
                'v_center_min', 'v_center_max', 'delta_E', 'min_gap', 'slope_threshold',
                'run_len', 'run_len_cb', 'baseline_k', 'amp_k', 'r2_min', 
                'curv_k', 'curv_percentile', 'run_len_curv', 'vb_q', 'sigma_multiplier',
                'v_display_min', 'v_display_max', 'sigma_v_min', 'sigma_v_max', 'stack_offset_k'
            ]
            filtered_params = {k: v for k, v in params.items() if k in valid_keys}
            
            figs, cbm_list, vbm_list = cbm_vbm_mod.plot_lix_with_offset_log(
                lix_fwd, lix_bwd, bias_vals, file_names, str(folder), False,
                show_plot=False, gn_norm=gn_norm, lam_factor=lam_factor, **filtered_params
            )
            
            ec_arr = np.array(cbm_list, dtype=float)
            ev_arr = np.array(vbm_list, dtype=float)
            eg_arr = ec_arr - ev_arr
            df_res = pd.DataFrame({
                "filename": file_names,
                "Ec": ec_arr, "Ev": ev_arr, "Eg": eg_arr,
                "Ec_raw": ec_arr.copy(), "Ev_raw": ev_arr.copy(), "Eg_raw": eg_arr.copy()
            })
            CURRENT_DATA.update({'df_res': df_res, 'bias': bias_vals, 'pos_nm': np.arange(len(file_names)),
                                 'M': np.stack(lix_fwd, axis=1), 'tool': tool, 'folder': folder, 'filename_prefix': filename_prefix, 'gn_norm': gn_norm})
            update_table_view(df_res)
            display_figure(figs)
        except Exception as e:
            log(f"Error running Line CBM VBM: {str(e)}")
    log(get_text('batch_complete_log') + str(folder) + "\n")


def open_param_window():
    """打开参数设置窗口"""
    tool = tool_var.get()
    if tool not in ["Ec_log", "newBG", "Line CBM VBM", "line"]:
        messagebox.showinfo(get_text('msg_hint'), "This tool has no parameters.")
        return

    # 各参数仅在对应方法被选中时才需要显示；未列出的参数视为所有方法通用
    METHOD_ROW_TAGS = {
        'sigma_level': ('Adaptive Thresholding', 'Linear Extrapolation'),
        'delta_E': ('Adaptive Thresholding', 'Linear Extrapolation'),
        'slope_threshold': ('Slope Detection',),
        'fit_window': ('Slope Detection',),
        'run_len': ('Slope Detection',),
        'run_len_cb': ('Slope Detection',),
        'baseline_k': ('Slope Detection',),
        'amp_k': ('Slope Detection',),
        'r2_min': ('Slope Detection',),
        'curv_k': ('Slope Detection',),
        'curv_percentile': ('Slope Detection',),
        'run_len_curv': ('Slope Detection',),
        'vb_q': ('Slope Detection',),
    }

    param_window = tk.Toplevel(root)
    param_window.title(get_text('param_window_title', tool=tool))
    param_window.geometry("400x500")

    param_frame = tk.Frame(param_window, padx=10, pady=10)
    param_frame.pack(fill="both", expand=True)

    scrollbar = tk.Scrollbar(param_frame)
    scrollbar.pack(side="right", fill="y")

    canvas = tk.Canvas(param_frame, yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=canvas.yview)

    inner_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    params_to_edit = {
        "Ec_log": EC_PARAMS,
        "newBG": NEWBG_PARAMS,
        "Line CBM VBM": CBM_VBM_PARAMS,
        "line": EC_PARAMS
    }[tool]

    vars_dict = {}
    var_types = {}
    optional_double_keys = {'v_display_min', 'v_display_max'}

    def sync_current_values():
        """把控件里尚未保存的输入同步回 params_to_edit，避免切换方法重建界面时丢失已填内容"""
        for key, var in vars_dict.items():
            if key == 'method':
                continue
            raw_value = var.get()
            var_type = var_types.get(key)
            try:
                if var_type == "double":
                    text_value = str(raw_value).strip()
                    if text_value == "":
                        params_to_edit[key] = None if key in optional_double_keys else (1.2 if key == 'stack_offset_k' else 0.0)
                    else:
                        value = float(text_value)
                        params_to_edit[key] = safe_stack_offset_k(value) if key == 'stack_offset_k' else value
                elif var_type == "int":
                    text_value = str(raw_value).strip()
                    params_to_edit[key] = int(text_value) if text_value != "" else 0
                else:
                    params_to_edit[key] = raw_value
            except ValueError:
                if key == 'stack_offset_k':
                    params_to_edit[key] = 1.2
                elif key in optional_double_keys:
                    params_to_edit[key] = None

    def add_row(label_key, param_key, row_idx, var_type="double"):
        tk.Label(inner_frame, text=get_text(label_key)).grid(row=row_idx, column=0, sticky="w", padx=5, pady=5)

        if var_type == "double":
            val = params_to_edit[param_key]
            var = tk.StringVar(value="" if val is None else str(val))
        elif var_type == "int":
            val = params_to_edit[param_key]
            var = tk.StringVar(value="" if val is None else str(val))
        elif var_type == "bool":
            val = params_to_edit[param_key]
            var = tk.BooleanVar(value=val if val is not None else False)
        else:
            val = params_to_edit[param_key]
            var = tk.StringVar(value=val if val is not None else "")

        if param_key == 'method':
            # 核心方法映射
            method_map = {
                get_text('method_adaptive'): "Adaptive Thresholding",
                get_text('method_extrapolation'): "Linear Extrapolation",
                get_text('method_slope'): "Slope Detection"
            }
            # 建立反向映射用于显示当前值
            reverse_map = {v: k for k, v in method_map.items()}

            display_var = tk.StringVar(value=reverse_map.get(params_to_edit[param_key], get_text('method_adaptive')))
            methods_display = list(method_map.keys())

            combo = ttk.Combobox(inner_frame, textvariable=display_var, values=methods_display, state="readonly")
            combo.grid(row=row_idx, column=1, padx=5, pady=5)

            def on_method_change(*args):
                new_method = method_map[display_var.get()]
                if new_method == params_to_edit.get(param_key):
                    return
                sync_current_values()
                params_to_edit[param_key] = new_method
                # 延迟到当前事件处理完毕后再重建界面，避免在 trace 回调中销毁触发它的控件
                param_window.after_idle(render_rows)

            display_var.trace_add("write", on_method_change)
            var_types[param_key] = var_type
            vars_dict[param_key] = display_var # 这里存 display_var 没关系，只要 trace 处理了实际值
        elif var_type == "bool":
            tk.Checkbutton(inner_frame, variable=var).grid(row=row_idx, column=1, padx=5, pady=5, sticky="w")
            var_types[param_key] = var_type
            vars_dict[param_key] = var
        else:
            tk.Entry(inner_frame, textvariable=var).grid(row=row_idx, column=1, padx=5, pady=5)
            var_types[param_key] = var_type
            vars_dict[param_key] = var

    # 完整参数行定义（含所有方法的参数，实际显示时按当前方法过滤）
    common_rows = [
        ('pattern', 'pattern', 'string'),
        ('method', 'method', 'string'),
        ('lam_factor', 'lam_factor', 'double'),
        ('map_smooth_sigma', 'map_smooth_sigma', 'double'),
        ('v_display_min', 'v_display_min', 'double'),
        ('v_display_max', 'v_display_max', 'double'),
        ('sigma_v_min', 'sigma_v_min', 'double'),
        ('sigma_v_max', 'sigma_v_max', 'double'),
        ('stack_offset_k', 'stack_offset_k', 'double'),
        ('sigma', 'sigma_level', 'double'),
        ('slope_threshold', 'slope_threshold', 'double'),
        ('fit_window', 'fit_window', 'double'),
        ('smooth_window', 'smooth_window', 'int'),
        ('v_center_min', 'v_center_min', 'double'),
        ('v_center_max', 'v_center_max', 'double'),
        ('delta_E', 'delta_E', 'double'),
        ('sigma_multiplier', 'sigma_multiplier', 'double'),
        ('run_len', 'run_len', 'int'),
        ('run_len_cb', 'run_len_cb', 'int'),
        ('baseline_k', 'baseline_k', 'double'),
        ('amp_k', 'amp_k', 'double'),
        ('r2_min', 'r2_min', 'double'),
        ('curv_k', 'curv_k', 'double'),
        ('curv_percentile', 'curv_percentile', 'double'),
        ('run_len_curv', 'run_len_curv', 'int'),
        ('vb_q', 'vb_q', 'double'),
    ]

    if tool == "Ec_log":
        # 移除通用的 v_center_min/max，因为我们要用特定的 ec_center_min/max 标签显示
        rows_all = [r for r in common_rows if r[1] not in ['v_center_min', 'v_center_max']]
        # 在原位置附近插入特定的 Ec 标签
        rows_all.insert(12, ('ec_center_min', 'v_center_min', 'double'))
        rows_all.insert(13, ('ec_center_max', 'v_center_max', 'double'))
    elif tool == "newBG":
        rows_all = common_rows[:2] + [('eg_min', 'eg_min', 'double'), ('eg_max', 'eg_max', 'double')] + common_rows[2:]
    elif tool == "Line CBM VBM":
        rows_all = common_rows + [('min_gap', 'min_gap', 'double')]
    elif tool == "line":
        # line 模式主要关心文件匹配和对数平移以及显示范围，没有 method 选择，全部参数通用
        rows_all = [
            ('pattern', 'pattern', 'string'),
            ('lam_factor', 'lam_factor', 'double'),
            ('v_display_min', 'v_display_min', 'double'),
            ('v_display_max', 'v_display_max', 'double'),
            ('sigma_v_min', 'sigma_v_min', 'double'),
            ('sigma_v_max', 'sigma_v_max', 'double'),
            ('stack_offset_k', 'stack_offset_k', 'double'),
            ('sigma_multiplier', 'sigma_multiplier', 'double'),
        ]

    def visible_rows():
        current_method = params_to_edit.get('method')
        return [r for r in rows_all if current_method in METHOD_ROW_TAGS.get(r[1], (current_method,))]

    def render_rows():
        for w in inner_frame.winfo_children():
            w.destroy()
        vars_dict.clear()
        var_types.clear()
        rows = visible_rows()
        for i, (lbl, key, vtype) in enumerate(rows):
            add_row(lbl, key, i, vtype)
        tk.Button(inner_frame, text=get_text('save_params'), command=save_params, width=15).grid(row=len(rows), column=0, columnspan=2, pady=20)
        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def save_params():
        sync_current_values()
        param_window.destroy()

    render_rows()

def run_analysis():
    folder = folder_var.get().strip()
    if not folder:
        messagebox.showwarning(get_text('msg_hint'), get_text('select_folder'))
        return

    root_folder = Path(folder)
    if not root_folder.exists():
        messagebox.showerror(get_text('msg_err'), f"Folder does not exist: {root_folder}")
        return

    tool = tool_var.get()
    save_png = True  # 默认保存PNG
    is_batch = False  # 默认非批量模式

    log_text.configure(state="normal")
    log_text.delete("1.0", "end")
    log_text.configure(state="disabled")
    stats_var.set("")

    try:
        # 单文件夹模式: 使用用户选择的文件夹名字作为文件名前缀
        run_single_folder(
            folder=root_folder,
            tool=tool,
            save_png=save_png,
            is_batch=False,
            filename_prefix=root_folder.name,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror(get_text('msg_err'), f"{e}")


# ========== 简化版 GUI 搭建 ==========

UI_WIDGETS = {}

def save_figure_manually(fig_idx):
    """手动保存指定的 Figure"""
    if 'figs' not in CURRENT_DATA or fig_idx >= len(CURRENT_DATA['figs']):
        return
    
    fig = CURRENT_DATA['figs'][fig_idx]
    if fig is None:
        messagebox.showinfo(get_text('msg_hint'), "No figure available. Please run analysis first.")
        return
    
    # 在 Windows 下，defaultextension 是关键：
    # 如果用户在下拉框选了 PDF，不输入后缀时，系统会根据所选 filter 自动补全
    file_path = filedialog.asksaveasfilename(
        defaultextension=".svg",
        filetypes=[
            ("SVG files", "*.svg"),
            ("PDF files", "*.pdf"),
            ("PNG files", "*.png")
        ],
        title=get_text(f'save_fig{fig_idx+1}')
    )
    
    if file_path:
        try:
            # 移除之前错误的手动强行追加 .svg 的代码，让 filedialog 自行处理后缀补全
            
            # 确保保存前绘制一次 canvas
            if hasattr(fig, 'canvas') and fig.canvas is not None:
                fig.canvas.draw()
                
            fig.savefig(file_path, bbox_inches="tight")
            log(get_text('save_success', path=file_path))
            messagebox.showinfo(get_text('msg_hint'), get_text('save_success', path=file_path))
        except Exception as e:
            messagebox.showerror(get_text('msg_err'), f"Failed to save figure: {str(e)}")


def save_results_manually():
    """手动保存带隙结果到 TXT"""
    selected_tool = tool_var.get() if tool_var is not None else CURRENT_DATA.get('tool')
    if selected_tool == "line":
        line_plot_df = CURRENT_DATA.get('line_plot_df')
        if CURRENT_DATA.get('tool') != "line" or line_plot_df is None or line_plot_df.empty:
            messagebox.showinfo(get_text('msg_hint'), get_text('run_analysis_hint'))
            return

        prefix = CURRENT_DATA.get('filename_prefix') or "line"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")],
            initialfile=f"{prefix}_line_plot_data.txt",
            title=get_text('save_line_plot_data')
        )
        if file_path:
            sep = "," if Path(file_path).suffix.lower() == ".csv" else "\t"
            line_plot_df.to_csv(file_path, sep=sep, index=False, float_format='%.10g')
            log(get_text('line_plot_data_saved') + str(file_path))
        return

    if CURRENT_DATA['df_res'] is None:
        messagebox.showinfo(get_text('msg_hint'), get_text('run_analysis_hint'))
        return
    
    file_path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv")],
        title=get_text('save_results')
    )
    if file_path:
        df = CURRENT_DATA['df_res']
        df.to_csv(file_path, sep='\t', index=False, float_format='%.6f')
        log(get_text('updated_txt_log') + str(file_path))


def update_save_results_button_text(*args):
    """根据当前分析模式更新第三个保存按钮的含义。"""
    if 'save_results_btn' not in UI_WIDGETS:
        return
    text_key = 'save_line_plot_data' if tool_var is not None and tool_var.get() == "line" else 'save_results'
    UI_WIDGETS['save_results_btn'].config(text=get_text(text_key))


def switch_language(event=None):
    """切换语言并更新所有 UI 文本"""
    global CURRENT_LANG
    new_lang = UI_WIDGETS['lang_combo'].get()
    if new_lang == "中文":
        CURRENT_LANG = 'zh'
    else:
        CURRENT_LANG = 'en'
    
    # 更新主窗口标题
    root.title(get_text('title'))
    
    # 更新各控件文本
    UI_WIDGETS['folder_lbl'].config(text=get_text('folder_label'))
    UI_WIDGETS['browse_btn'].config(text=get_text('browse_btn'))
    UI_WIDGETS['tool_lbl'].config(text=get_text('tool_label'))
    UI_WIDGETS['display_lbl'].config(text=get_text('display_options'))
    UI_WIDGETS['param_btn'].config(text=get_text('param_btn'))
    UI_WIDGETS['run_btn'].config(text=get_text('run_btn'))
    UI_WIDGETS['update_btn'].config(text=get_text('update_btn'))
    UI_WIDGETS['save_fig1_btn'].config(text=get_text('save_fig1'))
    UI_WIDGETS['save_fig2_btn'].config(text=get_text('save_fig2'))
    update_save_results_button_text()
    UI_WIDGETS['table_frm'].config(text=get_text('table_title'))
    UI_WIDGETS['log_lbl'].config(text=get_text('log_label'))
    UI_WIDGETS['lang_lbl'].config(text=get_text('lang_label'))
    UI_WIDGETS['log_li_chk'].config(text=get_text('log_li_label'))
    UI_WIDGETS['gn_norm_chk'].config(text=get_text('gn_norm_label'))
    
    # 更新列选择文本
    UI_WIDGETS['bias_lbl'].config(text=get_text('bias_col'))
    UI_WIDGETS['curr_lbl'].config(text=get_text('curr_col'))
    UI_WIDGETS['lix_lbl'].config(text=get_text('lix_col'))
    
    # 更新分析类型 ComboBox 选项 (如果正在显示，某些工具可能不支持)
    # 这里我们只更新主界面的 tool_box (如果需要翻译分析类型，但目前分析类型是固有的 ID)
    
    # 如果参数窗口开启，更新其内容 (由于参数窗口通常是模态或即时的，
    # 简单的做法是提示用户重新打开参数窗口，或者这里尝试查找并更新)
    
    # 更新 Treeview 表头
    result_tree.heading("id", text=get_text('id_col'))
    result_tree.heading("filename", text=get_text('file_col'))
    result_tree.heading("ec", text=get_text('ec_col'))
    result_tree.heading("ev", text=get_text('ev_col'))
    result_tree.heading("eg", text=get_text('eg_col'))
    
    # 更新占位符（如果存在）
    if 'placeholder_label' in UI_WIDGETS and UI_WIDGETS['placeholder_label'].winfo_exists():
        UI_WIDGETS['placeholder_label'].config(text=get_text('placeholder'))

def main():
    global folder_var, tool_var
    global bias_col_var, curr_col_var, lix_col_var
    global log_text, stats_var, log_li_var, gn_norm_var
    global root
    global canvas_frame, canvas_frames
    global result_tree

    root = tk.Tk()
    root.title(get_text('title'))
    root.geometry("1200x800")

    # ---------- 变量 ----------
    folder_var = tk.StringVar()
    tool_var = tk.StringVar(value="Ec_log")
    bias_col_var = tk.StringVar(value="Bias calc (V)")
    curr_col_var = tk.StringVar(value="Current (A)")
    lix_col_var = tk.StringVar(value="LI Demod 1 X (A)")
    stats_var = tk.StringVar(value="")
    log_li_var = tk.BooleanVar(value=False)
    gn_norm_var = tk.BooleanVar(value=False)

    # 主容器：左侧参数和日志，右侧绘图
    main_pane = tk.PanedWindow(root, orient="horizontal", sashrelief="raised", sashwidth=4)
    main_pane.pack(fill="both", expand=True)

    # 左侧框架
    left_frame = tk.Frame(main_pane, width=270)
    main_pane.add(left_frame, width=270)

    # 右侧框架（绘图区域）
    right_frame = tk.Frame(main_pane)
    main_pane.add(right_frame)

    # ---------- 左侧：参数区域 ----------
    frm_top = tk.Frame(left_frame)
    frm_top.pack(side="top", fill="x", padx=8, pady=8)

    row = 0
    
    # 语言选择 (新增)
    UI_WIDGETS['lang_lbl'] = tk.Label(frm_top, text=get_text('lang_label'))
    UI_WIDGETS['lang_lbl'].grid(row=row, column=0, sticky="e", padx=5, pady=5)
    lang_combo = ttk.Combobox(frm_top, values=["中文", "English"], state="readonly", width=10)
    lang_combo.set("中文")
    lang_combo.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    lang_combo.bind("<<ComboboxSelected>>", switch_language)
    UI_WIDGETS['lang_combo'] = lang_combo
    row += 1
    
    # 数据文件夹
    UI_WIDGETS['folder_lbl'] = tk.Label(frm_top, text=get_text('folder_label'))
    UI_WIDGETS['folder_lbl'].grid(row=row, column=0, sticky="e", padx=5, pady=5)
    tk.Entry(frm_top, textvariable=folder_var, width=20).grid(row=row, column=1, padx=5, pady=5)
    UI_WIDGETS['browse_btn'] = tk.Button(frm_top, text=get_text('browse_btn'), command=choose_folder)
    UI_WIDGETS['browse_btn'].grid(row=row, column=2, padx=5, pady=5)
    row += 1

    # 数据选择 (新增)
    UI_WIDGETS['bias_lbl'] = tk.Label(frm_top, text=get_text('bias_col'))
    UI_WIDGETS['bias_lbl'].grid(row=row, column=0, sticky="e", padx=5, pady=5)
    bias_combo = ttk.Combobox(frm_top, textvariable=bias_col_var, state="readonly", width=18)
    bias_combo.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    UI_WIDGETS['bias_combo'] = bias_combo
    row += 1
    
    UI_WIDGETS['curr_lbl'] = tk.Label(frm_top, text=get_text('curr_col'))
    UI_WIDGETS['curr_lbl'].grid(row=row, column=0, sticky="e", padx=5, pady=5)
    curr_combo = ttk.Combobox(frm_top, textvariable=curr_col_var, state="readonly", width=18)
    curr_combo.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    UI_WIDGETS['curr_combo'] = curr_combo
    row += 1
    
    UI_WIDGETS['lix_lbl'] = tk.Label(frm_top, text=get_text('lix_col'))
    UI_WIDGETS['lix_lbl'].grid(row=row, column=0, sticky="e", padx=5, pady=5)
    lix_combo = ttk.Combobox(frm_top, textvariable=lix_col_var, state="readonly", width=18)
    lix_combo.grid(row=row, column=1, sticky="w", padx=5, pady=5)
    UI_WIDGETS['lix_combo'] = lix_combo
    row += 1

    # 分析类型
    UI_WIDGETS['tool_lbl'] = tk.Label(frm_top, text=get_text('tool_label'))
    UI_WIDGETS['tool_lbl'].grid(row=row, column=0, sticky="e", padx=5, pady=5)
    tool_box = ttk.Combobox(
        frm_top,
        textvariable=tool_var,
        values=["Ec_log", "newBG", "line", "Line CBM VBM"],
        state="readonly",
        width=18,
    )
    tool_box.grid(row=row, column=1, padx=5, pady=5, sticky="w")
    row += 1

    # 显示选项 (原保存选项位置)
    UI_WIDGETS['display_lbl'] = tk.Label(frm_top, text=get_text('display_options'))
    UI_WIDGETS['display_lbl'].grid(row=row, column=0, sticky="e", padx=5, pady=5)
    display_frame = tk.Frame(frm_top)
    display_frame.grid(row=row, column=1, columnspan=2, sticky="w", padx=5, pady=5)
    
    UI_WIDGETS['log_li_chk'] = tk.Checkbutton(display_frame, text=get_text('log_li_label'), variable=log_li_var)
    UI_WIDGETS['log_li_chk'].grid(row=0, column=0, sticky="w")
    
    UI_WIDGETS['gn_norm_chk'] = tk.Checkbutton(display_frame, text=get_text('gn_norm_label'), variable=gn_norm_var)
    UI_WIDGETS['gn_norm_chk'].grid(row=1, column=0, sticky="w")
    row += 1

    # 按钮区域
    btn_frame = tk.Frame(frm_top)
    btn_frame.grid(row=row, column=0, columnspan=3, pady=10)
    
    UI_WIDGETS['param_btn'] = tk.Button(btn_frame, text=get_text('param_btn'), command=open_param_window, width=8)
    UI_WIDGETS['param_btn'].pack(side="left", padx=2)
    UI_WIDGETS['run_btn'] = tk.Button(btn_frame, text=get_text('run_btn'), command=run_analysis, width=12, bg="#4CAF50", fg="white")
    UI_WIDGETS['run_btn'].pack(side="left", padx=2)
    UI_WIDGETS['update_btn'] = tk.Button(btn_frame, text=get_text('update_btn'), command=update_from_table, width=8, bg="#2196F3", fg="white")
    UI_WIDGETS['update_btn'].pack(side="left", padx=2)
    row += 1

    # 手动保存按钮 (修改布局和逻辑)
    save_btn_frame = tk.Frame(frm_top)
    save_btn_frame.grid(row=row, column=0, columnspan=3, sticky="ew", padx=5, pady=2)
    
    UI_WIDGETS['save_fig1_btn'] = tk.Button(save_btn_frame, text=get_text('save_fig1'), command=lambda: save_figure_manually(0), width=10)
    UI_WIDGETS['save_fig1_btn'].pack(side="left", expand=True, padx=2)
    
    UI_WIDGETS['save_fig2_btn'] = tk.Button(save_btn_frame, text=get_text('save_fig2'), command=lambda: save_figure_manually(1), width=10)
    UI_WIDGETS['save_fig2_btn'].pack(side="left", expand=True, padx=2)

    UI_WIDGETS['save_results_btn'] = tk.Button(save_btn_frame, text=get_text('save_results'), command=save_results_manually, width=10)
    UI_WIDGETS['save_results_btn'].pack(side="left", expand=True, padx=2)
    tool_var.trace_add("write", update_save_results_button_text)
    update_save_results_button_text()
    row += 1

    # Eg 统计信息
    tk.Label(frm_top, textvariable=stats_var, fg="blue", wraplength=250).grid(
        row=row, column=0, columnspan=3, sticky="w", padx=5, pady=5
    )
    row += 1

    # ---------- 左侧：结果表格 ----------
    UI_WIDGETS['table_frm'] = tk.LabelFrame(left_frame, text=get_text('table_title'))
    UI_WIDGETS['table_frm'].pack(side="top", fill="both", expand=True, padx=8, pady=4)

    columns = ("id", "filename", "ec", "ev", "eg")
    result_tree = ttk.Treeview(UI_WIDGETS['table_frm'], columns=columns, show="headings", height=10)
    
    result_tree.heading("id", text=get_text('id_col'))
    result_tree.heading("filename", text=get_text('file_col'))
    result_tree.heading("ec", text=get_text('ec_col'))
    result_tree.heading("ev", text=get_text('ev_col'))
    result_tree.heading("eg", text=get_text('eg_col'))
    
    result_tree.column("id", width=30, anchor="center")
    result_tree.column("filename", width=80)
    result_tree.column("ec", width=45, anchor="center")
    result_tree.column("ev", width=45, anchor="center")
    result_tree.column("eg", width=45, anchor="center")
    
    tree_scroll = tk.Scrollbar(UI_WIDGETS['table_frm'], orient="vertical", command=result_tree.yview)
    result_tree.configure(yscrollcommand=tree_scroll.set)
    
    result_tree.pack(side="left", fill="both", expand=True)
    tree_scroll.pack(side="right", fill="y")
    
    result_tree.bind("<Double-1>", on_tree_double_click)

    # ---------- 左侧：日志输出 ----------
    frm_log = tk.Frame(left_frame)
    frm_log.pack(side="top", fill="x", padx=8, pady=4)

    UI_WIDGETS['log_lbl'] = tk.Label(frm_log, text=get_text('log_label'))
    UI_WIDGETS['log_lbl'].pack(anchor="w")
    log_text = tk.Text(frm_log, height=8, state="disabled")
    
    # 添加滚动条
    scrollbar = tk.Scrollbar(frm_log, command=log_text.yview)
    scrollbar.pack(side="right", fill="y")
    log_text.config(yscrollcommand=scrollbar.set)
    
    log_text.pack(side="left", fill="both", expand=True)

    # ---------- 右侧：绘图区域 ----------
    # 将右侧绘图区域改为 PanedWindow，支持两个独立的 Figure
    plot_pane = tk.PanedWindow(right_frame, orient="horizontal", sashrelief="raised", sashwidth=4)
    plot_pane.pack(fill="both", expand=True)
    
    canvas_frame1 = tk.Frame(plot_pane, bg="white")
    canvas_frame2 = tk.Frame(plot_pane, bg="white")
    
    # 使用 stretch="always" 确保初始平分
    plot_pane.add(canvas_frame1, stretch="always")
    plot_pane.add(canvas_frame2, stretch="always")
    
    # 更新全局引用的 canvas_frame 以及新增 canvas_frames 列表
    global canvas_frame, canvas_frames
    canvas_frame = canvas_frame1
    canvas_frames = [canvas_frame1, canvas_frame2]
    
    # 初始提示文字
    placeholder_label = tk.Label(canvas_frame, text=get_text('placeholder'), bg="white", font=("Arial", 14))
    placeholder_label.place(relx=0.5, rely=0.5, anchor="center")
    UI_WIDGETS['placeholder_label'] = placeholder_label

    def on_closing():
        """程序关闭时的回调，确保彻底退出"""
        try:
            plt.close('all') # 关闭所有 matplotlib 图表
            root.quit()      # 停止 mainloop
            root.destroy()   # 销毁窗口
        except Exception:
            pass
        finally:
            os._exit(0)      # 强制退出进程，防止后台残留

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()


if __name__ == "__main__":
    main()
