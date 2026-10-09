import numpy as np
import matplotlib.pyplot as plt
import os
import re
from band_edge_core import apply_gn_normalization, apply_vertical_shift


def _safe_stack_offset_k(stack_offset_k, default=1.2):
    try:
        stack_offset_k = float(stack_offset_k)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(stack_offset_k) or stack_offset_k <= 0:
        return default
    return stack_offset_k


def read_dat_file(file_path):
    """读取单个.dat文件并提取数据和元数据 (改进版)"""
    import pandas as pd
    from io import StringIO
    
    # 尝试不同编码读取
    encodings = ['utf8', 'latin-1', 'cp1252']
    lines = []
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                lines = f.read().splitlines()
            if lines: break
        except Exception:
            continue
            
    if not lines:
        return {}, {}

    x = y = z = None
    data_start = -1
    for i, line in enumerate(lines):
        t = line.strip()
        if "[DATA]" in t:
            data_start = i + 1
            break
        
        # 提取元数据 (坐标)
        if t.startswith("X (m)") or t.startswith("Y (m)") or (t.startswith("Z (m)") and "offset" not in t):
            parts = t.split()
            if len(parts) >= 2:
                try:
                    val = float(parts[-1].replace(",", "."))
                    if t.startswith("X (m)"): x = val
                    elif t.startswith("Y (m)"): y = val
                    elif t.startswith("Z (m)"): z = val
                except ValueError: pass
    
    if data_start == -1:
        return {}, {}

    # 数据从 [DATA] 后的第一行开始
    data_lines = lines[data_start:]
    header_idx = 0
    for i, line in enumerate(data_lines):
        if line.strip():
            header_idx = i
            break
            
    data_str = "\n".join(data_lines[header_idx:])
    
    try:
        df = pd.read_csv(StringIO(data_str), sep="\t")
        if len(df.columns) <= 1:
            df = pd.read_csv(StringIO(data_str), sep=None, engine='python')
    except Exception:
        df = pd.read_csv(StringIO(data_str), sep=None, engine='python')
        
    data_dict = {col: df[col].to_numpy() for col in df.columns}
    meta_dict = {"x": x, "y": y, "z": z}
    
    return data_dict, meta_dict

def process_all_files(directory, use_filtered=False, bias_col='Bias calc (V)', fwd_col='LI Demod 1 X (A)', bwd_col='LI Demod 1 X [bwd] (A)', curr_col='Current (A)', gn_norm=False, lam_factor=5.0):
    """处理目录中的所有数据文件
    
    Args:
        directory: 数据文件所在目录
        use_filtered: 是否使用滤波后的数据（[filt]列）
        bias_col: 电压列名
        fwd_col: 前向 LI 列名
        bwd_col: 后向 LI 列名
        curr_col: 电流列名 (用于归一化)
        gn_norm: 是否进行快速归一化 (gN)
        lam_factor: Soft Regularization 的倍率因子
    """
    # 获取所有.dat文件并按文件名中的数字排序
    files = [f for f in os.listdir(directory) if f.endswith('.dat')]
    # 使用正则表达式提取文件名中的数字进行排序
    files.sort(key=lambda x: int(re.search(r'(\d+)\.dat$', x).group(1)) if re.search(r'(\d+)\.dat$', x) else 0)
    
    lix_x_fwd_lines = []
    lix_x_bwd_lines = []
    bias_values = None
    
    # 根据是否使用滤波数据选择对应的列名
    if use_filtered:
        # 这里由于用户自定义了列名，如果勾选了滤波，可能需要一种逻辑来寻找滤波列
        # 为了简单起见，如果用户指定了列名且勾选了滤波，我们尝试在列名后加上 [filt]
        fwd_key = f"{fwd_col} [filt]" if "[filt]" not in fwd_col else fwd_col
        bwd_key = f"{bwd_col} [filt]" if "[filt]" not in bwd_col else bwd_col
        bias_key = bias_col
        curr_key = curr_col
    else:
        fwd_key = fwd_col
        bwd_key = bwd_col
        bias_key = bias_col
        curr_key = curr_col
    
    valid_files = []
    metadata = []
    for file in files:
        file_path = os.path.join(directory, file)
        data_dict, meta_dict = read_dat_file(file_path)
        
        # 检查必要的键是否存在
        required_keys = [bias_key, fwd_key]
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            continue
        
        # 只在第一个文件中获取bias值
        if bias_values is None:
            bias_values = data_dict[bias_key]
        
        # 提取对应的LI Demod 1 X分量
        lix_x_fwd = data_dict[fwd_key]
        lix_x_bwd = data_dict.get(bwd_key, np.zeros_like(lix_x_fwd)) # 后向可能是可选的
        
        if gn_norm and curr_key in data_dict:
            V = data_dict[bias_key]
            I = data_dict[curr_key]
            
            lix_x_fwd = apply_gn_normalization(V, I, lix_x_fwd, lam_factor=lam_factor)
            if bwd_key in data_dict:
                lix_x_bwd = apply_gn_normalization(V, I, lix_x_bwd, lam_factor=lam_factor)

        lix_x_fwd_lines.append(lix_x_fwd)
        lix_x_bwd_lines.append(lix_x_bwd)
        valid_files.append(file)
        metadata.append(meta_dict)
    
    return lix_x_fwd_lines, lix_x_bwd_lines, bias_values, valid_files, metadata

def plot_lix_with_offset_log(lix_fwd_lines, lix_bwd_lines, bias_values, save_directory='.', use_filtered=False, show_plot=True, log_li=False, gn_norm=False, lam_factor=5.0, sigma_multiplier=0.0, v_display_min=None, v_display_max=None, sigma_v_min=-1.0, sigma_v_max=0.2, stack_offset_k=1.2, metadata=None):
    """绘制LI Demod 1 X线图，使用自动计算的偏移量"""
    # 检查数据是否为空
    if len(lix_fwd_lines) == 0 or len(lix_bwd_lines) == 0:
        return None
    stack_offset_k = _safe_stack_offset_k(stack_offset_k)

    # 裁剪电压范围
    if v_display_min is not None and v_display_max is not None:
        mask = (bias_values >= v_display_min) & (bias_values <= v_display_max)
        if np.any(mask):
            bias_values = bias_values[mask]
            lix_fwd_lines = [line[mask] for line in lix_fwd_lines]
            lix_bwd_lines = [line[mask] for line in lix_bwd_lines]

    # 计算偏移量
    if log_li:
        all_lix_data = np.concatenate(lix_fwd_lines + lix_bwd_lines)
        all_abs = np.abs(all_lix_data)
        finite_mask = np.isfinite(all_abs) & (all_abs > 0)
        log_all = np.log10(all_abs[finite_mask]) if np.any(finite_mask) else np.array([0.0])
        std_dev = np.std(log_all)
        if not np.isfinite(std_dev) or std_dev <= 0:
            std_dev = 1.0
        offset_value = std_dev * stack_offset_k # 使用传入的系数
    else:
        # 线性模式下的偏移量估算
        all_lix_data = np.concatenate(lix_fwd_lines + lix_bwd_lines)
        all_abs = np.abs(all_lix_data)
        max_val = np.nanmax(all_abs) if np.any(np.isfinite(all_abs)) else 1e-9
        offset_value = max_val * 0.8 * stack_offset_k

    # 颜色映射
    colors = plt.cm.tab20(np.linspace(0, 1, len(lix_fwd_lines)))

    data_type_suffix = " [Filtered]" if use_filtered else ""
    title_suffix = f" (gN, λ factor={lam_factor})" if gn_norm else ""
    
    # ===================== Separate Figures =====================
    fig1 = plt.figure(figsize=(6, 10))
    ax_fwd = fig1.add_subplot(111)

    fig2 = plt.figure(figsize=(6, 10))
    ax_bwd = fig2.add_subplot(111)

    # 准备元数据文本
    meta_text = ""
    if metadata and len(metadata) > 0:
        m0 = metadata[0]
        mlast = metadata[-1]
        if m0.get('x') is not None:
            meta_text = f"Start: X={m0['x']*1e9:.2f}nm, Y={m0['y']*1e9:.2f}nm\n"
            meta_text += f"End: X={mlast['x']*1e9:.2f}nm, Y={mlast['y']*1e9:.2f}nm"

    # Forward
    for i, (line_data, color) in enumerate(zip(lix_fwd_lines, colors)):
        if log_li:
            # 使用垂直平移优化
            line_shifted, _ = apply_vertical_shift(bias_values, line_data, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
            plot_data = np.log10(np.maximum(line_shifted, 1e-20))
        else:
            plot_data = line_data
            
        offset_line_data = plot_data + i * offset_value
        ax_fwd.plot(bias_values, offset_line_data, color=color, label=f'Line {i+1}', alpha=0.8, linewidth=1)

    ax_fwd.set_title(f'LI Demod 1 X Forward{data_type_suffix}{title_suffix}')
    ax_fwd.set_xlabel('Bias Voltage (V)')
    
    base_label = 'gN' if gn_norm else 'LIx'
    y_label = f'log10({base_label} + offset) + stack_offset' if log_li else f'{base_label} + stack_offset'
    ax_fwd.set_ylabel(y_label)
    ax_fwd.grid(True, alpha=0.3)
    if meta_text:
        ax_fwd.text(0.02, 0.98, meta_text, transform=ax_fwd.transAxes, verticalalignment='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    # Backward
    for i, (line_data, color) in enumerate(zip(lix_bwd_lines, colors)):
        if log_li:
            # 使用垂直平移优化
            line_shifted, _ = apply_vertical_shift(bias_values, line_data, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
            plot_data = np.log10(np.maximum(line_shifted, 1e-20))
        else:
            plot_data = line_data
            
        offset_line_data = plot_data + i * offset_value
        ax_bwd.plot(bias_values, offset_line_data, color=color, label=f'Line {i+1}', alpha=0.8, linewidth=1)

    ax_bwd.set_title(f'LI Demod 1 X Backward{data_type_suffix}{title_suffix}')
    ax_bwd.set_xlabel('Bias Voltage (V)')
    ax_bwd.set_ylabel(y_label)
    ax_bwd.grid(True, alpha=0.3)
    if meta_text:
        ax_bwd.text(0.02, 0.98, meta_text, transform=ax_bwd.transAxes, verticalalignment='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))

    fig1.tight_layout()
    fig2.tight_layout()
    
    if show_plot:
        plt.show()
    
    return [fig1, fig2]
