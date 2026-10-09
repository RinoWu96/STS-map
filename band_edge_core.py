import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

def apply_gn_normalization(V, I, g, lam_factor=5.0):
    """
    应用 Soft Regularization (软正则) 的快速归一化 (gN)
    公式: gN = (dI/dV) * V / (I + sign(I) * lam)
    其中 lam = lam_factor * sigma_I
    """
    V = np.asarray(V)
    I = np.asarray(I)
    g = np.asarray(g)
    
    # 估计 I 的噪声水平 sigma_I
    # 使用高通滤波 (原始信号 - 平滑信号) 的标准差
    if I.ndim == 1:
        I_smooth = gaussian_filter1d(I, sigma=2)
        sigma_I = np.std(I - I_smooth)
    else:
        # 如果是 2D 矩阵 (Nbias, Npos)
        I_smooth = gaussian_filter1d(I, sigma=2, axis=0)
        sigma_I = np.std(I - I_smooth)
        
    lam = lam_factor * sigma_I
    # 避免 lam 太小导致的问题
    lam = max(lam, 1e-20)
    
    # 计算分母: I + sign(I) * lam
    # 为了避免 I=0 时 sign(I)=0 导致分母为 0，我们将 sign(0) 视为 1
    s = np.sign(I)
    if np.isscalar(s):
        if s == 0: s = 1
    else:
        s[s == 0] = 1
        
    I_soft = I + s * lam
    
    # 最终计算 gN
    gn = g * V / I_soft
    return gn

def apply_vertical_shift(V, signal, sigma_multiplier=0.0, sigma_v_min=-1.0, sigma_v_max=0.2):
    """
    垂直平移优化：基于指定电压窗口内数据的标准差进行平移。
    首先对信号取绝对值，以确保对数运算的鲁棒性。
    公式: offset = sigma_multiplier * std(abs(S)[sigma_v_min < V < sigma_v_max])
    """
    V = np.asarray(V)
    # 强制取绝对值，处理实验噪声中的负值
    S = np.abs(np.asarray(signal, dtype=float))
    
    # 找到指定的噪声采样区域
    mask = (V >= sigma_v_min) & (V <= sigma_v_max)
    
    if mask.any() and mask.sum() >= 3:
        sigma = np.std(S[mask], ddof=1)
    else:
        # 如果指定区域无效，回退到 0V 附近 +/- 0.1V
        mask_fallback = np.abs(V) < 0.1
        if mask_fallback.any():
            sigma = np.std(S[mask_fallback], ddof=1)
        else:
            sigma = np.std(S, ddof=1)
        
    offset = sigma_multiplier * sigma
    S_shifted = S + offset
    return S_shifted, offset

def linear_fit_R2(x, y):
    """线性拟合并计算 R2"""
    if len(x) < 3: return None, None, None
    p = np.polyfit(x, y, 1)
    # 计算 R2
    ss_res = np.sum((y - np.polyval(p, x))**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return p[0], p[1], float(R2)

def preprocess_signal(V, signal, smooth_window=11, sigma_multiplier=0.0, sigma_v_min=-1.0, sigma_v_max=0.2):
    """
    预处理信号：排序、垂直平移优化、对数转换并平滑
    改进自 STS_整合.py
    """
    V = np.asarray(V, dtype=float)
    
    # --- 垂直平移优化 ---
    S_shifted, offset = apply_vertical_shift(V, signal, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
    
    # 确保电压升序
    idx = np.argsort(V)
    V, S_shifted = V[idx], S_shifted[idx]
    S_raw_orig = np.asarray(signal, dtype=float)[idx]

    # log 转换
    logS_raw = np.log10(np.maximum(S_shifted, 1e-18))
    
    # 平滑
    win = int(smooth_window)
    if win % 2 == 0:
        win += 1
    if len(logS_raw) > win:
        logS_smooth = savgol_filter(logS_raw, win, polyorder=2)
    else:
        logS_smooth = logS_raw
    
    return V, logS_smooth, logS_raw, S_shifted, offset

def get_plateau_range(V, logS_raw):
    """
    自动识别 Gap 区域的平坦范围
    """
    vmin, vmax = float(V.min()), float(V.max())
    v_abs_max = max(abs(vmin), abs(vmax))

    mid_mask = np.abs(V) < 0.25 * v_abs_max
    if mid_mask.sum() < 10:
        mid_mask = np.abs(V) < 0.35 * v_abs_max
    if mid_mask.sum() < 10:
        mid_mask = np.ones_like(V, dtype=bool)

    log_mid = logS_raw[mid_mask]
    V_mid = V[mid_mask]
    base_guess = float(np.median(log_mid))
    plateau_mask_mid = log_mid < (base_guess + 0.3)
    
    if np.any(plateau_mask_mid):
        v_plateau_min = float(V_mid[plateau_mask_mid].min())
        v_plateau_max = float(V_mid[plateau_mask_mid].max())
    else:
        v_plateau_min, v_plateau_max = -0.3, 0.3

    margin = 0.05 * v_abs_max
    v_center_min = v_plateau_min - margin
    v_center_max = v_plateau_max + margin
    
    # 限制范围
    v_center_min = max(v_center_min, vmin + 0.05 * (vmax - vmin))
    v_center_max = min(v_center_max, vmax - 0.05 * (vmax - vmin))
    
    if v_center_min >= v_center_max:
        v_center_min, v_center_max = vmin * 0.2, vmax * 0.2
        
    return v_center_min, v_center_max

def find_edges_dynamic_core(V, logS, threshold):
    """
    核心寻找边缘逻辑：寻找第一个连续 10 点超过阈值的点并插值
    """
    Ev, Ec = np.nan, np.nan
    
    # --- 导带边缘 (Ec) ---
    mask_cb = (V > 0) & (logS > threshold)
    if np.any(mask_cb):
        idx_cb_all = np.where(mask_cb)[0]
        idx_trigger = None
        for k in idx_cb_all:
            if k < len(logS) - 9 and np.all(logS[k:k+10] > threshold):
                idx_trigger = k
                break
        
        if idx_trigger is not None and idx_trigger > 0:
            v1, v2 = V[idx_trigger-1], V[idx_trigger]
            y1, y2 = logS[idx_trigger-1], logS[idx_trigger]
            if abs(y2 - y1) > 1e-15:
                Ec = v1 + (threshold - y1) * (v2 - v1) / (y2 - y1)
            else:
                Ec = V[idx_trigger]
        elif len(idx_cb_all) > 0:
            Ec = V[idx_cb_all[0]]

    # --- 价带边缘 (Ev) ---
    mask_vb = (V < 0) & (logS > threshold)
    if np.any(mask_vb):
        idx_vb_all = np.where(mask_vb)[0]
        idx_trigger = None
        for k in sorted(idx_vb_all, reverse=True):
            if k >= 9 and np.all(logS[k-9:k+1] > threshold):
                idx_trigger = k
                break
        
        if idx_trigger is not None and idx_trigger < len(logS) - 1:
            v1, v2 = V[idx_trigger], V[idx_trigger+1]
            y1, y2 = logS[idx_trigger], logS[idx_trigger+1]
            if abs(y2 - y1) > 1e-15:
                Ev = v1 + (threshold - y1) * (v2 - v1) / (y2 - y1)
            else:
                Ev = V[idx_trigger]
        elif len(idx_vb_all) > 0:
            Ev = V[idx_vb_all[-1]]
            
    return Ev, Ec

def detect_band_edges_dynamic(V, logS_smooth, logS_raw, S_shifted, sigma_level=2.0, slope_percentile=85, min_gap=0.05, v_center_min=None, v_center_max=None, delta_E_fixed=None, **kwargs):
    """
    方法 1: 自适应阈值法 (Adaptive Thresholding)
    改进：在找到带边后，通过线性拟合与基准线的交点精细化带边位置
    """
    if v_center_min is None or v_center_max is None:
        v_center_min, v_center_max = get_plateau_range(V, logS_raw)

    center_mask = (V >= v_center_min) & (V <= v_center_max)
    if center_mask.sum() < 1:
        v_center_min, v_center_max = get_plateau_range(V, logS_raw)
        center_mask = (V >= v_center_min) & (V <= v_center_max)
    
    if np.any(center_mask):
        data_lin_raw = S_shifted[center_mask]
    else:
        data_lin_raw = S_shifted
        
    base_lin = float(np.mean(data_lin_raw))
    sigma_lin = float(np.std(data_lin_raw, ddof=1)) if len(data_lin_raw) > 1 else 0.0
    
    base = np.log10(base_lin + 1e-18)
    
    if sigma_level is not None and sigma_level > 0:
        thresh_lin = base_lin + sigma_level * sigma_lin
        sigma_log_contrib = np.log10(thresh_lin + 1e-18) - base
    else:
        sigma_log_contrib = 0.0

    left_high = np.percentile(logS_raw[V < v_center_min], 90) if (V < v_center_min).any() else base + 1.0
    right_high = np.percentile(logS_raw[V > v_center_max], 90) if (V > v_center_max).any() else base + 1.0
    span = max(0.5, min(left_high - base, right_high - base))
    
    # 动态调整 delta_E 或使用固定值
    if delta_E_fixed is not None:
        delta_E = delta_E_fixed
    else:
        delta_E_floor = 0.05
        delta_E = max(delta_E_floor, min(0.3 * span, 1.2))
    
    thr_add = max(delta_E, sigma_log_contrib)
    threshold = float(base + thr_add)

    # 计算斜率阈值 (参考 STS_整合.py)
    dlog_dV = np.gradient(logS_raw, V)
    slope_thr = float(np.percentile(np.abs(dlog_dV), slope_percentile))

    # 第一步：自适应阈值判定初始带边
    Ev, Ec = find_edges_dynamic_core(V, logS_raw, threshold)

    # 第二步：线性拟合精细化 (改进逻辑)
    # 拟合 Ec
    a_cb, b_cb, R2_cb = None, None, None
    if not np.isnan(Ec):
        cb_mask = (V >= Ec) & (V <= Ec + delta_E) & (V > 0)
        a_cb, b_cb, R2_cb = linear_fit_R2(V[cb_mask], logS_raw[cb_mask])
        if a_cb is not None and a_cb > 0:
            Ec = (base - b_cb) / a_cb

    # 拟合 Ev
    a_vb, b_vb, R2_vb = None, None, None
    if not np.isnan(Ev):
        vb_mask = (V >= Ev - delta_E) & (V <= Ev) & (V < 0)
        a_vb, b_vb, R2_vb = linear_fit_R2(V[vb_mask], logS_raw[vb_mask])
        if a_vb is not None and a_vb < 0:
            Ev = (base - b_vb) / a_vb

    Eg = Ec - Ev if not (np.isnan(Ec) or np.isnan(Ev)) else np.nan
    if not np.isnan(Eg) and Eg < min_gap: Ev, Ec, Eg = np.nan, np.nan, np.nan

    return Ev, Ec, Eg, {
        "base": base, "threshold": threshold, "slope_thr": slope_thr,
        "v_center_min": v_center_min, "v_center_max": v_center_max,
        "method": "Adaptive Thresholding",
        "R2_vb": R2_vb, "R2_cb": R2_cb
    }

def extract_gap_log_plateau(V, logS_smooth, logS_raw, S_shifted, v_center_min=-0.3, v_center_max=0.3, delta_E=0.15, min_gap=0.1, sigma_level=2.0):
    """
    方法 2: 线性外推法 (Linear Extrapolation)
    """
    center_mask = (V >= v_center_min) & (V <= v_center_max)
    if center_mask.sum() < 1:
        v_center_min_auto, v_center_max_auto = get_plateau_range(V, logS_raw)
        center_mask = (V >= v_center_min_auto) & (V <= v_center_max_auto)

    if center_mask.sum() > 0:
        # 基准线直接在 log 空间统计（中位数），而不是先在线性空间平均再取 log。
        # 线性平均会被正负噪音相互抵消拉向 0，导致 log10(mean) 系统性偏低，
        # 落在 log 坐标图上噪音散点云的下方而非中间。
        floor_log = float(np.median(logS_raw[center_mask]))
        sigma_log = float(np.std(logS_raw[center_mask], ddof=1)) if len(logS_raw[center_mask]) > 1 else 0.0
    else:
        floor_log = float(np.median(logS_raw))
        sigma_log = float(np.std(logS_raw))

    y_thresh_log = floor_log + sigma_level * sigma_log
    y_floor = floor_log

    def find_crossing_index(mask, is_cb=True):
        if not mask.any(): return None
        indices = np.where(mask)[0]
        if is_cb:
            for i in indices:
                # 检查连续10个点大于阈值 (同步 STS_line.py 的鲁棒性)
                if i < len(logS_raw) - 9 and np.all(logS_raw[i:i+10] > y_thresh_log):
                    return i
        else:
            # 价带向负压方向搜索
            for i in sorted(indices, reverse=True):
                # 检查连续10个点（向负压方向延伸）大于阈值
                if i >= 9 and np.all(logS_raw[i-9:i+1] > y_thresh_log):
                    return i
        return None

    idx_ev = find_crossing_index(V < 0, is_cb=False)
    idx_ec = find_crossing_index(V > 0, is_cb=True)

    if idx_ev is None or idx_ec is None:
        return np.nan, np.nan, np.nan, {"method": "Linear Extrapolation", "note": "No 2sigma points"}

    idx_ev_start = min(idx_ev + 1, len(V) - 1)
    if V[idx_ev_start] >= 0: idx_ev_start = idx_ev 
    idx_ec_start = max(idx_ec - 1, 0)
    if V[idx_ec_start] <= 0: idx_ec_start = idx_ec 

    v_ev_start = V[idx_ev_start]
    v_ec_start = V[idx_ec_start]

    vb_mask = (V >= v_ev_start - delta_E) & (V <= v_ev_start) & (V < 0)
    a_vb, b_vb, R2_vb = linear_fit_R2(V[vb_mask], logS_raw[vb_mask])
    if a_vb is not None and a_vb >= 0: a_vb, b_vb, R2_vb = None, None, None

    cb_mask = (V >= v_ec_start) & (V <= v_ec_start + delta_E) & (V > 0)
    a_cb, b_cb, R2_cb = linear_fit_R2(V[cb_mask], logS_raw[cb_mask])
    if a_cb is not None and a_cb <= 0: a_cb, b_cb, R2_cb = None, None, None

    if a_vb is None or a_cb is None:
        return np.nan, np.nan, np.nan, {"method": "Linear Extrapolation", "note": "Fit failed"}

    Evbm, Ecbm = (y_floor - b_vb) / a_vb, (y_floor - b_cb) / a_cb
    Eg = Ecbm - Evbm
    if Eg < min_gap: Evbm, Ecbm, Eg = np.nan, np.nan, np.nan

    return Evbm, Ecbm, Eg, {
        "method": "Linear Extrapolation", 
        "floor_log": floor_log,
        "R2_vb": R2_vb, "R2_cb": R2_cb
    }

def _first_run(mask: np.ndarray, run: int, from_left: bool = True):
    """辅助函数：在布尔掩码中寻找第一个连续 run 点的片段起点"""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    # 按照索引是否连续进行分割
    splits = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    if from_left:
        for seg in splits:
            if len(seg) >= run:
                return int(seg[0])
    else:
        for seg in splits[::-1]:
            if len(seg) >= run:
                return int(seg[-run])
    return None

def detect_band_edges_slope(
    V, logS_smooth, logS_raw, S_shifted, offset, 
    v_center_min=None, v_center_max=None,
    slope_threshold=0.5, min_gap=0.05,
    # --- 内部参数：保持与 STS_line.py 一致的默认值 ---
    run_len: int = 8,
    run_len_cb: int = 5,
    fit_window: float = 0.15,
    baseline_k: float = 0.0,
    amp_k: float = 2.0,
    r2_min: float = 0.90,
    # --- VB 曲率兜底参数 ---
    curv_k: float = 2.0,
    curv_percentile: float = 95,
    run_len_curv: int = 3,
    vb_q: float = 0.80,
    smooth_window: int = 11,
    raw_signal=None,
    **kwargs
):
    """
    改进的斜率判定法：完全同步自 STS_line.py 的实现。
    使用线性 dI/dV 空间的梯度触发 + 线性外推，并加入 VB 曲率兜底。
    """
    S_lin = S_shifted - offset

    # 使用 Savitzky-Golay 平滑线性数据
    win = int(smooth_window)
    if win % 2 == 0:
        win += 1
    if len(S_lin) > win:
        S_smooth = savgol_filter(S_lin, win, polyorder=2)
    else:
        S_smooth = S_lin.copy()

    # 计算梯度和曲率
    g = np.gradient(S_smooth, V)
    curv = np.gradient(g, V)

    # 确定平台区域以估计噪声
    if v_center_min is None or v_center_max is None:
        v_center_min, v_center_max = get_plateau_range(V, logS_raw)
    plateau_mask = (V >= float(v_center_min)) & (V <= float(v_center_max))
    if plateau_mask.sum() < 10:
        plateau_mask = np.abs(V) < 0.3 * np.max(np.abs(V))

    # 噪声估计：使用标准差和分位数
    sigma_g = float(np.std(g[plateau_mask], ddof=1)) if plateau_mask.sum() > 1 else float(np.std(g))
    noise_g = float(np.percentile(np.abs(g[plateau_mask]), 80)) if plateau_mask.any() else float(np.percentile(np.abs(g), 80))
    sigma_g = max(sigma_g, 1e-30)
    noise_g = max(noise_g, 1e-30)
    # 核心阈值 g_thr
    g_thr = float(max(slope_threshold * sigma_g, noise_g))

    # 曲率噪声估计
    sigma_curv = float(np.std(curv[plateau_mask], ddof=1)) if plateau_mask.sum() > 1 else float(np.std(curv))
    noise_curv = float(np.percentile(np.abs(curv[plateau_mask]), curv_percentile)) if plateau_mask.any() else float(np.percentile(np.abs(curv), curv_percentile))
    sigma_curv = max(sigma_curv, 1e-30)
    noise_curv = max(noise_curv, 1e-30)
    curv_thr = float(max(curv_k * sigma_curv, noise_curv))

    # 基准幅度门槛 (amp_thr) 以抑制软尾
    floor_lin = float(np.median(S_smooth[plateau_mask])) if plateau_mask.any() else float(np.median(S_smooth))
    sigma_lin = float(np.std(S_smooth[plateau_mask], ddof=1)) if plateau_mask.sum() > 1 else float(np.std(S_smooth))
    sigma_lin = max(sigma_lin, 1e-30)
    y_floor = float(floor_lin + baseline_k * sigma_lin)
    amp_thr = float(y_floor + amp_k * sigma_lin)

    # VB 触发：连续 run_len 点满足 g < -g_thr 且满足幅度门槛
    amp_mask = S_smooth > amp_thr
    vb_mask = (V < 0) & (g < -g_thr) & amp_mask
    idx_vb0 = _first_run(vb_mask, run_len, from_left=False)

    # CB 触发：连续 run_len_cb 点满足 g > g_thr (不加幅度门槛更宽松)
    cb_mask = (V > 0) & (g > g_thr)
    idx_cb0 = _first_run(cb_mask, run_len_cb, from_left=True)

    Ev = Ec = np.nan
    Ev_trigger = float(V[idx_vb0]) if idx_vb0 is not None else np.nan
    Ec_trigger = float(V[idx_cb0]) if idx_cb0 is not None else np.nan

    # --- VB 线性外推 ---
    Ev_candidate = Ec_candidate = np.nan
    Ev_curv = np.nan
    a_vb = b_vb = R2_vb = None
    a_cb = b_cb = R2_cb = None
    vb_fit_V = vb_fit_y = None
    cb_fit_V = cb_fit_y = None

    if idx_vb0 is not None:
        v_trig, g_trig = V[idx_vb0], g[idx_vb0]
        # 窗口向更负方向延伸 (交点后)
        near_mask = (V >= v_trig - fit_window) & (V <= v_trig) & (V < 0)
        # 宽松一点的过滤：只要斜率方向对即可
        valid_mask = near_mask & (g < 0)
        if valid_mask.sum() >= 3:
            a_vb = float(np.mean(g[valid_mask]))
            avg_V = float(np.mean(V[valid_mask]))
            avg_S = float(np.mean(S_smooth[valid_mask]))
            b_vb = float(avg_S - a_vb * avg_V)
        else:
            a_vb = float(g_trig)
            b_vb = float(S_smooth[idx_vb0] - a_vb * v_trig)
        R2_vb = 1.0
        if a_vb < 0:
            Ev_candidate = float((y_floor - b_vb) / a_vb)
            Ev = Ev_candidate
            # 用于绘图的拟合线
            v_start = min(Ev, v_trig - fit_window) - 0.05
            v_end = max(Ev, v_trig) + 0.05
            vb_fit_V = np.array([v_start, v_end])
            vb_fit_y = a_vb * vb_fit_V + b_vb

    # --- CB 线性外推 ---
    if idx_cb0 is not None:
        v_trig, g_trig = V[idx_cb0], g[idx_cb0]
        # 窗口向更正方向延伸 (交点后)
        near_mask = (V >= v_trig) & (V <= v_trig + fit_window) & (V > 0)
        valid_mask = near_mask & (g > 0)
        if valid_mask.sum() >= 3:
            a_cb = float(np.mean(g[valid_mask]))
            avg_V = float(np.mean(V[valid_mask]))
            avg_S = float(np.mean(S_smooth[valid_mask]))
            b_cb = float(avg_S - a_cb * avg_V)
        else:
            a_cb = float(g_trig)
            b_cb = float(S_smooth[idx_cb0] - a_cb * v_trig)
        R2_cb = 1.0
        if a_cb > 0:
            Ec_candidate = float((y_floor - b_cb) / a_cb)
            Ec = Ec_candidate
            # 用于绘图的拟合线
            v_start = min(Ec, v_trig) - 0.05
            v_end = max(Ec, v_trig + fit_window) + 0.05
            cb_fit_V = np.array([v_start, v_end])
            cb_fit_y = a_cb * cb_fit_V + b_cb

    # --- VB 曲率兜底（当 VB 外推没有给出 Ev 时才启用） ---
    if np.isnan(Ev):
        neg_mask = V < 0
        q_thr = float(np.percentile(S_smooth[neg_mask], vb_q * 100.0)) if np.any(neg_mask) else amp_thr
        amp_thr2 = float(max(amp_thr, q_thr))
        vb_curv_mask = (V < 0) & (np.abs(curv) > curv_thr) & (g < 0) & (S_smooth > amp_thr2)
        idx_vb_curv = _first_run(vb_curv_mask, run_len_curv, from_left=False)
        if idx_vb_curv is None:
            cand = np.where((V < 0) & (g < 0) & (S_smooth > amp_thr2))[0]
            if cand.size > 0:
                j = cand[np.argmax(np.abs(curv[cand]))]
                if np.abs(curv[j]) > curv_thr:
                    idx_vb_curv = int(j)
        if idx_vb_curv is not None:
            Ev_curv = float(V[idx_vb_curv])
            Ev = Ev_curv

    # 最终计算带隙
    Eg = 0.0
    note = "Gap not found"
    if not (np.isnan(Ev) or np.isnan(Ec)) and Ev < Ec:
        Eg = float(Ec - Ev)
        if Eg < min_gap:
            Eg, Ev, Ec = 0.0, np.nan, np.nan
            note = f"Eg<{min_gap}"
        else:
            note = "Success"
    else:
        reasons = []
        if np.isnan(Ec_trigger):
            reasons.append("CB trigger not found (too strict)")
        if np.isnan(Ev_trigger) and np.isnan(Ev_curv):
            reasons.append("VB trigger/curv not found")
        note = "; ".join(reasons) if reasons else note

    return Ev, Ec, Eg, {
        "Eg": Eg,
        "Ev": float(Ev) if not np.isnan(Ev) else np.nan,
        "Ec": float(Ec) if not np.isnan(Ec) else np.nan,
        "Ev_candidate": float(Ev_candidate) if not np.isnan(Ev_candidate) else np.nan,
        "Ec_candidate": float(Ec_candidate) if not np.isnan(Ec_candidate) else np.nan,
        "Ev_curv": float(Ev_curv) if not np.isnan(Ev_curv) else np.nan,
        "V": V,
        "I": S_smooth + offset,
        "logS": logS_smooth,
        "raw_I": raw_signal if raw_signal is not None else (S_smooth + offset),
        "logS_raw": logS_raw,
        "S_raw_lin": S_lin,
        "S_smooth_lin": S_smooth,
        "slopes": g,
        "curv": curv,
        "threshold": g_thr,
        "curv_thr": curv_thr,
        "sigma_g": sigma_g,
        "noise_g": noise_g,
        "sigma_curv": sigma_curv,
        "noise_curv": noise_curv,
        "floor_lin": floor_lin,
        "sigma_lin": sigma_lin,
        "y_floor": y_floor,
        "amp_thr": amp_thr,
        "Ev_trigger": Ev_trigger,
        "Ec_trigger": Ec_trigger,
        "a_vb": a_vb,
        "b_vb": b_vb,
        "R2_vb": R2_vb,
        "a_cb": a_cb,
        "b_cb": b_cb,
        "R2_cb": R2_cb,
        "vb_fit_V_lin": vb_fit_V,
        "vb_fit_y_lin": vb_fit_y,
        "cb_fit_V_lin": cb_fit_V,
        "cb_fit_y_lin": cb_fit_y,
        "v_center_min": float(v_center_min),
        "v_center_max": float(v_center_max),
        "method": "Slope Detection",
        "offset": offset,
        "note": note,
        "fit_space": "linear",
        "params": {
            "run_len": run_len,
            "run_len_cb": run_len_cb,
            "fit_window": fit_window,
            "baseline_k": baseline_k,
            "amp_k": amp_k,
            "r2_min": r2_min,
            "curv_k": curv_k,
            "curv_percentile": curv_percentile,
            "run_len_curv": run_len_curv,
            "vb_q": vb_q,
        },
    }

def detect_band_edges(
    V,
    signal,
    method="Adaptive Thresholding",
    mode="both",
    smooth_window=11,
    sigma_level=2.0,
    slope_percentile=85,
    min_gap=0.05,
    v_center_min=-0.3,
    v_center_max=0.3,
    delta_E=0.15,
    fit_window=0.05,
    slope_threshold=0.5,
    sigma_multiplier=0.0,
    sigma_v_min=-1.0,
    sigma_v_max=0.2,
    **kwargs
):
    """
    通用带边检测接口，支持三种新方法
    """
    V_proc, logS_smooth, logS_raw, S_shifted, offset = preprocess_signal(V, signal, smooth_window, sigma_multiplier=sigma_multiplier, sigma_v_min=sigma_v_min, sigma_v_max=sigma_v_max)
    
    if method == "Adaptive Thresholding" or method == "Dynamic Threshold":
        return detect_band_edges_dynamic(V_proc, logS_smooth, logS_raw, S_shifted, sigma_level, slope_percentile, min_gap, v_center_min, v_center_max, delta_E_fixed=delta_E, **kwargs)
    elif method == "Linear Extrapolation" or method == "Supplement (Linear Fit)":
        return extract_gap_log_plateau(V_proc, logS_smooth, logS_raw, S_shifted, v_center_min, v_center_max, delta_E, min_gap, sigma_level, **kwargs)
    elif method == "Dynamic Range Ratio" or method == "Peak-to-Threshold Drop" or method == "Slope Detection":
        # 如果是旧方法名称，兼容性处理
        s_thresh = kwargs.pop('slope_threshold', slope_threshold)
        return detect_band_edges_slope(
            V_proc, logS_smooth, logS_raw, S_shifted, offset, 
            v_center_min=v_center_min, v_center_max=v_center_max,
            slope_threshold=s_thresh, min_gap=min_gap,
            smooth_window=smooth_window, raw_signal=signal, **kwargs
        )
    else:
        # 默认回退到自适应阈值
        return detect_band_edges_dynamic(V_proc, logS_smooth, logS_raw, S_shifted, sigma_level, slope_percentile, min_gap, v_center_min, v_center_max)
