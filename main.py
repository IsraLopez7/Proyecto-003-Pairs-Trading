"""
Main – Universo 50 → 25 pares → EG → mejor par → Kalman + señales → Backtest → Métricas
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf

from config import *
from pair_selection import rank_pairs
from cointegration import CointegrationAnalysis
from kalman_filter import KalmanFilter
from backtesting import Backtester

def chronological_split(df: pd.DataFrame, train_ratio=0.6, test_ratio=0.2):
    n = len(df)
    i1 = int(n*train_ratio)
    i2 = int(n*(train_ratio+test_ratio))
    train = df.iloc[:i1].copy()
    test  = df.iloc[i1:i2].copy()
    val   = df.iloc[i2:].copy()
    return train, test, val

def thresholds_from_train(z_abs: np.ndarray):
    if USE_TRAINED_THRESHOLDS:
        e = np.clip(np.percentile(z_abs, TRAIN_QUANT_ENTRY*100),   1.6, 2.5)
        x = np.clip(np.percentile(z_abs, TRAIN_QUANT_EXIT*100),    0.25, 0.8)
        c = min(max(CONFIRM_MIN, e - TRAIN_QUANT_CONFIRM*0.25), e)
    else:
        e, x = Z_SCORE_ENTRY, Z_SCORE_EXIT
        c = max(CONFIRM_MIN, min(e - CONFIRM_GAP, e))
    return float(e), float(x), float(c)

def main():
    print("="*70)
    print("UNIVERSO 50 → 25 PARES → MEJOR PAR → KALMAN + BACKTEST")
    print("="*70)

    # 1) Ranking de pares
    print("\n1) Descargando universo y rankeando pares...")
    prices, ranking = rank_pairs()
    
    # MODIFICACIÓN: Verificar que el mejor par tenga buena cointegración
    valid_pairs = ranking[ranking['adf_resid_p'] < 0.10]  # Solo pares con p < 0.10
    
    if len(valid_pairs) == 0:
        print("⚠️ No hay pares bien cointegrados. Forzando KO-PEP...")
        # Forzar KO-PEP
        ko = yf.download('KO', start='2010-01-01', progress=False)['Close']
        pep = yf.download('PEP', start='2010-01-01', progress=False)['Close']
        prices = pd.DataFrame({'KO': ko, 'PEP': pep}).dropna()
        A, B = 'KO', 'PEP'
    else:
        best = valid_pairs.iloc[0]
        A, B = best["A"], best["B"]
        
    print(f"\nMejor par seleccionado: {A}-{B}")
    
    # Resto del código igual pero con umbrales más agresivos
    # Modificar el grid search:
    grids_E = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # Más agresivo
    grids_X = [0.05, 0.10, 0.15, 0.20]  # Salidas más rápidas

    px = prices[[A,B]].dropna()
    train_df, test_df, val_df = chronological_split(px, TRAIN_RATIO, TEST_RATIO)
    print("\nTamaños:", len(train_df), len(test_df), len(val_df))

    # 3) EG/Kalman en TRAIN
    co = CointegrationAnalysis()
    beta0, beta1, resid = co.run_ols_regression(train_df[A].values, train_df[B].values)
    coint, pval = co.test_stationarity(resid)
    hl = co.calculate_half_life(resid)

    print("\nDiagnóstico TRAIN:")
    print(f"  OLS: P_{A} = {beta0:.4f} + {beta1:.4f}·P_{B} + ε")
    print(f"  ADF resid p = {pval:.4f}  |  Half-life = {hl:.1f} días")

    kf = KalmanFilter()
    _, z_train, _, _ = kf.run_filter(train_df[A].values, train_df[B].values, residuals_train=resid)
    z_abs = np.abs(z_train[np.isfinite(z_train)])
    entry0, exit0, confirm0 = thresholds_from_train(z_abs)
    print(f"\nUmbrales base: entry={entry0:.2f}  exit={exit0:.2f}  confirm={confirm0:.2f}")

    # 4) Backtest en TEST (pequeño grid robusto)
    grids_E = [0.8, 1.0, 1.2, 1.4]  # Más opciones
    grids_X = [0.15, 0.25, 0.35, 0.45]  # Más opciones
    cfgs = []
    for E in grids_E:
        for X in grids_X:
            C = max(CONFIRM_MIN, min(E - CONFIRM_GAP, E))
            cfgs.append({"E":float(E),"X":float(X),"C":float(C),"S":STOP_LOSS_Z})

    A_t, B_t = test_df[A].values, test_df[B].values
    dates_t = test_df.index.values
    best_cfg, best_score = None, -1e9

    print("\n4) Grid en TEST:")
    for cfg in cfgs:
        bt = Backtester(test_df.copy())
        res = bt.run_backtest(
            A_t, B_t, dates_t,
            residuals_train=resid,
            half_life_days=hl,
            entry_thr=cfg["E"], exit_thr=cfg["X"], confirm_thr=cfg["C"],
            stop_loss_thr=cfg["S"],
            min_hold_days=MIN_HOLD_DAYS, cooldown_days=ENTRY_COOLDOWN_DAYS
        )
        st = res.get("daily_stats", {}) if res else {}
        ret   = float(st.get("total_return", 0.0) or 0.0)
        mdd   = float(st.get("max_drawdown", 1.0) or 1.0)
        sharpe= float(st.get("sharpe", -999) or -999)
        score = (ret*100) - (mdd*20) + (max(-10, sharpe)*5)
        print(f"  cfg {cfg} → Ret {ret*100:.2f}% | Sharpe {sharpe:.2f} | MDD {mdd*100:.2f}%")
        if score > best_score:
            best_score, best_cfg = score, cfg

    if best_cfg is None:
        best_cfg = {"E":entry0,"X":exit0,"C":confirm0,"S":STOP_LOSS_Z}

    print(f"\nMejor cfg TEST → E={best_cfg['E']:.2f}  X={best_cfg['X']:.2f}  C={best_cfg['C']:.2f}")

    # 5) Validación final
    print("\n5) Validación final...")
    A_v, B_v = val_df[A].values, val_df[B].values
    dates_v = val_df.index.values
    bt_v = Backtester(val_df.copy())
    res_v = bt_v.run_backtest(
        A_v, B_v, dates_v,
        residuals_train=resid,
        half_life_days=hl,
        entry_thr=best_cfg["E"], exit_thr=best_cfg["X"], confirm_thr=best_cfg["C"],
        stop_loss_thr=best_cfg["S"],
        min_hold_days=MIN_HOLD_DAYS, cooldown_days=ENTRY_COOLDOWN_DAYS
    )
    bt_v.print_statistics()

    # 6) GRÁFICOS
    print("\n6) Gráficos")
    bt_v.plot_results(save_plots=SAVE_PLOTS)
    bt_v.plot_signals_and_trades(save_plot=SAVE_PLOTS)  # NUEVO
    bt_v.plot_returns_distribution(save_plot=SAVE_PLOTS)

    print("\n" + "="*70)
    print(f"Resumen: mejor par {A}-{B} | EG p={pval:.4f} | HL={hl:.1f} días")
    print("="*70)

if __name__ == "__main__":
    main()
