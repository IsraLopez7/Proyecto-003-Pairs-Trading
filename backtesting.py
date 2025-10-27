# backtesting.py
# =============================================================================
# Backtester para estrategia de pairs trading con Kalman (alpha-beta dinámicos)
# - Mark-to-market diario
# - Comisiones y costo de préstamo diarios
# - Señales tipo pizarrón (z-score)
# - Gráficas y métricas consistentes con la equity mostrada
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter
from config import (
    # Capital y costos
    INITIAL_CAPITAL, POSITION_SIZE, COMMISSION_RATE,
    BORROW_RATE, BORROW_DAY_BASIS, MIN_TRADE_VALUE,

    # Señales y tiempos
    EXECUTION_LAG, STOP_LOSS_Z, MIN_HOLD_DAYS, ENTRY_COOLDOWN_DAYS,
    TIME_STOP_MULTIPLIER,

    # Output
    SAVE_PLOTS,
)

# -----------------------------------------------------------------------------


@dataclass
class Trade:
    side: int                   # +1: long A / short B | -1: short A / long B
    entry_i: int
    entry_A: float
    entry_B: float
    exit_i: Optional[int] = None
    exit_A: Optional[float] = None
    exit_B: Optional[float] = None


# -----------------------------------------------------------------------------


def metrics_from_equity(equity: pd.Series) -> Dict:
    """
    Métricas sobre NAV (equity normalizada a 1.0 en el inicio).
    De este modo MDD, Sharpe y el retorno total son siempre consistentes
    con la curva que se grafica/imprime.
    """
    eq = pd.Series(equity).dropna().astype(float)
    if len(eq) < 3:
        return {
            "total_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
        }

    nav = eq / eq.iloc[0]
    rets = nav.pct_change().fillna(0.0)

    ann = 252
    mu = rets.mean() * ann
    vol = rets.std() * np.sqrt(ann)
    neg = rets[rets < 0].std() * np.sqrt(ann)

    sharpe = float(mu / vol) if vol > 0 else 0.0
    sortino = float(mu / neg) if neg > 0 else 0.0

    peak = np.maximum.accumulate(nav.values)
    dd = (peak - nav.values) / peak
    mdd = float(dd.max()) if dd.size else 0.0

    total = float(nav.iloc[-1] - 1.0)
    calmar = float(total / mdd) if mdd > 0 else 0.0

    return {
        "total_return": total,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
    }


# -----------------------------------------------------------------------------


class Backtester:
    """
    Backtester de un par (A,B) con z-score de Kalman y reglas tipo pizarrón.
    Uso:
        bt = Backtester(df_val)  # df solo para index/fechas si quieres
        res = bt.run_backtest(A, B, dates, residuals_train, half_life_days,
                              entry_thr, exit_thr, confirm_thr, stop_loss_thr,
                              min_hold_days, cooldown_days, exec_lag)
        bt.print_statistics()
        bt.plot_results()
        bt.plot_returns_distribution()
        bt.plot_signals_and_trades()
    """

    def __init__(self, df: Optional[pd.DataFrame] = None, initial_capital: float = INITIAL_CAPITAL):
        self.df = df
        self.initial_capital = float(initial_capital)

        # Resultados por corrida
        self.equity: Optional[pd.Series] = None
        self.zscore: Optional[pd.Series] = None
        self.betas: Optional[pd.Series] = None
        self.alphas: Optional[pd.Series] = None
        self.trades: List[Trade] = []

        # Para las leyendas de plots
        self._last_params = {}

    # ------------------------------------------------------------------

    def _compute_kalman(self,
                        y_prices: np.ndarray,
                        x_prices: np.ndarray,
                        residuals_train: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Ejecuta KalmanFilter y devuelve (spreads, z, betas, alphas)
        """
        kf = KalmanFilter()
        spreads, z, betas, alphas = kf.run_filter(
            y_prices, x_prices, residuals_train=residuals_train
        )
        return spreads, z, betas, alphas

    # ------------------------------------------------------------------

    def run_backtest(
        self,
        A_prices: np.ndarray,
        B_prices: np.ndarray,
        dates: np.ndarray,

        residuals_train: Optional[np.ndarray] = None,
        half_life_days: Optional[float] = None,

        entry_thr: float = 1.6,
        exit_thr: float = 0.45,
        confirm_thr: float = 1.3,
        stop_loss_thr: float = STOP_LOSS_Z,

        min_hold_days: int = MIN_HOLD_DAYS,
        cooldown_days: int = ENTRY_COOLDOWN_DAYS,
        exec_lag: int = EXECUTION_LAG,
    ) -> Dict:
        """
        Simula el par sobre las series A/B dadas.
        Retorna dict con equity, z, trades y métricas.
        """
        y = np.asarray(A_prices, dtype=float)  # tratamos A como 'y'
        x = np.asarray(B_prices, dtype=float)  # y B como 'x'
        n = len(y)
        assert len(x) == n == len(dates), "Longitudes inconsistentes en backtest."

        # 1) Kalman
        spreads, z, betas, alphas = self._compute_kalman(y, x, residuals_train)
        idx = pd.to_datetime(dates)
        z_series = pd.Series(z, index=idx)
        self.spread: Optional[pd.Series] = None

        # 2) Timers/umbrales
        # time-stop basado en half-life
        if (half_life_days is None) or (not np.isfinite(half_life_days)):
            half_life_days = 60.0
        tstop = int(max(10, TIME_STOP_MULTIPLIER * float(half_life_days)))

        # 3) Estado de la simulación
        cash = float(self.initial_capital)
        pos = 0                 # +1 longA/shortB, -1 shortA/longB
        k_units = 0
        entry_A = None
        entry_B = None
        holding = 0
        cooldown = 0

        equity: List[float] = []
        trades: List[Trade] = []

        def mark_to_market(pa: float, pb: float) -> float:
            """Valoriza PnL no realizado de la posición abierta."""
            if pos == 0 or k_units == 0 or entry_A is None or entry_B is None:
                return cash
            pnl = k_units * ((pa - entry_A) - pos * (pb - entry_B))
            return cash + pnl

        # 4) Loop diario
        for i in range(n):
            pa = float(y[i])
            pb = float(x[i])

            # señal defasada (exec delay)
            t_sig = i - exec_lag
            zt = float(z[t_sig]) if t_sig >= 0 and np.isfinite(z[t_sig]) else np.nan

            # Decrementa cooldown
            if cooldown > 0:
                cooldown -= 1

            # REGLA: cerrar
            need_close = False
            if pos != 0:
                holding += 1
                # salida, stop o time-stop
                if (abs(zt) <= exit_thr) or (abs(zt) >= stop_loss_thr) or (holding >= tstop):
                    need_close = True

            if need_close and pos != 0:
                # realiza PnL
                cash = mark_to_market(pa, pb)

                # comisiones de salida
                cash -= COMMISSION_RATE * (k_units * pa + k_units * pb)
                # último día de borrow del lado corto
                borrow_notional = k_units * (pb if pos == +1 else pa)
                cash -= borrow_notional * (BORROW_RATE / BORROW_DAY_BASIS)

                # cierra trade
                trades[-1].exit_i = i
                trades[-1].exit_A = pa
                trades[-1].exit_B = pb

                # reset
                pos = 0
                k_units = 0
                entry_A = None
                entry_B = None
                holding = 0
                cooldown = cooldown_days

                equity.append(cash)
                continue

            # REGLA: abrir
            opened = False
            if pos == 0 and cooldown == 0 and np.isfinite(zt):
                open_long = (zt <= -entry_thr) and (abs(zt) >= confirm_thr)
                open_short = (zt >= entry_thr) and (abs(zt) >= confirm_thr)

                if open_long or open_short:
                    side = +1 if open_long else -1

                    # tamaño de la posición (notional 80% del cash)
                    notional = cash * POSITION_SIZE
                    k = int(np.floor(notional / (pa + pb)))
                    if k > 0 and (k * (pa + pb)) >= MIN_TRADE_VALUE:
                        # comisiones de entrada
                        comm = COMMISSION_RATE * (k * pa + k * pb)
                        cash -= comm
                        # primer día de borrow del lado corto
                        borrow_notional = k * (pb if side == +1 else pa)
                        cash -= borrow_notional * (BORROW_RATE / BORROW_DAY_BASIS)

                        trades.append(Trade(side=side, entry_i=i, entry_A=pa, entry_B=pb))
                        pos = side
                        k_units = k
                        entry_A = pa
                        entry_B = pb
                        holding = 0
                        opened = True

            # COSTO DE PRÉSTAMO DIARIO si hay posición y no se abrió hoy
            if pos != 0 and not opened:
                borrow_notional = k_units * (pb if pos == +1 else pa)
                cash -= borrow_notional * (BORROW_RATE / BORROW_DAY_BASIS)
                # equity mark-to-market
                equity.append(mark_to_market(pa, pb))
            else:
                # sin posición (o justo abrió): equity = cash
                equity.append(cash)

        # 5) Armar resultados
        equity = pd.Series(equity, index=idx)
        stats = metrics_from_equity(equity)

        # guardar en el objeto
        self.equity = equity
        self.zscore = z_series
        self.betas = pd.Series(betas, index=idx)
        self.alphas = pd.Series(alphas, index=idx)
        self.trades = trades
        self._last_params = dict(E=entry_thr, X=exit_thr, C=confirm_thr, S=stop_loss_thr)
        self.spread = pd.Series(spreads, index=idx)


        return {
            "equity": equity,
            "z": z_series,
            "betas": self.betas,
            "alphas": self.alphas,
            "trades": trades,
            "daily_stats": stats,
            "cfg_used": self._last_params.copy(),
        }

    # ------------------------------------------------------------------

    def print_statistics(self):
        if self.equity is None:
            print("No hay resultados cargados.")
            return
        st = metrics_from_equity(self.equity)
        n_trades = len(self.trades)
        closed = sum(1 for t in self.trades if t.exit_i is not None)
        avg_days = 0.0
        if closed > 0:
            lengths = [t.exit_i - t.entry_i for t in self.trades if t.exit_i is not None]
            if lengths:
                avg_days = float(np.mean(lengths))

        print("\n" + "=" * 50)
        print("ESTADÍSTICAS DIARIAS (equity)")
        print("=" * 50)
        print(f"Retorno total: {st['total_return']*100:6.2f}%")
        print(f"Sharpe (anualizado): {st['sharpe']:+.2f}")
        print(f"Sortino (anualizado): {st['sortino']:+.2f}")
        print(f"Calmar: {st['calmar']:+.2f}")
        print(f"Máx Drawdown: {st['max_drawdown']*100:6.2f}%")
        print("-" * 50)
        print(f"Trades ejecutados: {n_trades}")
        print(f"Trades cerrados:   {closed}")
        print(f"Días promedio por trade: {avg_days:.1f}")
        print("=" * 50)

    # ------------------------------------------------------------------

    def plot_results(self, save_plots: bool = SAVE_PLOTS, fname: str = "pairs_trading_results.png"):
        if any(x is None for x in (self.zscore, self.betas, self.equity)):
            print("No hay resultados para graficar.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        # Z-score
        axes[0].plot(self.zscore.index, self.zscore.values, label="Z-score")
        E = self._last_params.get("E", 1.6)
        X = self._last_params.get("X", 0.45)
        axes[0].axhline(+E, color="red", ls="--", alpha=0.7)
        axes[0].axhline(-E, color="red", ls="--", alpha=0.7, label=f"Entry ±{E:.2f}")
        axes[0].axhline(+X, color="green", ls="--", alpha=0.5)
        axes[0].axhline(-X, color="green", ls="--", alpha=0.5, label=f"Exit ±{X:.2f}")
        axes[0].set_title("Z-score (Kalman)")
        axes[0].legend(loc="upper right")

        # β dinámico
        axes[1].plot(self.betas.index, self.betas.values, label="β (hedge ratio)")
        axes[1].legend(loc="upper right")
        axes[1].set_title("β dinámico")

        # Equity
        axes[2].plot(self.equity.index, self.equity.values, label="Equity", color="tab:blue")
        base = float(self.equity.iloc[0])
        axes[2].axhline(base, color="red", ls="--", alpha=0.5, label="Capital inicial")
        axes[2].legend(loc="upper right")
        axes[2].set_title("Equity curve")
        axes[2].set_xlabel("Tiempo")

        plt.tight_layout()
        if save_plots:
            plt.savefig(fname, dpi=140)
        plt.show()

    # ------------------------------------------------------------------

    def plot_returns_distribution(self, save_plot: bool = SAVE_PLOTS, fname: str = "returns_distribution.png"):
        if self.equity is None:
            print("No hay equity para graficar distribución.")
            return

        nav = self.equity / self.equity.iloc[0]
        rets = nav.pct_change().dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.hist(rets * 100.0, bins=50)
        ax1.set_title("Distribución de retornos diarios (%)")
        mean_pct = rets.mean() * 100.0
        ax1.axvline(mean_pct, color="green", lw=2, label=f"Media: {mean_pct:.2f}%")
        ax1.legend()

        ax2.boxplot(rets * 100.0, vert=True)
        ax2.set_title("Box plot de retornos diarios (%)")

        plt.tight_layout()
        if save_plot:
            plt.savefig(fname, dpi=140)
        plt.show()

    def plot_spread_evolution(self, window: int = 60, bands: float = 2.0,
                            save_plot: bool = SAVE_PLOTS,
                            fname: str = "spread_evolution.png"):
        if self.spread is None:
            print("No hay spread para graficar. Ejecuta el backtest primero.")
            return

        s = self.spread.astype(float).dropna()
        mu = s.rolling(window).mean()
        sd = s.rolling(window).std()

        plt.figure(figsize=(12, 5))
        plt.plot(s.index, s.values, label="Spread", linewidth=1)
        plt.plot(mu.index, mu.values, linestyle="--", label=f"Media ({window})")
        plt.plot(mu.index, (mu + bands*sd).values, linestyle="--", alpha=0.7)
        plt.plot(mu.index, (mu - bands*sd).values, linestyle="--", alpha=0.7)
        plt.title("Evolución del Spread")
        plt.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig(fname, dpi=140)
        plt.show()

    def plot_trade_pnl_distribution(self, save_plot: bool = SAVE_PLOTS,
                                    fname: str = "trade_pnl_distribution.png"):
        if not self.trades:
            print("No hay operaciones para graficar.")
            return

        # Retorno porcentual por trade sobre el notional de entrada (entry_A + entry_B)
        pnl_pct = []
        for t in self.trades:
            if t.exit_i is None:
                continue
            # PnL por unidad: (ΔA) - side*(ΔB)
            pnl_per_unit = (t.exit_A - t.entry_A) - t.side * (t.exit_B - t.entry_B)
            denom = max(1e-12, (t.entry_A + t.entry_B))  # evitar división por 0
            pnl_pct.append(100.0 * pnl_per_unit / denom)

        if len(pnl_pct) == 0:
            print("No hay trades cerrados para graficar.")
            return

        pnl_pct = np.array(pnl_pct, dtype=float)

        plt.figure(figsize=(10, 5))
        plt.hist(pnl_pct, bins=30)
        plt.axvline(pnl_pct.mean(), linewidth=2, label=f"Media: {pnl_pct.mean():.2f}%")
        plt.title("Distribución de retornos por trade (%)")
        plt.legend()
        plt.tight_layout()
        if save_plot:
            plt.savefig(fname, dpi=140)
        plt.show()

    # ------------------------------------------------------------------

    def plot_signals_and_trades(self, save_plot: bool = SAVE_PLOTS, fname: str = "trading_signals.png"):
        if any(x is None for x in (self.zscore, self.equity)):
            print("No hay resultados para graficar señales.")
            return

        E = self._last_params.get("E", 1.6)
        X = self._last_params.get("X", 0.45)

        # Reconstruimos la serie de posición a partir de trades
        pos_series = pd.Series(0, index=self.zscore.index)
        for t in self.trades:
            start = self.zscore.index[t.entry_i]
            end = self.zscore.index[t.exit_i] if t.exit_i is not None else self.zscore.index[-1]
            pos_series.loc[start:end] = t.side

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Z-score + señales
        ax1.plot(self.zscore.index, self.zscore.values, color="tab:blue", lw=0.9, label="Z-score")
        ax1.axhline(+E, color="red", ls="--", alpha=0.7)
        ax1.axhline(-E, color="red", ls="--", alpha=0.7, label=f"Entry ±{E:.1f}")
        ax1.axhline(+X, color="green", ls="--", alpha=0.5)
        ax1.axhline(-X, color="green", ls="--", alpha=0.5, label=f"Exit ±{X:.1f}")

        for t in self.trades:
            ax1.scatter(self.zscore.index[t.entry_i], self.zscore.iloc[t.entry_i],
                        marker="^" if t.side == +1 else "v", color="green" if t.side == +1 else "red", zorder=5)
            if t.exit_i is not None:
                ax1.scatter(self.zscore.index[t.exit_i], self.zscore.iloc[t.exit_i],
                            marker="x", color="black", zorder=5)
        ax1.legend(loc="upper right")
        ax1.set_title("Z-score con Señales de Trading")

        # Posición
        ax2.plot(pos_series.index, pos_series.values, color="k")
        ax2.set_title("Posiciones: Long (+1) / Flat (0) / Short (-1)")
        ax2.set_ylim(-1.5, 1.5)

        # Equity
        ax3.plot(self.equity.index, self.equity.values, color="tab:blue", label="Portfolio Value")
        base = float(self.equity.iloc[0])
        ax3.axhline(base, color="red", ls="--", alpha=0.5, label="Capital Inicial")
        ax3.legend(loc="upper right")
        ax3.set_title("Equity Curve")
        ax3.set_xlabel("Tiempo (días)")

        plt.tight_layout()
        if save_plot:
            plt.savefig(fname, dpi=140)
        plt.show()
