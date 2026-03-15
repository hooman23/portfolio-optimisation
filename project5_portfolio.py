"""
================================================================================
PROJECT 5 — PORTFOLIO OPTIMISATION
================================================================================
  1. Markowitz Mean-Variance Optimisation
  2. Efficient Frontier
  3. Maximum Sharpe Ratio Portfolio
  4. Minimum Variance Portfolio
  5. Black-Litterman Model (investor views)
  6. Monte Carlo random portfolios
  7. Capital Market Line
  8. Risk contribution analysis
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

DARK   = "#0f172a"; CARD   = "#1e293b"; ACCENT = "#3b82f6"
GREEN  = "#10b981"; AMBER  = "#f59e0b"; RED    = "#ef4444"
LIGHT  = "#e2e8f0"; MUTED  = "#64748b"; GRID   = "#334155"
PURPLE = "#8b5cf6"; TEAL   = "#14b8a6"; PINK   = "#ec4899"

ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "XOM", "JNJ", "BRK", "SPY"]
N      = len(ASSETS)
RF     = 0.04   # risk free rate 4%


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SIMULATE ASSET RETURNS
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_returns(n_years=10, seed=42):
    """
    Simulate correlated daily returns for 10 assets.
    Realistic correlations: tech stocks correlated, sectors less so.
    """
    rng = np.random.default_rng(seed)
    n_days = n_years * 252

    # Annual expected returns (mu) and vols (sigma)
    mu_annual    = np.array([0.15, 0.14, 0.13, 0.16, 0.10,
                              0.11, 0.08, 0.09, 0.11, 0.10])
    sigma_annual = np.array([0.28, 0.26, 0.25, 0.30, 0.22,
                              0.24, 0.20, 0.18, 0.19, 0.17])

    # Correlation matrix (realistic sector correlations)
    corr = np.array([
        [1.00, 0.85, 0.80, 0.75, 0.35, 0.38, 0.20, 0.25, 0.40, 0.70],
        [0.85, 1.00, 0.82, 0.72, 0.38, 0.40, 0.22, 0.27, 0.42, 0.72],
        [0.80, 0.82, 1.00, 0.70, 0.32, 0.35, 0.18, 0.22, 0.38, 0.68],
        [0.75, 0.72, 0.70, 1.00, 0.30, 0.32, 0.20, 0.20, 0.35, 0.65],
        [0.35, 0.38, 0.32, 0.30, 1.00, 0.80, 0.40, 0.35, 0.55, 0.55],
        [0.38, 0.40, 0.35, 0.32, 0.80, 1.00, 0.38, 0.32, 0.52, 0.58],
        [0.20, 0.22, 0.18, 0.20, 0.40, 0.38, 1.00, 0.30, 0.35, 0.45],
        [0.25, 0.27, 0.22, 0.20, 0.35, 0.32, 0.30, 1.00, 0.40, 0.42],
        [0.40, 0.42, 0.38, 0.35, 0.55, 0.52, 0.35, 0.40, 1.00, 0.60],
        [0.70, 0.72, 0.68, 0.65, 0.55, 0.58, 0.45, 0.42, 0.60, 1.00],
    ])

    # Cholesky decomposition for correlated returns
    sigma_daily = sigma_annual / np.sqrt(252)
    mu_daily    = mu_annual    / 252
    cov_daily   = np.diag(sigma_daily) @ corr @ np.diag(sigma_daily)
    L           = np.linalg.cholesky(cov_daily)

    Z    = rng.standard_normal((n_days, N))
    rets = mu_daily + Z @ L.T

    dates = pd.bdate_range("2014-01-01", periods=n_days)
    return pd.DataFrame(rets, index=dates, columns=ASSETS)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PORTFOLIO STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def portfolio_stats(weights, mu, cov):
    """Annual return, volatility, Sharpe for a given weight vector."""
    ret   = weights @ mu * 252
    vol   = np.sqrt(weights @ cov @ weights * 252)
    sharpe = (ret - RF) / vol
    return ret, vol, sharpe


def portfolio_variance(weights, cov):
    return weights @ cov @ weights * 252


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EFFICIENT FRONTIER
# ═══════════════════════════════════════════════════════════════════════════════

def efficient_frontier(mu, cov, n_points=80):
    """Compute efficient frontier by minimising vol for target returns."""
    n = len(mu)
    target_rets = np.linspace(mu.min() * 252 * 0.8,
                               mu.max() * 252 * 1.1, n_points)
    ef_vols = []
    ef_weights = []

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]
    bounds = [(0, 1)] * n

    for target in target_rets:
        constraints_t = constraints + [
            {"type": "eq", "fun": lambda w, t=target: portfolio_stats(w, mu, cov)[0] - t}
        ]
        w0  = np.ones(n) / n
        res = minimize(portfolio_variance, w0, args=(cov,),
                       method="SLSQP", bounds=bounds,
                       constraints=constraints_t,
                       options={"maxiter": 1000, "ftol": 1e-9})
        if res.success:
            ef_vols.append(np.sqrt(res.fun))
            ef_weights.append(res.x)

    return np.array(target_rets[:len(ef_vols)]), np.array(ef_vols), ef_weights


def max_sharpe_portfolio(mu, cov):
    """Find portfolio with maximum Sharpe ratio."""
    n = len(mu)
    def neg_sharpe(w):
        r, v, s = portfolio_stats(w, mu, cov)
        return -s

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n
    w0          = np.ones(n) / n

    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"maxiter": 1000})
    return res.x


def min_variance_portfolio(mu, cov):
    """Find global minimum variance portfolio."""
    n  = len(mu)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = [(0, 1)] * n
    w0          = np.ones(n) / n

    res = minimize(portfolio_variance, w0, args=(cov,),
                   method="SLSQP", bounds=bounds,
                   constraints=constraints)
    return res.x


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MONTE CARLO RANDOM PORTFOLIOS
# ═══════════════════════════════════════════════════════════════════════════════

def monte_carlo_portfolios(mu, cov, n_portfolios=3000):
    """Simulate random portfolios to visualise the feasible set."""
    rng  = np.random.default_rng(42)
    rets, vols, sharpes = [], [], []
    weights_list = []

    for _ in range(n_portfolios):
        w   = rng.dirichlet(np.ones(N))
        r, v, s = portfolio_stats(w, mu, cov)
        rets.append(r); vols.append(v); sharpes.append(s)
        weights_list.append(w)

    return np.array(rets), np.array(vols), np.array(sharpes), weights_list


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BLACK-LITTERMAN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def black_litterman(mu_eq, cov, views_P, views_Q, omega=None, tau=0.05):
    """
    Black-Litterman combines market equilibrium returns with investor views.

    mu_eq:   equilibrium (market implied) returns  N x 1
    cov:     covariance matrix                      N x N
    views_P: pick matrix (K x N) — which assets each view is about
    views_Q: view returns (K x 1) — what you think those returns will be
    omega:   uncertainty in views (K x K diagonal) — defaults to proportional
    tau:     scalar uncertainty in prior
    """
    K = views_P.shape[0]
    if omega is None:
        omega = np.diag(np.diag(tau * views_P @ cov @ views_P.T))

    tau_cov   = tau * cov
    M_inv     = np.linalg.inv(
        np.linalg.inv(tau_cov) + views_P.T @ np.linalg.inv(omega) @ views_P
    )
    mu_bl     = M_inv @ (
        np.linalg.inv(tau_cov) @ mu_eq + views_P.T @ np.linalg.inv(omega) @ views_Q
    )
    cov_bl    = cov + M_inv
    return mu_bl, cov_bl


# ═══════════════════════════════════════════════════════════════════════════════
# 6. RISK CONTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def risk_contribution(weights, cov):
    """How much each asset contributes to total portfolio risk."""
    port_vol = np.sqrt(weights @ cov @ weights)
    marginal  = cov @ weights / port_vol
    contrib   = weights * marginal
    return contrib / contrib.sum()   # as % of total risk


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BACKTEST PORTFOLIO vs EQUAL WEIGHT vs SPY
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_portfolio(returns, weights, label="Portfolio"):
    """Simple buy-and-hold portfolio backtest."""
    port_ret = returns @ weights
    equity   = (1 + port_ret).cumprod() * 100_000
    dd       = (equity - equity.cummax()) / equity.cummax()
    years    = len(equity) / 252
    cagr     = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1
    sharpe   = port_ret.mean() / port_ret.std() * np.sqrt(252)
    return {
        "label": label, "equity": equity, "drawdown": dd,
        "metrics": {
            "Total Return (%)": round((equity.iloc[-1]/100_000 - 1)*100, 2),
            "CAGR (%)":         round(cagr*100, 2),
            "Sharpe Ratio":     round(sharpe, 3),
            "Max Drawdown (%)": round(dd.min()*100, 2),
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TEARSHEET
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tearsheet(save_path="tearsheet_portfolio.png"):
    print("  [1/5] Simulating returns...")
    returns = simulate_returns()
    mu      = returns.mean().values
    cov     = returns.cov().values

    print("  [2/5] Computing efficient frontier...")
    ef_rets, ef_vols, ef_w = efficient_frontier(mu, cov)
    w_sharpe = max_sharpe_portfolio(mu, cov)
    w_minvar = min_variance_portfolio(mu, cov)
    mc_rets, mc_vols, mc_sharpes, mc_weights = monte_carlo_portfolios(mu, cov)

    print("  [3/5] Black-Litterman...")
    mu_eq    = mu * 252
    views_P  = np.zeros((2, N))
    views_P[0, 0] = 1; views_P[0, 1] = -1   # View: AAPL outperforms MSFT by 3%
    views_P[1, 3] = 1                         # View: AMZN returns 18%
    views_Q  = np.array([0.03, 0.18])
    mu_bl, cov_bl = black_litterman(mu_eq, cov, views_P, views_Q)
    w_bl = max_sharpe_portfolio(mu_bl / 252, cov_bl)

    print("  [4/5] Backtesting...")
    eq_weights = np.ones(N) / N
    bt_eq     = backtest_portfolio(returns, eq_weights,    "Equal Weight")
    bt_sharpe = backtest_portfolio(returns, w_sharpe,      "Max Sharpe")
    bt_minvar = backtest_portfolio(returns, w_minvar,      "Min Variance")
    bt_bl     = backtest_portfolio(returns, w_bl,          "Black-Litterman")

    print("  [5/5] Plotting...")
    fig = plt.figure(figsize=(22, 28), facecolor=DARK)
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            hspace=0.52, wspace=0.35,
                            top=0.93, bottom=0.04,
                            left=0.07, right=0.97)

    def sa(ax):
        ax.set_facecolor(CARD); ax.tick_params(colors=MUTED, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(True, color=GRID, lw=0.4, ls="--", alpha=0.5)

    tk = dict(color=LIGHT, fontweight="bold", fontsize=10)

    fig.text(0.5, 0.965, "PROJECT 5 — PORTFOLIO OPTIMISATION",
             ha="center", fontsize=18, fontweight="bold",
             color="#f8fafc", fontfamily="monospace")
    fig.text(0.5, 0.950,
             "Markowitz Efficient Frontier  |  Max Sharpe  |  Min Variance  |  Black-Litterman  |  Risk Attribution",
             ha="center", fontsize=10, color=MUTED, fontfamily="monospace")

    # ── 1. Efficient Frontier + Monte Carlo ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    sa(ax1); ax1.set_title("Efficient Frontier & Random Portfolios", **tk)

    sc = ax1.scatter(mc_vols*100, mc_rets*100, c=mc_sharpes,
                     cmap="RdYlGn", alpha=0.4, s=8, zorder=2)
    plt.colorbar(sc, ax=ax1, label="Sharpe Ratio")

    ax1.plot(ef_vols*100, ef_rets*100, color=ACCENT, lw=3,
             label="Efficient Frontier", zorder=5)

    r_s, v_s, sr_s = portfolio_stats(w_sharpe, mu, cov)
    r_m, v_m, sr_m = portfolio_stats(w_minvar, mu, cov)
    r_b, v_b, sr_b = portfolio_stats(w_bl, mu_bl/252, cov_bl)

    ax1.scatter(v_s*100, r_s*100, color=GREEN,  s=200, zorder=6,
                marker="*", label=f"Max Sharpe (SR={sr_s:.2f})")
    ax1.scatter(v_m*100, r_m*100, color=AMBER,  s=200, zorder=6,
                marker="D", label=f"Min Variance")
    ax1.scatter(v_b*100, r_b*100, color=PURPLE, s=200, zorder=6,
                marker="^", label=f"Black-Litterman")

    # Capital Market Line
    vols_cml = np.linspace(0, ef_vols.max()*1.2*100, 100)
    cml      = RF*100 + (r_s - RF) / v_s * vols_cml/100 * 100
    ax1.plot(vols_cml, cml, color=RED, lw=1.5, ls="--",
             alpha=0.8, label="Capital Market Line")

    ax1.set_xlabel("Annual Volatility (%)", color=MUTED)
    ax1.set_ylabel("Annual Return (%)", color=MUTED)
    ax1.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 2. Portfolio weights comparison ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    sa(ax2); ax2.set_title("Portfolio Weights Comparison", **tk)
    x   = np.arange(N)
    w   = 0.25
    ax2.bar(x - w, w_sharpe * 100, width=w, color=GREEN,  alpha=0.85, label="Max Sharpe")
    ax2.bar(x,     w_minvar * 100, width=w, color=AMBER,  alpha=0.85, label="Min Variance")
    ax2.bar(x + w, w_bl    * 100, width=w, color=PURPLE, alpha=0.85, label="Black-Litterman")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ASSETS, fontsize=8, color=MUTED)
    ax2.set_ylabel("Weight (%)", color=MUTED)
    ax2.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 3. Risk contribution ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    sa(ax3); ax3.set_title("Risk Contribution — Max Sharpe Portfolio", **tk)
    rc  = risk_contribution(w_sharpe, cov) * 100
    colors_rc = [GREEN if r < 15 else AMBER if r < 25 else RED for r in rc]
    bars = ax3.bar(ASSETS, rc, color=colors_rc, alpha=0.85, zorder=3)
    ax3.axhline(100/N, color=MUTED, lw=1.5, ls="--",
                label=f"Equal contrib ({100/N:.1f}%)")
    ax3.set_ylabel("Risk Contribution (%)", color=MUTED)
    ax3.set_xticklabels(ASSETS, fontsize=8, color=MUTED)
    for bar, v in zip(bars, rc):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", fontsize=7, color=LIGHT)
    ax3.legend(fontsize=8, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 4. Equity curves ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    sa(ax4); ax4.set_title("Portfolio Backtest — Equity Curves", **tk)
    for bt, col in [(bt_eq, MUTED), (bt_minvar, AMBER),
                    (bt_sharpe, GREEN), (bt_bl, PURPLE)]:
        ax4.plot(bt["equity"].index, bt["equity"],
                 color=col, lw=2, label=bt["label"])
    ax4.set_ylabel("Portfolio Value ($)", color=MUTED)
    ax4.legend(fontsize=9, framealpha=0.2, facecolor=CARD,
               edgecolor=GRID, labelcolor=LIGHT)

    # ── 5. Drawdown ───────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    sa(ax5); ax5.set_title("Drawdown", **tk)
    for bt, col in [(bt_eq, MUTED), (bt_minvar, AMBER),
                    (bt_sharpe, GREEN), (bt_bl, PURPLE)]:
        ax5.fill_between(bt["drawdown"].index, bt["drawdown"]*100, 0,
                         color=col, alpha=0.3)
        ax5.plot(bt["drawdown"].index, bt["drawdown"]*100, color=col, lw=1)
    ax5.set_ylabel("Drawdown (%)", color=MUTED)

    # ── 6. Correlation heatmap ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.set_facecolor(CARD)
    ax6.set_title("Asset Correlation Matrix", **tk)
    corr_mat = pd.DataFrame(cov, columns=ASSETS, index=ASSETS)
    corr_mat = corr_mat / np.outer(np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov)))
    im = ax6.imshow(corr_mat.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax6.set_xticks(range(N)); ax6.set_yticks(range(N))
    ax6.set_xticklabels(ASSETS, fontsize=7, color=MUTED, rotation=45)
    ax6.set_yticklabels(ASSETS, fontsize=7, color=MUTED)
    for i in range(N):
        for j in range(N):
            ax6.text(j, i, f"{corr_mat.values[i,j]:.2f}",
                     ha="center", va="center", fontsize=6, color="black")
    plt.colorbar(im, ax=ax6)

    # ── 7. Metrics table ──────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[4, :])
    ax7.set_facecolor(CARD); ax7.axis("off")
    ax7.set_title("Performance Metrics Summary", color=LIGHT,
                  fontweight="bold", fontsize=10)

    rows = []
    for bt in [bt_eq, bt_minvar, bt_sharpe, bt_bl]:
        m = bt["metrics"]
        r_a, v_a, sr_a = portfolio_stats(
            w_sharpe if bt["label"] == "Max Sharpe" else
            w_minvar if bt["label"] == "Min Variance" else
            w_bl     if bt["label"] == "Black-Litterman" else eq_weights,
            mu, cov
        )
        rows.append([
            bt["label"],
            f"{m['Total Return (%)']:.1f}%",
            f"{m['CAGR (%)']:.1f}%",
            f"{m['Sharpe Ratio']:.2f}",
            f"{m['Max Drawdown (%)']:.1f}%",
            f"{v_a*100:.1f}%",
            f"{r_a*100:.1f}%",
        ])

    cols = ["Strategy", "Total Return", "CAGR", "Sharpe", "Max DD",
            "Ann. Vol", "Ann. Return"]
    tbl  = ax7.table(cellText=rows, colLabels=cols,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
    for (r2, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r2 == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="#93c5fd", fontweight="bold")
        else:
            bg = "#14532d" if rows[r2-1][0] == "Max Sharpe" else \
                 "#1e293b" if r2%2==0 else "#172032"
            cell.set_facecolor(bg)
            cell.set_text_props(color="#4ade80" if bg == "#14532d" else LIGHT)

    plt.savefig(save_path, dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [✓] Portfolio tearsheet → {save_path}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  PROJECT 5: PORTFOLIO OPTIMISATION")
    print("="*60)
    returns  = simulate_returns()
    mu       = returns.mean().values
    cov      = returns.cov().values
    w_sharpe = max_sharpe_portfolio(mu, cov)
    w_minvar = min_variance_portfolio(mu, cov)
    r_s, v_s, sr_s = portfolio_stats(w_sharpe, mu, cov)
    r_m, v_m, sr_m = portfolio_stats(w_minvar, mu, cov)
    print(f"\n  Max Sharpe Portfolio:")
    print(f"    Return: {r_s*100:.1f}%  Vol: {v_s*100:.1f}%  Sharpe: {sr_s:.2f}")
    print(f"  Min Variance Portfolio:")
    print(f"    Return: {r_m*100:.1f}%  Vol: {v_m*100:.1f}%  Sharpe: {sr_m:.2f}")
    print(f"\n  Top 3 weights (Max Sharpe):")
    top3 = np.argsort(w_sharpe)[::-1][:3]
    for i in top3:
        print(f"    {ASSETS[i]}: {w_sharpe[i]*100:.1f}%")
    print("\n  Building tearsheet...")
    plot_tearsheet()
