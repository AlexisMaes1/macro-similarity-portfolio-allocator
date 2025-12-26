import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ============================================================
# === PARAMÈTRES GLOBAUX
# ============================================================

MACRO_CSV = "macro_indicators_v2.csv"
MOM_DAILY_CSV = "sp500_daily_moml.csv"   
PRICES_CSV = "asset_prices_daily_11ETF.csv"

BACKTEST_START = "2015-01-01"
CUTOFF_MARKET = "2006-01-01"

# mapping classes d'actifs -> colonnes du CSV de prix
ASSETS = [
    "Equity_SP500",
    "GovBonds_10y+",
    "Gold",
    "Commodities",
    "REITs_US",
    "Cash_TBills",
    "Equity_EM_VWO",
    "Equity_Japan_EWJ",
    "Equity_Eurozone_EZU",
    "Equity_EM_EEM",
    "QQQ",
]

TARGET_VOL_ANN = 0.15
LONG_ONLY = True
MAX_WEIGHT = 0.25
LAMBDA_SHRINK = 0
K_TOP = 12

FEATURES_LEVEL = [
    "GDP_YoY",
    "CPI_YoY",
    "FedFunds",
    "UST10Y",
    "Spread_10Y_2Y",
    "Unemployment",
    "VIX",
    "INDPRO_YoY",
    #"RetailSales_YoY",
    "HousingStarts_YoY",
    "BAA10Y_Spread",
    "UMich_Sentiment",
    "SP500_MOM_3M",
]

MOM_BASE = [
    "GDP_YoY",
    "CPI_YoY",
    "FedFunds",
    "UST10Y",
    "Spread_10Y_2Y",
    "Unemployment",
    "VIX",
]

FEATURES_MOM = [f"{c}_mom3" for c in MOM_BASE]

TOLS = {
    "GDP_YoY":   1,
    "CPI_YoY":   1,
    "FedFunds":  0.5,
    "UST10Y":    0.5,
    "Spread_10Y_2Y":     1.8,
    "Unemployment":      2,
    "VIX":               2,
    "INDPRO_YoY":        1.3,
    "HousingStarts_YoY": 2.3,
    "BAA10Y_Spread":     1.3,
    "UMich_Sentiment":   0.5,
    "SP500_MOM_3M": 2,
}

REGIME_COL = "Regime"

# ============================================================
# === UTILITAIRES MATH
# ============================================================

def solve(A, b):
    return np.linalg.solve(A, b)

def annual_vol_to_daily(sigma_annual, periods=252):
    return sigma_annual / np.sqrt(periods)

def ann_from_daily(mu_d, sig_d, periods=252):
    mu_a = (1 + mu_d)**periods - 1
    sig_a = sig_d * np.sqrt(periods)
    return mu_a, sig_a

def shrink_covariance(Sigma: pd.DataFrame, lam: float = 0.25) -> pd.DataFrame:
    diag_vals = np.diag(np.diag(Sigma.values))
    T = pd.DataFrame(diag_vals, index=Sigma.index, columns=Sigma.columns)
    Sigma_shrink = lam * T + (1.0 - lam) * Sigma
    return Sigma_shrink

def project_long_only_capped(w: pd.Series, cap: float = 0.5) -> pd.Series:
    w2 = w.clip(lower=0.0)
    w2 = w2.clip(upper=cap)
    s = w2.sum()
    return w2 / s if s > 0 else w2

# ============================================================
# === CHARGEMENT & PRÉPA DONNÉES
# ============================================================

def load_macro(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()

    # Momentum 3M Macro (calculé mensuellement par défaut)
    for col in MOM_BASE:
        if col in df.columns:
            df[f"{col}_mom3"] = df[col].diff(3)

    # Labellisation régime
    def label_regime(row):
        gdp_m = row.get("GDP_YoY_mom3", np.nan)
        unemp_m = row.get("Unemployment_mom3", np.nan)
        eps_gdp = 0.3
        eps_unemp = 0.1
        if pd.isna(gdp_m) or pd.isna(unemp_m):
            return "Stable"
        if (gdp_m >= eps_gdp) and (unemp_m <= -eps_unemp):
            return "Croissance"
        elif (gdp_m <= -eps_gdp) and (unemp_m >= eps_unemp):
            return "Récession"
        else:
            return "Stable"

    df[REGIME_COL] = df.apply(label_regime, axis=1)
    return df

def load_daily_momentum(path: str) -> pd.DataFrame:
    """Charge le momentum daily du SP500 généré séparément."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    # On s'attend à une colonne 'SP500_MOM_3M'
    return df

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    available_assets = [a for a in ASSETS if a in df.columns]
    return df[available_assets].dropna(how="all")

def vectorisation(macro_hist: pd.DataFrame):
    df = macro_hist[FEATURES_LEVEL].dropna(how="any").copy()
    mu = df.mean()
    sigma = df.std(ddof=0).replace(0, np.nan)
    Z = (df - mu) / sigma
    return Z, mu, sigma

def similar_period(mu: pd.Series,
                   sigma: pd.Series,
                   Z: pd.DataFrame,
                   macro_hist: pd.DataFrame,
                   x_level: pd.Series,
                   k: int,
                   regime_choice: str):

    mask_regime = macro_hist[REGIME_COL] == regime_choice
    if not mask_regime.any():
        return None

    Z_reg = Z.loc[mask_regime]
    if Z_reg.empty:
        return None

    # Z-score du point courant
    zx = (x_level - mu) / sigma
    
    common_cols = Z_reg.columns.intersection(zx.index)
    if len(common_cols) == 0:
        return None

    tol_scales = [1.0, 1.5, 2.0, 2.5, 3, 4, 5]
    similar = None

    for scale in tol_scales:
        mask = np.ones(len(Z_reg), dtype=bool)
        for f, base_tol in TOLS.items():
            if f not in Z_reg.columns or f not in zx.index:
                continue
            tol = base_tol * scale
            mask &= (Z_reg[f] - zx[f]).abs() <= tol

        similar = Z_reg[mask]
        if not similar.empty:
            break

    if similar is None or similar.empty:
        return None

    similar_num = similar[common_cols].astype(float)
    zx_num = zx[common_cols].astype(float)
    diff = similar_num.sub(zx_num, axis=1)
    dist_array = np.sqrt(np.sum(diff.values ** 2, axis=1))
    dist = pd.Series(dist_array, index=similar.index)

    k_eff = min(k, len(dist))
    nearest = dist.nsmallest(k_eff)
    return pd.DatetimeIndex(nearest.index)


def compute_momentum_score(macro_hist: pd.DataFrame,
                           similar_dates: pd.DatetimeIndex,
                           mom_features: list[str]) -> float:
    available_mom = [c for c in mom_features if c in macro_hist.columns]
    if not available_mom:
        return 0.0
    mom_hist = macro_hist.loc[similar_dates, available_mom].dropna(how="any")
    if mom_hist.empty:
        return 0.0
    global_mom = macro_hist[available_mom].dropna(how="any")
    mu_m = global_mom.mean()
    sigma_m = global_mom.std(ddof=0).replace(0, np.nan)
    Z_mom = (mom_hist - mu_m) / sigma_m
    score = Z_mom.abs().mean().mean()
    return float(score)


def adjust_mu_sigma(m: pd.Series,
                    Sigma: pd.DataFrame,
                    regime: str,
                    momentum_score: float) -> tuple[pd.Series, pd.DataFrame]:
    base_beta = 0.3
    if regime == "Récession":
        gamma = 0.7
    elif regime == "Croissance":
        gamma = 0.3
    else:
        gamma = 0.5

    factor_sigma = 1.0 + base_beta * momentum_score
    factor_mu = 1.0 + gamma * momentum_score

    Sigma_adj = Sigma * factor_sigma
    m_adj = m / factor_mu
    return m_adj, Sigma_adj


# ============================================================
# === ALLOCATION À UNE DATE t (MODIFIÉE POUR DAILY MOM)
# ============================================================

def compute_allocation_at_date(
    decision_date: pd.Timestamp,
    macro_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    daily_mom_df: pd.DataFrame  # <--- Ajout de l'argument
) -> tuple[pd.Series, float, float]:
    
    # 1. Récupérer l'historique macro strictement avant t
    macro_hist = macro_df.loc[macro_df.index < decision_date].copy()
    if macro_hist.empty:
        return None, None, None

    # 2. Construire le "Point courant" (row_t)
    #    Comme decision_date est un vendredi (ex: 15 mai), la macro disponible est celle fin avril.
    #    On utilise 'asof' ou une recherche index <= date
    past_macro_dates = macro_df.index[macro_df.index <= decision_date]
    if len(past_macro_dates) == 0:
        return None, None, None
    
    # On prend la dernière donnée macro connue
    last_macro_date = past_macro_dates[-1]
    row_t = macro_df.loc[last_macro_date].copy()
    
    # --- AJUSTEMENT VITAL : INJECTION DU MOMENTUM DAILY ---
    # On écrase la valeur mensuelle de 'SP500_MOM_3M' par la valeur précise du jour (si dispo)
    # On cherche la valeur dispo dans daily_mom_df à la decision_date (ou juste avant)
    if not daily_mom_df.empty:
        # On utilise asof pour trouver la date valide la plus proche dans le passé immédiat
        idx_loc = daily_mom_df.index.get_indexer([decision_date], method='pad')[0]
        if idx_loc != -1:
            fresh_date = daily_mom_df.index[idx_loc]
            # On vérifie qu'on ne prend pas une donnée trop vieille (ex: > 10 jours)
            if (decision_date - fresh_date).days < 10:
                fresh_val = daily_mom_df.iloc[idx_loc]['SP500_MOM_3M']
                row_t['SP500_MOM_3M'] = fresh_val
                # print(f"DEBUG {decision_date.date()}: Updated MOM to {fresh_val:.4f} from {fresh_date.date()}")

    regime_t = row_t[REGIME_COL]
    x_level = row_t[FEATURES_LEVEL]

    # Vectorisation sur historique
    Z_hist, mu_level, sigma_level = vectorisation(macro_hist)

    # Périodes similaires 
    similar_dates = similar_period(
        mu_level, sigma_level, Z_hist, macro_hist, x_level, K_TOP, regime_t
    )
    if similar_dates is None or len(similar_dates) == 0:
        return None, None, None

    # Filtre cutoff marché
    cutoff = pd.Timestamp(CUTOFF_MARKET)
    similar_dates = similar_dates[similar_dates >= cutoff]
    if len(similar_dates) == 0:
        return None, None, None

    momentum_score = compute_momentum_score(macro_hist, similar_dates, FEATURES_MOM)

    # Fenêtres de marché (Market Window)
    months = similar_dates.to_period("M").unique().sort_values()
    extended_periods = sorted({p + i for p in months for i in range(0, 4)})
    extended_periods = [p for p in extended_periods if p.end_time < decision_date]

    if not extended_periods:
        return None, None, None

    global_start = extended_periods[0].start_time.normalize()
    global_end = extended_periods[-1].end_time.normalize()
    prices_train = prices_df.loc[global_start:global_end].dropna(how="all")
    if prices_train.empty:
        return None, None, None

    chunks = []
    for p in extended_periods:
        m = (prices_train.index >= p.start_time) & (prices_train.index <= p.end_time)
        chunk = prices_train.loc[m]
        if not chunk.empty:
            chunks.append(chunk)
    if not chunks:
        return None, None, None

    daily_on_similar = pd.concat(chunks).sort_index()
    returns = daily_on_similar.pct_change().dropna(how="all")
    if returns.empty:
        return None, None, None

    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    Sigma = cov_matrix.fillna(0)
    Sigma_shrink = shrink_covariance(Sigma, lam=LAMBDA_SHRINK)
    mu_s = mean_returns.fillna(0)
    mu_adj, Sigma_adj = adjust_mu_sigma(mu_s, Sigma_shrink, regime_t, momentum_score)

    # Markowitz
    assets = mu_adj.index
    S = Sigma_adj.values
    m_vec = mu_adj.values
    one = np.ones_like(m_vec)

    try:
        Sinv_1 = solve(S, one)
        Sinv_m = solve(S, m_vec)
    except Exception:
        return None, None, None

    a = float(one @ Sinv_1)
    b = float(one @ Sinv_m)
    c = float(m_vec @ Sinv_m)
    denom = (a * c - b**2)
    if denom <= 0 or a <= 0:
        return None, None, None

    w_gmv = Sinv_1 / a
    w_mrr = Sinv_m / b
    sigma_gmv = np.sqrt(1.0 / a)

    base_target = TARGET_VOL_ANN
    # Ajustement cible selon régime
    if regime_t == "Récession":
        adjusted_target = base_target * 1
    elif regime_t == "Croissance":
        adjusted_target = base_target * 1
    else:
        adjusted_target = base_target * 1

    sigma_target_daily = annual_vol_to_daily(adjusted_target)

    if sigma_target_daily < sigma_gmv:
        w_vec = w_gmv
    else:
        tau = np.sqrt((a * (sigma_target_daily**2) - 1.0) / denom)
        w_vec = (b * tau) * w_mrr + (1.0 - b * tau) * w_gmv
    w = pd.Series(w_vec, index=assets)

    if LONG_ONLY:
        w = project_long_only_capped(w, cap=MAX_WEIGHT)

    return w, momentum_score, sigma_target_daily


def plot_performance_comparison(equity_df: pd.DataFrame,
                                prices_df: pd.DataFrame,
                                output_path: str = "backtest_comparison.png"):
    strat_eq = equity_df["Equity"].copy()
    strat_eq = strat_eq / strat_eq.iloc[0]

    start, end = strat_eq.index.min(), strat_eq.index.max()
    px = prices_df.loc[start:end].copy()
    rets = px.pct_change().dropna(how="all")

    sp_eq = None
    if "Equity_SP500" in rets.columns:
        sp_rets = rets["Equity_SP500"]
        sp_eq = (1.0 + sp_rets).cumprod()
    
    curves = {"Ma stratégie (Weekly Rebal)": strat_eq}
    if sp_eq is not None:
        curves["S&P 500 seul"] = sp_eq

    curves_df = pd.DataFrame(curves)
    for col in curves_df.columns:
        s = curves_df[col].dropna()
        if len(s) == 0: continue
        curves_df[col] = curves_df[col] / s.iloc[0]

    curves_df.dropna(how="all", inplace=True)

    plt.figure(figsize=(10, 6))
    for col in curves_df.columns:
        plt.plot(curves_df.index, curves_df[col], label=col)
    plt.xlabel("Date")
    plt.ylabel("Valeur normalisée")
    plt.title("Comparaison Stratégie Hebdomadaire vs SP500")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Graphique sauvegardé : {output_path}")


# ============================================================
# === BACKTEST (MAIN) - VERSION HEBDOMADAIRE
# ============================================================

def backtest():
    print("Chargement des données...")
    macro_df = load_macro(MACRO_CSV)
    prices_df = load_prices(PRICES_CSV)
    
    # CHARGEMENT DU MOMENTUM DAILY
    try:
        mom_daily_df = load_daily_momentum(MOM_DAILY_CSV)
        print(f"Fichier Momentum Daily chargé : {len(mom_daily_df)} lignes.")
    except FileNotFoundError:
        print(f"ERREUR : Le fichier {MOM_DAILY_CSV} est introuvable.")
        return

    # === LOGIQUE TEMPORELLE MODIFIÉE : FREQUENCE HEBDOMADAIRE ===
    # Au lieu de prendre les dates du fichier macro, on génère des Vendredis (W-FRI)
    # On commence à BACKTEST_START et on s'arrête à la fin des prix dispos
    end_date = prices_df.index[-1]
    
    # On crée la liste des dates de rebalancement (tous les vendredis)
    rebal_dates = pd.date_range(start=BACKTEST_START, end=end_date, freq='W-FRI')
    
    # On filtre pour commencer après CUTOFF_MARKET si besoin
    rebal_dates = rebal_dates[rebal_dates >= pd.Timestamp(CUTOFF_MARKET)]

    equity_curve = []
    all_portfolios = []
    all_daily_returns = []

    current_portfolio = None
    current_value = 1.0

    print(f"Démarrage du backtest hebdomadaire sur {len(rebal_dates)} périodes...")

    for i in range(len(rebal_dates) - 1):
        t = rebal_dates[i]
        t_next = rebal_dates[i+1]

        # Calcul allocation en passant le daily_mom_df
        w, mom_score, sigma_target = compute_allocation_at_date(t, macro_df, prices_df, mom_daily_df)
        
        if w is None:
            # Si calcul impossible (manque de data), on garde le portefeuille précédent
            print(f"[{t.date()}] ECHEC : Pas de période similaire trouvée (ou données manquantes). On garde les positions.")
            if current_portfolio is None:
                continue
        else:
            current_portfolio = w
            
            # Log simple pour debug (1 fois sur 10)
            if i % 10 == 0:
                print(f"Rebal {t.date()} | Regime: {macro_df.index[macro_df.index<=t][-1].date()} | Mom Score: {mom_score:.2f}")

            port_row = {
                "Date": t,
                "MomentumScore": mom_score,
                "SigmaTargetDaily": sigma_target,
            }
            for asset in ASSETS:
                port_row[f"w_{asset}"] = current_portfolio.get(asset, 0.0)
            all_portfolios.append(port_row)

        # Application des poids sur la semaine à venir (t exclu -> t_next inclus)
        if current_portfolio is not None:
            window_prices = prices_df.loc[(prices_df.index > t) & (prices_df.index <= t_next)]
            if window_prices.empty:
                continue

            window_rets = window_prices.pct_change().dropna(how="all")
            valid_assets = [a for a in ASSETS if a in window_rets.columns]
            if not valid_assets:
                continue

            w_vec = current_portfolio.reindex(valid_assets).fillna(0.0)
            port_rets = (window_rets[valid_assets] * w_vec).sum(axis=1)

            for dt, r in port_rets.items():
                current_value *= (1.0 + r)
                equity_curve.append((dt, current_value))
                all_daily_returns.append((dt, r))

    if not equity_curve:
        print("Aucun résultat de backtest.")
        return

    equity_df = pd.DataFrame(equity_curve, columns=["Date", "Equity"])
    equity_df.set_index("Date", inplace=True)

    daily_ret_df = pd.DataFrame(all_daily_returns, columns=["Date", "Ret"])
    daily_ret_df.set_index("Date", inplace=True)

    # Stats
    mu_d = daily_ret_df["Ret"].mean()
    sig_d = daily_ret_df["Ret"].std(ddof=0)
    mu_a, sig_a = ann_from_daily(mu_d, sig_d)
    sharpe = mu_a / sig_a if sig_a > 0 else 0.0

    print("\n=== Résultats Backtest Hebdomadaire ===")
    print(f"Période : {equity_df.index.min().date()} → {equity_df.index.max().date()}")
    print(f"Rendement annualisé : {mu_a*100:.2f}%")
    print(f"Volatilité annualisée : {sig_a*100:.2f}%")
    print(f"Sharpe (rf≈0) : {sharpe:.2f}")

    # Benchmark SP500
    sp_col = "Equity_SP500"
    if sp_col in prices_df.columns and not daily_ret_df.empty:
        start_date = daily_ret_df.index.min()
        end_date = daily_ret_df.index.max()
        sp_prices = prices_df.loc[start_date:end_date, sp_col]
        sp_rets = sp_prices.pct_change().dropna()
        mu_a_sp, sig_a_sp = ann_from_daily(sp_rets.mean(), sp_rets.std(ddof=0))
        sharpe_sp = mu_a_sp / sig_a_sp if sig_a_sp > 0 else 0.0

        print("\n=== Benchmark S&P 500 ===")
        print(f"Rendement annualisé : {mu_a_sp*100:.2f}%")
        print(f"Volatilité annualisée : {sig_a_sp*100:.2f}%")
        print(f"Sharpe : {sharpe_sp:.2f}")

    # Sauvegardes
    equity_df.to_csv("backtest_equity_curve.csv", index=True)
    if all_portfolios:
        pd.DataFrame(all_portfolios).set_index("Date").to_csv("backtest_portfolios.csv", index=True)

    plot_performance_comparison(equity_df, prices_df, output_path="backtest_comparison.png")

    # Max Drawdown
    def get_max_drawdown(series):
        roll_max = series.cummax()
        dd = (series - roll_max) / roll_max
        return dd.min()

    print(f"Max Drawdown Stratégie : {get_max_drawdown(equity_df['Equity'])*100:.2f}%")

if __name__ == "__main__":
    backtest()
