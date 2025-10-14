#!/usr/bin/env python3
"""
Application Streamlit interactive pour l'analyse de portefeuille
avec curseurs pour modifier les paramètres en temps réel
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import cvxpy as cp
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(
    page_title="Portfolio Rebalancing Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("📈 Portfolio Analysis & Rebalancing Tool")
st.markdown("**Application interactive pour l'analyse et le rééquilibrage de portefeuille**")

# Fonctions du notebook (adaptées pour Streamlit)
@st.cache_data
def fetch_prices(tickers, start_date, end_date):
    """Télécharge les prix avec cache pour Streamlit"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        data = data.sort_index().ffill().dropna(how="all")
        return data
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données: {e}")
        return None

def to_returns(prices):
    return prices.pct_change().dropna(how="all")

def annualize_return(daily_returns, periods_per_year=252):
    cum = (1 + daily_returns).prod()
    n = len(daily_returns)
    return cum ** (periods_per_year / n) - 1 if n > 0 else np.nan

def annualize_vol(daily_returns, periods_per_year=252):
    return daily_returns.std(ddof=0) * np.sqrt(periods_per_year)

def sharpe(daily_returns, rf_daily=0.0):
    excess = daily_returns - rf_daily
    vol = excess.std(ddof=0)
    return (excess.mean() / vol) * np.sqrt(252) if float(vol) > 0 else np.nan

def max_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return drawdown.min()

def min_variance_weights(cov, max_weight=None, min_weight=0):
    n = cov.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0]
    if max_weight is not None:
        constraints.append(w <= max_weight)
    if min_weight > 0:
        constraints.append(w >= min_weight)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov)), constraints)
    prob.solve(verbose=False)
    if w.value is None:
        raise RuntimeError("Optimization failed.")
    wv = np.clip(w.value, 0, 1)
    return wv / wv.sum() if wv.sum() > 0 else wv

def equal_weight_targets(assets):
    # S'assurer que assets est une liste simple
    if hasattr(assets, 'columns'):
        assets = list(assets.columns)
    elif not isinstance(assets, list):
        assets = list(assets)
    
    w = np.repeat(1/len(assets), len(assets))
    return pd.Series(w, index=assets)

def min_variance_targets(returns_window, max_weight=0.5, min_weight=0):
    cov = returns_window.cov().values
    assets = list(returns_window.columns)
    w = min_variance_weights(cov, max_weight=max_weight, min_weight=min_weight)
    w = w / w.sum() if w.sum() > 0 else w
    return pd.Series(w, index=assets)

def min_variance_quarterly(prices, lookback_days=252, tc_bps=5.0, max_weight=None, min_weight=0):
    """Minimum Variance avec rééquilibrage trimestriel"""
    prices = prices.dropna(how="all").sort_index()
    rets = to_returns(prices).fillna(0.0)
    assets = list(prices.columns)
    
    # Rebalancing trimestriel
    rebal_dates = pd.to_datetime(sorted(set(rets.index.to_period("Q").asfreq("Q").to_timestamp()))).intersection(rets.index)
    
    w = pd.Series(0.0, index=assets)
    equity = [1.0]
    daily_pnl = []
    weight_history = []
    
    for t, date in enumerate(rets.index):
        # Rebalance si date dans rebal_dates et historique suffisant
        if date in rebal_dates:
            window_start = rets.index.searchsorted(date) - lookback_days
            if window_start >= 0:
                window = rets.iloc[max(0, window_start):rets.index.get_loc(date)]
                if window.shape[0] >= 60:  # au moins ~3 mois
                    # Calcul du portefeuille Minimum Variance
                    cov = window.cov().values
                    
                    # Optimisation Min Variance avec contraintes
                    n = len(assets)
                    w_opt = cp.Variable(n)
                    constraints = [cp.sum(w_opt) == 1, w_opt >= 0]
                    
                    # Ajouter contraintes de poids
                    if max_weight is not None:
                        constraints.append(w_opt <= max_weight)
                    if min_weight > 0:
                        constraints.append(w_opt >= min_weight)
                    
                    prob = cp.Problem(cp.Minimize(cp.quad_form(w_opt, cov)), constraints)
                    prob.solve(verbose=False)
                    
                    if w_opt.value is not None:
                        target_w = pd.Series(w_opt.value, index=assets)
                        target_w = target_w / target_w.sum() if target_w.sum() > 0 else target_w
                    else:
                        # Fallback vers equal weight
                        target_w = pd.Series(1/len(assets), index=assets)
                    
                    # Coûts de transaction
                    delta = (target_w - w).abs().sum()
                    cost = delta * (tc_bps / 10000.0)
                    equity[-1] = equity[-1] * (1 - cost)
                    
                    w = target_w.copy()
        
        # PnL quotidien
        r = (w * rets.loc[date]).sum()
        daily_pnl.append(r)
        equity.append(equity[-1] * (1 + r))
        weight_history.append(w.copy())
    
    equity_series = pd.Series(equity[1:], index=rets.index, name="equity")
    pnl_series = pd.Series(daily_pnl, index=rets.index, name="ret")
    weights_df = pd.DataFrame(weight_history, index=rets.index)
    
    return {"equity": equity_series, "returns": pnl_series, "weights": weights_df}

def optimal_portfolio_monthly(prices, lookback_days=252, tc_bps=5.0, max_weight=None, min_weight=0):
    """Portefeuille optimal avec rééquilibrage mensuel"""
    prices = prices.dropna(how="all").sort_index()
    rets = to_returns(prices).fillna(0.0)
    assets = list(prices.columns)
    
    # Rebalancing mensuel
    rebal_dates = pd.to_datetime(sorted(set(rets.index.to_period("M").asfreq("M").to_timestamp()))).intersection(rets.index)
    
    w = pd.Series(0.0, index=assets)
    equity = [1.0]
    daily_pnl = []
    weight_history = []
    
    for t, date in enumerate(rets.index):
        # Rebalance si date dans rebal_dates et historique suffisant
        if date in rebal_dates:
            window_start = rets.index.searchsorted(date) - lookback_days
            if window_start >= 0:
                window = rets.iloc[max(0, window_start):rets.index.get_loc(date)]
                if window.shape[0] >= 30:  # au moins ~1 mois
                    # Calcul du portefeuille optimal (Max Sharpe)
                    mu = window.mean().values
                    cov = window.cov().values
                    
                    # Optimisation Max Sharpe avec contraintes
                    n = len(assets)
                    w_opt = cp.Variable(n)
                    constraints = [cp.sum(w_opt) == 1, w_opt >= 0]
                    
                    # Ajouter contraintes de poids
                    if max_weight is not None:
                        constraints.append(w_opt <= max_weight)
                    if min_weight > 0:
                        constraints.append(w_opt >= min_weight)
                    
                    prob = cp.Problem(cp.Maximize(mu @ w_opt), constraints)
                    prob.solve(verbose=False)
                    
                    if w_opt.value is not None:
                        target_w = pd.Series(w_opt.value, index=assets)
                        target_w = target_w / target_w.sum() if target_w.sum() > 0 else target_w
                    else:
                        # Fallback vers equal weight
                        target_w = pd.Series(1/len(assets), index=assets)
                    
                    # Coûts de transaction
                    delta = (target_w - w).abs().sum()
                    cost = delta * (tc_bps / 10000.0)
                    equity[-1] = equity[-1] * (1 - cost)
                    
                    w = target_w.copy()
        
        # PnL quotidien
        r = (w * rets.loc[date]).sum()
        daily_pnl.append(r)
        equity.append(equity[-1] * (1 + r))
        weight_history.append(w.copy())
    
    equity_series = pd.Series(equity[1:], index=rets.index, name="equity")
    pnl_series = pd.Series(daily_pnl, index=rets.index, name="ret")
    weights_df = pd.DataFrame(weight_history, index=rets.index)
    
    return {"equity": equity_series, "returns": pnl_series, "weights": weights_df}

def optimal_portfolio_quarterly(prices, lookback_days=252, tc_bps=5.0, max_weight=None, min_weight=0):
    """Portefeuille optimal avec rééquilibrage trimestriel"""
    prices = prices.dropna(how="all").sort_index()
    rets = to_returns(prices).fillna(0.0)
    assets = list(prices.columns)
    
    # Rebalancing trimestriel
    rebal_dates = pd.to_datetime(sorted(set(rets.index.to_period("Q").asfreq("Q").to_timestamp()))).intersection(rets.index)
    
    w = pd.Series(0.0, index=assets)
    equity = [1.0]
    daily_pnl = []
    weight_history = []
    
    for t, date in enumerate(rets.index):
        # Rebalance si date dans rebal_dates et historique suffisant
        if date in rebal_dates:
            window_start = rets.index.searchsorted(date) - lookback_days
            if window_start >= 0:
                window = rets.iloc[max(0, window_start):rets.index.get_loc(date)]
                if window.shape[0] >= 60:  # au moins ~3 mois
                    # Calcul du portefeuille optimal (Max Sharpe)
                    mu = window.mean().values
                    cov = window.cov().values
                    
                    # Optimisation Max Sharpe avec contraintes
                    n = len(assets)
                    w_opt = cp.Variable(n)
                    constraints = [cp.sum(w_opt) == 1, w_opt >= 0]
                    
                    # Ajouter contraintes de poids
                    if max_weight is not None:
                        constraints.append(w_opt <= max_weight)
                    if min_weight > 0:
                        constraints.append(w_opt >= min_weight)
                    
                    prob = cp.Problem(cp.Maximize(mu @ w_opt), constraints)
                    prob.solve(verbose=False)
                    
                    if w_opt.value is not None:
                        target_w = pd.Series(w_opt.value, index=assets)
                        target_w = target_w / target_w.sum() if target_w.sum() > 0 else target_w
                    else:
                        # Fallback vers equal weight
                        target_w = pd.Series(1/len(assets), index=assets)
                    
                    # Coûts de transaction
                    delta = (target_w - w).abs().sum()
                    cost = delta * (tc_bps / 10000.0)
                    equity[-1] = equity[-1] * (1 - cost)
                    
                    w = target_w.copy()
        
        # PnL quotidien
        r = (w * rets.loc[date]).sum()
        daily_pnl.append(r)
        equity.append(equity[-1] * (1 + r))
        weight_history.append(w.copy())
    
    equity_series = pd.Series(equity[1:], index=rets.index, name="equity")
    pnl_series = pd.Series(daily_pnl, index=rets.index, name="ret")
    weights_df = pd.DataFrame(weight_history, index=rets.index)
    
    return {"equity": equity_series, "returns": pnl_series, "weights": weights_df}

def simple_backtest(prices, target_func, lookback_days=252, tc_bps=5.0):
    """Version simplifiée du backtest pour l'interface"""
    prices = prices.dropna(how="all").sort_index()
    rets = to_returns(prices).fillna(0.0)
    assets = list(prices.columns)
    
    # Rebalancing mensuel
    rebal_dates = pd.to_datetime(sorted(set(rets.index.to_period("M").asfreq("M").to_timestamp()))).intersection(rets.index)
    
    w = pd.Series(0.0, index=assets)
    equity = [1.0]
    daily_pnl = []
    weight_history = []
    
    for t, date in enumerate(rets.index):
        # Rebalance si date dans rebal_dates et historique suffisant
        if date in rebal_dates:
            window_start = rets.index.searchsorted(date) - lookback_days
            if window_start >= 0:
                window = rets.iloc[max(0, window_start):rets.index.get_loc(date)]
                if window.shape[0] >= 60:  # au moins ~3 mois
                    if target_func == equal_weight_targets:
                        target_w = target_func(window.columns)
                    else:
                        target_w = target_func(window)
                    
                    # Coûts de transaction
                    delta = (target_w - w).abs().sum()
                    cost = delta * (tc_bps / 10000.0)
                    equity[-1] = equity[-1] * (1 - cost)
                    
                    w = target_w.copy()
        
        # PnL quotidien
        r = (w * rets.loc[date]).sum()
        daily_pnl.append(r)
        equity.append(equity[-1] * (1 + r))
        weight_history.append(w.copy())
    
    equity_series = pd.Series(equity[1:], index=rets.index, name="equity")
    pnl_series = pd.Series(daily_pnl, index=rets.index, name="ret")
    weights_df = pd.DataFrame(weight_history, index=rets.index)
    
    return {"equity": equity_series, "returns": pnl_series, "weights": weights_df}

# Sidebar avec les contrôles
st.sidebar.header("🎛️ Paramètres du Portefeuille")

# Sélection des tickers
st.sidebar.subheader("📊 Actions")
available_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "NFLX"]
selected_tickers = st.sidebar.multiselect(
    "Sélectionnez les actions:",
    available_tickers,
    default=["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
)

# Période
st.sidebar.subheader("📅 Période d'analyse")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Date de début", value=datetime(2020, 1, 1))
with col2:
    end_date = st.date_input("Date de fin", value=datetime.today())

# Paramètres de backtest
st.sidebar.subheader("⚙️ Paramètres de Backtest")
lookback_days = st.sidebar.slider("Fenêtre de lookback (jours)", 60, 500, 252, 10)
tc_bps = st.sidebar.slider("Coûts de transaction (bps)", 0, 50, 5, 1)
max_weight = st.sidebar.slider("Poids maximum par action (%)", 10, 100, 50, 5) / 100
min_weight = st.sidebar.slider("Poids minimum par action (%)", 0, 20, 0, 1) / 100

# Stratégies
st.sidebar.subheader("📈 Stratégies")
show_equal_weight = st.sidebar.checkbox("Equal Weight", value=True)
show_min_variance = st.sidebar.checkbox("Minimum Variance (Mensuel)", value=True)
show_min_variance_quarterly = st.sidebar.checkbox("Minimum Variance (Trimestriel)", value=False)
show_optimal_monthly = st.sidebar.checkbox("Portefeuille Optimal (Mensuel)", value=False)
show_optimal_quarterly = st.sidebar.checkbox("Portefeuille Optimal (Trimestriel)", value=False)
show_benchmark = st.sidebar.checkbox("Benchmark (SPY)", value=True)

# Bouton de mise à jour
if st.sidebar.button("🔄 Mettre à jour l'analyse", type="primary"):
    st.rerun()

# Contenu principal
if not selected_tickers:
    st.warning("⚠️ Veuillez sélectionner au moins une action dans la barre latérale.")
    st.stop()

# Téléchargement des données
with st.spinner("📥 Téléchargement des données..."):
    prices = fetch_prices(selected_tickers, start_date, end_date)
    
    if prices is None or prices.empty:
        st.error("❌ Impossible de télécharger les données. Vérifiez les tickers et la période.")
        st.stop()

# Calculs des stratégies
results = {}

if show_equal_weight:
    with st.spinner("🔄 Calcul de la stratégie Equal Weight..."):
        try:
            bt_eq = simple_backtest(prices, equal_weight_targets, lookback_days, tc_bps)
            results["Equal Weight"] = bt_eq
        except Exception as e:
            st.error(f"Erreur dans Equal Weight: {e}")

if show_min_variance:
    with st.spinner("🔄 Calcul de la stratégie Minimum Variance (Mensuel)..."):
        try:
            bt_minv = simple_backtest(prices, lambda x: min_variance_targets(x, max_weight, min_weight), lookback_days, tc_bps)
            results["Min Variance (Mensuel)"] = bt_minv
        except Exception as e:
            st.error(f"Erreur dans Minimum Variance (Mensuel): {e}")

if show_min_variance_quarterly:
    with st.spinner("🔄 Calcul de la stratégie Minimum Variance (Trimestriel)..."):
        try:
            bt_minv_q = min_variance_quarterly(prices, lookback_days, tc_bps, max_weight, min_weight)
            results["Min Variance (Trimestriel)"] = bt_minv_q
        except Exception as e:
            st.error(f"Erreur dans Minimum Variance (Trimestriel): {e}")

if show_optimal_monthly:
    with st.spinner("🔄 Calcul du Portefeuille Optimal (Mensuel)..."):
        try:
            bt_optimal_m = optimal_portfolio_monthly(prices, lookback_days, tc_bps, max_weight, min_weight)
            results["Portefeuille Optimal (Mensuel)"] = bt_optimal_m
        except Exception as e:
            st.error(f"Erreur dans Portefeuille Optimal (Mensuel): {e}")

if show_optimal_quarterly:
    with st.spinner("🔄 Calcul du Portefeuille Optimal (Trimestriel)..."):
        try:
            bt_optimal = optimal_portfolio_quarterly(prices, lookback_days, tc_bps, max_weight, min_weight)
            results["Portefeuille Optimal (Trimestriel)"] = bt_optimal
        except Exception as e:
            st.error(f"Erreur dans Portefeuille Optimal (Trimestriel): {e}")

if show_benchmark:
    with st.spinner("🔄 Calcul du benchmark SPY..."):
        try:
            spy = fetch_prices(["SPY"], start_date, end_date)
            if spy is not None and not spy.empty:
                spy_ret = spy["SPY"].pct_change().reindex(prices.index).fillna(0.0)
                spy_equity = (1 + spy_ret).cumprod()
                results["SPY Benchmark"] = {"equity": spy_equity, "returns": spy_ret}
        except Exception as e:
            st.error(f"Erreur dans le benchmark: {e}")

# Affichage des résultats
if results:
    # Métriques de performance
    st.header("📊 Métriques de Performance")
    
    metrics_data = []
    for name, result in results.items():
        if "returns" in result:
            ret = result["returns"]
            metrics_data.append({
                "Stratégie": name,
                "Rendement Annuel (%)": f"{annualize_return(ret) * 100:.2f}",
                "Volatilité Annuelle (%)": f"{annualize_vol(ret) * 100:.2f}",
                "Ratio de Sharpe": f"{sharpe(ret):.2f}",
                "Drawdown Max (%)": f"{max_drawdown(result['equity']) * 100:.2f}",
                "Rendement Total (%)": f"{(result['equity'].iloc[-1] - 1) * 100:.2f}"
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
    
    # Graphique des courbes d'équité
    st.header("📈 Courbes d'Équité")
    
    fig_equity = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (name, result) in enumerate(results.items()):
        if "equity" in result:
            fig_equity.add_trace(go.Scatter(
                x=result["equity"].index,
                y=result["equity"].values,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    
    fig_equity.update_layout(
        title="Évolution de la valeur du portefeuille",
        xaxis_title="Date",
        yaxis_title="Valeur (normalisée à 1)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Graphique spécial pour le Portefeuille Optimal (Mensuel)
    if "Portefeuille Optimal (Mensuel)" in results:
        st.header("🎯 Portefeuille Optimal (Mensuel) - Analyse Détaillée")
        
        optimal_result = results["Portefeuille Optimal (Mensuel)"]
        
        # Métriques spéciales pour le portefeuille optimal mensuel
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rendement Annuel", f"{annualize_return(optimal_result['returns']) * 100:.2f}%")
        with col2:
            st.metric("Volatilité Annuelle", f"{annualize_vol(optimal_result['returns']) * 100:.2f}%")
        with col3:
            st.metric("Ratio de Sharpe", f"{sharpe(optimal_result['returns']):.2f}")
        
        # Graphique des poids du portefeuille optimal mensuel
        if "weights" in optimal_result:
            st.subheader("📊 Répartition du Portefeuille Optimal (Mensuel)")
            
            fig_optimal_weights = go.Figure()
            
            for asset in optimal_result["weights"].columns:
                fig_optimal_weights.add_trace(go.Scatter(
                    x=optimal_result["weights"].index,
                    y=optimal_result["weights"][asset] * 100,
                    mode='lines',
                    name=asset,
                    stackgroup='one',
                    fill='tonexty'
                ))
            
            fig_optimal_weights.update_layout(
                title="Évolution des poids du Portefeuille Optimal (Rééquilibrage Mensuel)",
                xaxis_title="Date",
                yaxis_title="Poids (%)",
                hovermode='x unified',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_optimal_weights, use_container_width=True)
            
            # Tableau des poids moyens
            st.subheader("📋 Poids Moyens du Portefeuille Optimal (Mensuel)")
            avg_weights = optimal_result["weights"].mean() * 100
            weights_df = pd.DataFrame({
                'Action': avg_weights.index,
                'Poids Moyen (%)': avg_weights.values.round(2)
            }).sort_values('Poids Moyen (%)', ascending=False)
            
            st.dataframe(weights_df, use_container_width=True)
    
    # Graphique spécial pour le Portefeuille Optimal (Trimestriel)
    if "Portefeuille Optimal (Trimestriel)" in results:
        st.header("🎯 Portefeuille Optimal (Trimestriel) - Analyse Détaillée")
        
        optimal_result = results["Portefeuille Optimal (Trimestriel)"]
        
        # Métriques spéciales pour le portefeuille optimal trimestriel
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rendement Annuel", f"{annualize_return(optimal_result['returns']) * 100:.2f}%")
        with col2:
            st.metric("Volatilité Annuelle", f"{annualize_vol(optimal_result['returns']) * 100:.2f}%")
        with col3:
            st.metric("Ratio de Sharpe", f"{sharpe(optimal_result['returns']):.2f}")
        
        # Graphique des poids du portefeuille optimal trimestriel
        if "weights" in optimal_result:
            st.subheader("📊 Répartition du Portefeuille Optimal (Trimestriel)")
            
            fig_optimal_weights = go.Figure()
            
            for asset in optimal_result["weights"].columns:
                fig_optimal_weights.add_trace(go.Scatter(
                    x=optimal_result["weights"].index,
                    y=optimal_result["weights"][asset] * 100,
                    mode='lines',
                    name=asset,
                    stackgroup='one',
                    fill='tonexty'
                ))
            
            fig_optimal_weights.update_layout(
                title="Évolution des poids du Portefeuille Optimal (Rééquilibrage Trimestriel)",
                xaxis_title="Date",
                yaxis_title="Poids (%)",
                hovermode='x unified',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_optimal_weights, use_container_width=True)
            
            # Tableau des poids moyens
            st.subheader("📋 Poids Moyens du Portefeuille Optimal (Trimestriel)")
            avg_weights = optimal_result["weights"].mean() * 100
            weights_df = pd.DataFrame({
                'Action': avg_weights.index,
                'Poids Moyen (%)': avg_weights.values.round(2)
            }).sort_values('Poids Moyen (%)', ascending=False)
            
            st.dataframe(weights_df, use_container_width=True)
    
    # Graphique spécial pour Minimum Variance Trimestriel
    if "Min Variance (Trimestriel)" in results:
        st.header("🛡️ Minimum Variance (Trimestriel) - Analyse Détaillée")
        
        minvar_result = results["Min Variance (Trimestriel)"]
        
        # Métriques spéciales pour Minimum Variance trimestriel
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rendement Annuel", f"{annualize_return(minvar_result['returns']) * 100:.2f}%")
        with col2:
            st.metric("Volatilité Annuelle", f"{annualize_vol(minvar_result['returns']) * 100:.2f}%")
        with col3:
            st.metric("Ratio de Sharpe", f"{sharpe(minvar_result['returns']):.2f}")
        
        # Graphique des poids de Minimum Variance trimestriel
        if "weights" in minvar_result:
            st.subheader("📊 Répartition du Minimum Variance (Trimestriel)")
            
            fig_minvar_weights = go.Figure()
            
            for asset in minvar_result["weights"].columns:
                fig_minvar_weights.add_trace(go.Scatter(
                    x=minvar_result["weights"].index,
                    y=minvar_result["weights"][asset] * 100,
                    mode='lines',
                    name=asset,
                    stackgroup='one',
                    fill='tonexty'
                ))
            
            fig_minvar_weights.update_layout(
                title="Évolution des poids du Minimum Variance (Rééquilibrage Trimestriel)",
                xaxis_title="Date",
                yaxis_title="Poids (%)",
                hovermode='x unified',
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig_minvar_weights, use_container_width=True)
            
            # Tableau des poids moyens
            st.subheader("📋 Poids Moyens du Minimum Variance (Trimestriel)")
            avg_weights = minvar_result["weights"].mean() * 100
            weights_df = pd.DataFrame({
                'Action': avg_weights.index,
                'Poids Moyen (%)': avg_weights.values.round(2)
            }).sort_values('Poids Moyen (%)', ascending=False)
            
            st.dataframe(weights_df, use_container_width=True)
    
    # Graphique des poids (pour les autres stratégies de portefeuille)
    portfolio_strategies = {k: v for k, v in results.items() if k not in ["SPY Benchmark", "Portefeuille Optimal (Mensuel)", "Portefeuille Optimal (Trimestriel)", "Min Variance (Trimestriel)"]}
    
    if portfolio_strategies:
        st.header("⚖️ Évolution des Poids des Autres Stratégies")
        
        for name, result in portfolio_strategies.items():
            if "weights" in result:
                st.subheader(f"Poids - {name}")
                
                fig_weights = go.Figure()
                
                for asset in result["weights"].columns:
                    fig_weights.add_trace(go.Scatter(
                        x=result["weights"].index,
                        y=result["weights"][asset] * 100,
                        mode='lines',
                        name=asset,
                        stackgroup='one'
                    ))
                
                fig_weights.update_layout(
                    title=f"Répartition du portefeuille - {name}",
                    xaxis_title="Date",
                    yaxis_title="Poids (%)",
                    hovermode='x unified',
                    height=400
                )
                
                st.plotly_chart(fig_weights, use_container_width=True)
    
    # Graphique de corrélation
    if len(selected_tickers) > 1:
        st.header("🔗 Matrice de Corrélation")
        
        returns = to_returns(prices)
        corr_matrix = returns.corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Matrice de corrélation des rendements"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Statistiques détaillées
    st.header("📋 Statistiques Détaillées")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Données de base")
        st.write(f"**Période:** {start_date} à {end_date}")
        st.write(f"**Actions:** {', '.join(selected_tickers)}")
        st.write(f"**Nombre de jours:** {len(prices)}")
        st.write(f"**Fenêtre de lookback:** {lookback_days} jours")
        st.write(f"**Coûts de transaction:** {tc_bps} bps")
    
    with col2:
        st.subheader("📈 Performance comparative")
        if len(results) > 1:
            # Comparaison des rendements
            comparison_data = []
            for name, result in results.items():
                if "equity" in result:
                    final_return = (result["equity"].iloc[-1] - 1) * 100
                    comparison_data.append({"Stratégie": name, "Rendement Final (%)": final_return})
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
        
        # Informations sur les contraintes de poids
        st.subheader("⚖️ Contraintes de Poids")
        st.write(f"**Poids maximum par action:** {max_weight*100:.1f}%")
        st.write(f"**Poids minimum par action:** {min_weight*100:.1f}%")
        st.write(f"**Coûts de transaction:** {tc_bps} bps")
        st.write(f"**Fenêtre de lookback:** {lookback_days} jours")

else:
    st.warning("⚠️ Aucune stratégie sélectionnée. Veuillez en sélectionner au moins une dans la barre latérale.")

# Footer
st.markdown("---")
st.markdown("**💡 Conseil:** Modifiez les paramètres dans la barre latérale pour voir l'impact sur les performances en temps réel!")
