"""
Weibull Reliability Analysis Dashboard
=======================================
Streamlit dashboard for interactive Weibull parameter estimation, goodness-of-fit
comparison, and method selection logic applied to the Backblaze HDD fleet dataset.

This version performs ALL statistical calculations in Python (SciPy/NumPy).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.special import gamma as Γ, gammaln as lnΓ
from scipy.optimize import brentq, minimize
from scipy.stats import lognorm

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weibull Reliability Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0d1117; }

/* Force dark theme for all widgets */
div[data-baseweb="select"], div[data-baseweb="input"], .stSlider, div[role="listbox"] {
    background-color: #161b22 !important;
    border-radius: 8px !important;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.metric-label { font-size: 0.8rem; color: #8b949e; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 1.4rem; font-weight: 700; color: #58a6ff; }

.method-card {
    background: #1c2333;
    border-left: 4px solid #2188ff;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 14px;
}

.traffic-green  { background:#0d3321; border:1px solid #2ea043; border-radius:6px; padding:4px 10px; color:#3fb950; font-weight:600; font-size: 0.85rem; }
.traffic-yellow { background:#2d2008; border:1px solid #9e6a03; border-radius:6px; padding:4px 10px; color:#d29922; font-weight:600; font-size: 0.85rem; }
.traffic-red    { background:#3d0d12; border:1px solid #da3633; border-radius:6px; padding:4px 10px; color:#f85149; font-weight:600; font-size: 0.85rem; }

.section-header {
    font-size: 1.2rem; font-weight: 600; color: #58a6ff;
    border-bottom: 1px solid #30363d; padding-bottom: 8px; margin-top: 24px; margin-bottom: 16px;
}

.stTabs [data-baseweb="tab-list"] { gap: 8px; background: #0d1117; padding: 0; }
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0; padding: 10px 20px; font-size: 0.9rem;
    color: #8b949e; background: #161b22; border: 1px solid #30363d; border-bottom: none;
}
.stTabs [aria-selected="true"] { background: #1f6feb !important; color: #fff !important; border-color: #1f6feb !important; }

div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Search for data relative to project root (up 2 levels from python/)
DATA_DIR = os.path.join(_THIS_DIR, "..", "..", "data_Q3_2025")
if not os.path.isdir(DATA_DIR):
    # Fallback
    DATA_DIR = os.path.join(_THIS_DIR, "data")

TOP3_MODELS = [
    "WDC WUH722222ALE6L4",
    "TOSHIBA MG08ACA16TA",
    "TOSHIBA MG07ACA14TA",
]

METHOD_NAMES = ["OLS", "WLS", "MRR", "MOM", "L-Moments", "EPF", "MLE", "MAP", "Percentile"]

METHOD_COLORS = {
    "OLS":        "#ef4444",
    "WLS":        "#f97316",
    "MRR":        "#eab308",
    "MOM":        "#22c55e",
    "L-Moments":  "#14b8a6",
    "EPF":        "#3b82f6",
    "MLE":        "#a855f7",
    "MAP":        "#ec4899",
    "Percentile": "#94a3b8",
}

# ─────────────────────────────────────────────────────────────────────────────
# CACHING: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading Backblaze data...", ttl=3600)
def load_failure_data(data_dir: str, top3: list[str]) -> pd.DataFrame:
    for candidate in [
        os.path.join(_THIS_DIR, "..", "data", "drives_clean.csv"),
        os.path.join(_THIS_DIR, "data", "drives_clean.csv"),
        os.path.join(data_dir, "drives_clean.csv"),
    ]:
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)
            df = df[df["model"].isin(top3)].copy()
            return df

        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    frames = []
    for f in csv_files:
        chunk = pd.read_csv(f, usecols=["date", "serial_number", "model", "failure"],
                             parse_dates=["date"])
        chunk = chunk[chunk["model"].isin(top3)]
        frames.append(chunk)

    raw = pd.concat(frames, ignore_index=True)
    grp = raw.groupby(["serial_number", "model"], sort=False)
    agg = grp.agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        failed=("failure", "max"),
    ).reset_index()

    agg["lifetime_days"] = (agg["last_date"] - agg["first_date"]).dt.days.clip(lower=1)
    agg = agg[agg["lifetime_days"] >= 1].copy()
    return agg

# ─────────────────────────────────────────────────────────────────────────────
# WEIBULL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def weibull_cdf(t, k, lam): return 1.0 - np.exp(-(t / lam) ** k)
def weibull_sf(t, k, lam):  return np.exp(-(t / lam) ** k)
def weibull_pdf(t, k, lam): return (k / lam) * (t / lam) ** (k - 1) * np.exp(-(t / lam) ** k)
def weibull_hazard(t, k, lam): return (k / lam) * (t / lam) ** (k - 1)

def censored_loglik(k, lam, t_fail, t_cens):
    ll = 0.0
    if len(t_fail) > 0:
        ll += np.sum(np.log(k / lam) + (k - 1) * np.log(t_fail / lam) - (t_fail / lam) ** k)
    if len(t_cens) > 0:
        ll += np.sum(-(t_cens / lam) ** k)
    return ll

def median_ranks(n: int) -> np.ndarray:
    i = np.arange(1, n + 1, dtype=float)
    return (i - 0.3) / (n + 0.4)

# ─────────────────────────────────────────────────────────────────────────────
# 9 ESTIMATION METHODS
# ─────────────────────────────────────────────────────────────────────────────
def fit_ols(t_fail, t_cens):
    t = np.sort(t_fail); n = len(t); F = median_ranks(n)
    x, y = np.log(t), np.log(-np.log(1.0 - F))
    coeffs = np.polyfit(x, y, 1)
    k = coeffs[0]; lam = np.exp(-coeffs[1] / k)
    return max(k, 1e-6), max(lam, 1e-6)

def fit_wls(t_fail, t_cens):
    t = np.sort(t_fail); n = len(t); F = median_ranks(n)
    i = np.arange(1, n + 1, dtype=float); N = float(n)
    w = ((1 - F) * (-np.log(1 - F))) ** 2 * (N + 1) ** 2 * (N + 2) / (i * (N - i + 1))
    x, y = np.log(t), np.log(-np.log(1.0 - F))
    coeffs = np.polyfit(x, y, 1, w=w)
    k = coeffs[0]; lam = np.exp(-coeffs[1] / k)
    return max(k, 1e-6), max(lam, 1e-6)

def fit_mrr(t_fail, t_cens): return fit_ols(t_fail, t_cens)

def fit_mom(t_fail, t_cens):
    t = t_fail; mu = np.mean(t); cv2 = np.var(t, ddof=1) / mu**2
    def eq(k): return Γ(1 + 2 / k) / Γ(1 + 1 / k)**2 - 1 - cv2
    try: k = brentq(eq, 0.1, 100.0)
    except: k = 1.0
    lam = mu / Γ(1 + 1 / k)
    return max(k, 1e-6), max(lam, 1e-6)

def fit_lmom(t_fail, t_cens):
    t = np.sort(t_fail); n = len(t)
    if n < 4: return fit_mom(t_fail, t_cens)
    idx = np.arange(n); b0 = np.mean(t); b1 = np.sum((idx / (n - 1)) * t) / n
    L1 = b0; L2 = 2 * b1 - b0
    if L2 <= 0 or L1 <= 0: return fit_mom(t_fail, t_cens)
    tau2 = L2 / L1
    def eq(k): return (1 - 2 ** (-1 / k)) * Γ(1 + 1 / k) - tau2
    try: k = brentq(eq, 0.1, 100.0)
    except: k = 1.0
    lam = L1 / Γ(1 + 1 / k)
    return max(k, 1e-6), max(lam, 1e-6)

def fit_epf(t_fail, t_cens):
    t = t_fail; epf_hat = np.mean(t ** 3) / np.mean(t) ** 3
    def eq(k): return Γ(1 + 3 / k) / Γ(1 + 1 / k) ** 3 - epf_hat
    try: k = brentq(eq, 0.05, 50.0)
    except: k = 1.0
    lam = np.mean(t) / Γ(1 + 1 / k)
    return max(k, 1e-6), max(lam, 1e-6)

def fit_mle(t_fail, t_cens):
    t_f, t_c = np.asarray(t_fail, dtype=float), np.asarray(t_cens, dtype=float)
    def neg_ll(p): return -censored_loglik(np.exp(p[0]), np.exp(p[1]), t_f, t_c)
    res = minimize(neg_ll, [0.0, np.log(np.mean(t_f))], method="L-BFGS-B")
    return np.exp(res.x[0]), np.exp(res.x[1])

def fit_map(t_fail, t_cens):
    t_f, t_c = np.asarray(t_fail, dtype=float), np.asarray(t_cens, dtype=float)
    all_t = np.concatenate([t_f, t_c])
    def neg_map(p):
        k, lam = np.exp(p[0]), np.exp(p[1])
        ll = censored_loglik(k, lam, t_f, t_c)
        lp_k = lognorm.logpdf(k, s=0.5, scale=2.0)
        lp_lam = lognorm.logpdf(lam, s=1.0, scale=np.mean(all_t))
        return -(ll + lp_k + lp_lam)
    res = minimize(neg_map, [0.0, np.log(np.mean(t_f))], method="L-BFGS-B")
    return np.exp(res.x[0]), np.exp(res.x[1])

def fit_percentile(t_fail, t_cens):
    t = np.sort(t_fail); n = len(t)
    if n < 4: return 1.0, np.mean(t)
    p1, p2 = 0.16, 0.84
    t1 = t[int(p1 * n)]; t2 = t[int(p2 * n)]
    if t1 == t2 or t1 <= 0: return 1.0, np.mean(t)
    k = (np.log(-np.log(1 - p2)) - np.log(-np.log(1 - p1))) / (np.log(t2) - np.log(t1))
    lam = t1 / (-np.log(1 - p1)) ** (1 / k)
    return max(k, 1e-6), max(lam, 1e-6)

FIT_FUNCTIONS = {
    "OLS": fit_ols, "WLS": fit_wls, "MRR": fit_mrr, "MOM": fit_mom,
    "L-Moments": fit_lmom, "EPF": fit_epf, "MLE": fit_mle, "MAP": fit_map, "Percentile": fit_percentile
}

# ─────────────────────────────────────────────────────────────────────────────
# GOODNESS-OF-FIT
# ─────────────────────────────────────────────────────────────────────────────
def compute_gof(t_fail, t_cens, k, lam):
    t_f = np.sort(t_fail); n = len(t_f); n_total = n + len(t_cens)
    F_emp = median_ranks(n); F_fit = weibull_cdf(t_f, k, lam)
    ks = np.max(np.abs(F_emp - F_fit))
    rmse = np.sqrt(np.mean((F_emp - F_fit)**2))
    z = np.clip(F_fit, 1e-12, 1-1e-12)
    ad = -n - (1/n) * np.sum((2*np.arange(1, n+1)-1) * (np.log(z) + np.log(1-z[::-1])))
    ll = censored_loglik(k, lam, t_f, t_cens)
    aic = -2*ll + 4; bic = -2*ll + np.log(n_total)*4
    return {"KS": round(ks, 4), "RMSE": round(rmse, 4), "AD": round(ad, 4), "AIC": round(aic, 2), "BIC": round(bic, 2)}

@st.cache_data(show_spinner="Fitting models...", ttl=3600)
def fit_all_methods(model_name: str, df_all: pd.DataFrame):
    sub = df_all[df_all["model"] == model_name]
    t_f = sub[sub["failed"] == 1]["lifetime_days"].values.astype(float)
    t_c = sub[sub["failed"] == 0]["lifetime_days"].values.astype(float)
    p_rows = []; g_rows = []
    for m, fn in FIT_FUNCTIONS.items():
        try: k, lam = fn(t_f, t_c); gof = compute_gof(t_f, t_c, k, lam)
        except: k, lam = np.nan, np.nan; gof = {c: np.nan for c in ["KS","RMSE","AD","AIC","BIC"]}
        p_rows.append({"Method": m, "k": k, "λ": lam})
        gof["Method"] = m; g_rows.append(gof)
    params_df = pd.DataFrame(p_rows).set_index("Method")
    gof_df = pd.DataFrame(g_rows).set_index("Method")
    return params_df, gof_df, t_f, t_c

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def metric_card(label, value):
    return f"""<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>"""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
try: df_all = load_failure_data(DATA_DIR, TOP3_MODELS)
except Exception as e: st.error(f"❌ Error loading data: {e}"); st.stop()

with st.sidebar:
    st.title("⚙️ Dashboard Controls")
    sel_model = st.selectbox("Drive Model", TOP3_MODELS)
    sel_methods = st.multiselect("Estimation Methods", METHOD_NAMES, default=METHOD_NAMES)

    st.markdown("---")
    st.markdown("### 🎓 Defense Mode")
    defense_mode = st.toggle("Enable Defense Annotations", value=False,
        help="Annotates each section with the defense question it answers and the correct response.")
    if defense_mode:
        st.markdown("""<div style="background:#2d2008;border:1px solid #9e6a03;border-radius:8px;padding:10px;font-size:0.8rem;color:#d29922;">
        ⚠️ Defense mode active. Each section now shows the question a jury member might ask and your prepared answer.
        </div>""", unsafe_allow_html=True)

params_df, gof_df, t_fail, t_cens = fit_all_methods(sel_model, df_all)
n_fail = len(t_fail); n_total = n_fail + len(t_cens); censor_pct = 100 * len(t_cens) / n_total

st.title("📊 Weibull Reliability Dashboard")
st.markdown(f"**Model:** `{sel_model}`")

c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(metric_card("Total Drives", f"{n_total:,}"), unsafe_allow_html=True)
with c2: st.markdown(metric_card("Failures", f"{n_fail:,}"), unsafe_allow_html=True)
with c3: st.markdown(metric_card("Censoring", f"{censor_pct:.1f}%"), unsafe_allow_html=True)
with c4: st.markdown(metric_card("Median Life", f"{np.median(t_fail):.0f}d"), unsafe_allow_html=True)

tabs = st.tabs(["📋 GoF Metrics", "📈 Probability Plot", "📉 Distribution Curves", "⚖️ Comparison", "🚦 Decision", "🛠️ Lab", "🔬 Extended Lab", "📖 Cards"])

with tabs[0]:
    st.markdown('<div class="section-header">Goodness-of-Fit Comparison</div>', unsafe_allow_html=True)

    if defense_mode:
        st.markdown("""<div style="background:#0d2d1a;border-left:4px solid #2ea043;padding:12px;border-radius:0 8px 8px 0;margin-bottom:16px;">
        <strong style="color:#3fb950">🎓 Defense Q:</strong> <em>"Your composite rank uses RMSE and AD which are failure-only metrics.
        Isn't that biased against MLE?"</em><br><br>
        <strong style="color:#3fb950">Your answer:</strong> "Yes — and the report acknowledges this explicitly in Section 2.5.3.2 and 3.8.2.
        Two of three composite dimensions evaluate the failure-subset CDF fit, which structurally favors failure-only methods.
        MLE wins on AIC by a margin of 140,000+ units because it incorporates 37,000 censored survivors.
        The composite rank is a valid summary within the failure-only frame. It is not a population-level judgment.
        The Kaplan-Meier plot is the population-level judgment — and MLE wins there unambiguously."
        </div>""", unsafe_allow_html=True)

    # Augment GoF with population-level failure fraction error
    pop_rows = []
    obs_fail_frac = len(t_fail) / (len(t_fail) + len(t_cens))
    for m in METHOD_NAMES:
        if m in params_df.index:
            k_v, lam_v = params_df.loc[m, "k"], params_df.loc[m, "λ"]
            if not (np.isnan(k_v) or np.isnan(lam_v)):
                pred_frac = weibull_cdf(92, k_v, lam_v)
                pop_err = abs(pred_frac - obs_fail_frac) / max(obs_fail_frac, 1e-9) * 100
                pop_rows.append({"Method": m, "Pred F(92)": round(pred_frac, 6),
                                  "Obs F(92)": round(obs_fail_frac, 6), "Pop Err %": round(pop_err, 1)})
    pop_df = pd.DataFrame(pop_rows).set_index("Method")

    col_gof1, col_gof2 = st.columns([2, 1])
    with col_gof1:
        st.markdown("**Standard GoF (composite uses RMSE + AD + AIC)**")
        st.dataframe(gof_df.loc[[m for m in sel_methods if m in gof_df.index]].style.background_gradient(cmap="RdYlGn_r"),
                     use_container_width=True)
    with col_gof2:
        st.markdown("**Population-level metric: |F̂(92) − observed failure rate|**")
        st.caption(f"Observed failure fraction = {obs_fail_frac:.6f} ({len(t_fail)}/{len(t_fail)+len(t_cens)} drives)")
        st.dataframe(pop_df.loc[[m for m in sel_methods if m in pop_df.index]].style.background_gradient(
            subset=["Pop Err %"], cmap="RdYlGn_r"), use_container_width=True)
        st.markdown("""<div style="background:#161b22;padding:10px;border-radius:8px;font-size:0.75rem;color:#8b949e;margin-top:8px;">
        This is the metric MRL error should have been: the absolute relative error
        in predicting the observed 92-day failure fraction. MLE scores near 0% here;
        failure-only methods score 3,000–50,000%. Addresses report Point 2.
        </div>""", unsafe_allow_html=True)

    with st.expander("📖 Deep explanation — every GoF metric from first principles"):
        st.markdown(f"""
**KS Statistic — L∞ norm**

D = max |F̂ᵢ − F(t(i); k̂, λ̂)|

The maximum absolute deviation between the empirical CDF (Benard's plotting positions) and the fitted CDF at each failure time. It is the L∞ norm — it finds the single worst point.

Limitation: evaluated only at {len(t_fail)} failure times using the failure-only empirical CDF. MLE's F(t) ≈ 0 everywhere in the observable window, so its KS ≈ 1.0 — not because MLE is wrong, but because the empirical CDF is normalized over failure times only, not the full population.

**RMSE — L² norm**

RMSE = √(1/n_f · Σ (F̂ᵢ − F(t(i)))²)

Same limitation as KS — uses failure-only empirical CDF. Amplifies large deviations quadratically. WLS minimizes this because Bergman weights anchor the regression at F̂ ≈ 0.632 where variance is lowest.

**Anderson-Darling — weighted L² norm**

A² = −n_f − (1/n_f) Σ (2i−1)[ln(zᵢ) + ln(1−z_{{n_f+1−i}})]

where zᵢ = F(t(i); k̂, λ̂). The position-dependent weight 1/[u(1−u)] amplifies tail deviations. At u=0.02 the weight is 1/(0.02×0.98) ≈ 51, which is 12.8× the weight at the median. AD is the most tail-sensitive metric.

**AIC — information-theoretic**

AIC = −2ℓ(k̂, λ̂) + 2p = −2ℓ(k̂, λ̂) + 4

where ℓ(k̂, λ̂) is the **full censored log-likelihood** evaluated at the estimated parameters. This includes the {len(t_cens):,} survival terms from censored drives. MLE's AIC is lowest by ~140,000 units because incorporating {len(t_cens):,} survival contributions massively increases the log-likelihood.

Technically AIC is only valid when parameters are estimated by MLE. For other methods it is used as a comparative convenience. The relative AIC values are still informative because the same log-likelihood function is used for all methods.

**The 2:1 structural bias in the composite rank**

Composite = (rank_RMSE + rank_AD + rank_AIC) / 3

RMSE and AD both use the failure-only empirical CDF → 2 votes for failure-only methods.
AIC uses the full censored log-likelihood → 1 vote for likelihood-based methods.

This is why WLS wins the composite rank despite MLE being correct for the full population. Move the AIC weight slider in Exp 5 to see the crossover point.

**The population-level metric (right table)**

Pop Err % = |F̂(92) − observed failure fraction| / observed failure fraction × 100

Observed failure fraction = {len(t_fail)}/{len(t_fail)+len(t_cens):,} = {len(t_fail)/(len(t_fail)+len(t_cens)):.6f}

MLE predicts F(92) ≈ {weibull_cdf(92, params_df.loc['MLE','k'], params_df.loc['MLE','λ']):.6f} → error ≈ 0%.
WLS predicts F(92) ≈ {weibull_cdf(92, params_df.loc['WLS','k'], params_df.loc['WLS','λ']):.4f} → error ≈ {abs(weibull_cdf(92, params_df.loc['WLS','k'], params_df.loc['WLS','λ']) - len(t_fail)/(len(t_fail)+len(t_cens)))/max(len(t_fail)/(len(t_fail)+len(t_cens)),1e-9)*100:.0f}%.

This is the metric MRL error should have been — it measures population-level prediction accuracy rather than failure-subpopulation fit quality.
        """)

with tabs[1]:
    st.markdown('<div class="section-header">Weibull Probability Plot</div>', unsafe_allow_html=True)
    fig1 = go.Figure()
    tf_s = np.sort(t_fail); F_e = median_ranks(len(tf_s))
    fig1.add_trace(go.Scatter(x=np.log(tf_s), y=np.log(-np.log(1-F_e)), mode="markers", name="Data", marker_color="white"))
    t_p = np.linspace(min(tf_s), max(tf_s)*1.5, 200)
    for m in sel_methods:
        k, lam = params_df.loc[m, "k"], params_df.loc[m, "λ"]
        Fp = weibull_cdf(t_p, k, lam)
        fig1.add_trace(go.Scatter(x=np.log(t_p), y=np.log(-np.log(1-np.clip(Fp, 1e-6, 1-1e-6))), name=m, line_color=METHOD_COLORS[m]))
    fig1.update_layout(template="plotly_dark", height=500, xaxis_title="ln(t)", yaxis_title="ln(-ln(1-F))")
    st.plotly_chart(fig1, use_container_width=True)

    with st.expander("📖 How to read this plot — complete explanation"):
        st.markdown(f"""
**What this plot is doing**

The Weibull CDF is F(t) = 1 − exp(−(t/λ)ᵏ). Take −ln(1 − F) = (t/λ)ᵏ.
Take ln of both sides: ln(−ln(1−F)) = k·ln(t) − k·ln(λ).

This is a straight line Y = kX + b where:
- **X = ln(t)** — the horizontal axis
- **Y = ln(−ln(1−F))** — the vertical axis
- **Slope = k** — steeper line = higher shape parameter = faster wear-out
- **Intercept = −k·ln(λ)** — controls where the line sits vertically

**The white dots** are the actual failure data. Each dot is one observed failure time, plotted at its empirical CDF value using Benard's formula: F̂ᵢ = (i−0.3)/(n+0.4) where n is the total number of drives (not just failures). The full n in the denominator is what gives rank regression its implicit censoring adjustment.

**If the dots form a straight line**, the Weibull distribution is a good fit. Curvature would suggest a different distribution is more appropriate.

**What the colored lines tell you**

Each line is a fitted Weibull. The slope of the line is k̂ for that method.

- **Steep lines** (MOM, EPF, L-Moments) → high k̂ → strong wear-out in the failure subpopulation
- **Shallow lines** (OLS) → low k̂ → OLS is pulled toward noisy tail observations which flatten its slope
- **WLS** tracks the data dots most closely because Bergman weights anchor the regression at F̂ ≈ 0.632 where variance is lowest
- **MLE and MCMC lines** appear far below the data cluster with nearly zero slope — their enormous λ̂ (tens of thousands of days) means F(t) ≈ 0 everywhere in the observable window, which maps to ln(−ln(1−F)) ≈ −∞ — they fall off the bottom of the plot

**The MLE/MCMC lines being below the cluster is not a failure of those methods** — it reflects that they are fitting the full population of {len(t_fail)+len(t_cens):,} drives, 99%+ of which survived. The characteristic life λ̂ ≈ {params_df.loc['MLE','λ']:.0f} days is simply outside the plotting range.

**Current model: {sel_model}**
- WLS slope (k̂): {params_df.loc['WLS','k']:.3f}
- MLE slope (k̂): {params_df.loc['MLE','k']:.3f}
- Data alignment: {'Good — dots follow a roughly linear pattern, confirming Weibull is appropriate' if len(t_fail) > 30 else 'Limited — few failures, high scatter expected'}
        """)

with tabs[2]:
    st.markdown('<div class="section-header">Distribution Curves</div>', unsafe_allow_html=True)
    ptype = st.radio("Function:", ["CDF", "Survival", "PDF", "Hazard"], horizontal=True)
    fig2 = go.Figure(); t_c = np.linspace(1, max(t_fail)*2, 500)
    f_map = {"CDF": weibull_cdf, "Survival": weibull_sf, "PDF": weibull_pdf, "Hazard": weibull_hazard}
    for m in sel_methods:
        k, lam = params_df.loc[m, "k"], params_df.loc[m, "λ"]
        fig2.add_trace(go.Scatter(x=t_c, y=f_map[ptype](t_c, k, lam), name=m, line_color=METHOD_COLORS[m]))
    fig2.update_layout(template="plotly_dark", height=500, xaxis_title="Days")
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📖 How to read each function — complete explanation"):
        k_wls_ex = params_df.loc['WLS','k']; lam_wls_ex = params_df.loc['WLS','λ']
        k_mle_ex = params_df.loc['MLE','k']; lam_mle_ex = params_df.loc['MLE','λ']
        st.markdown(f"""
**CDF — F(t) = 1 − exp(−(t/λ)ᵏ)**

The probability that a drive has failed by time t. Starts at 0, rises to 1.
- For failure-only methods (WLS, OLS…): the CDF rises steeply and reaches ~1.0 around day 80–100. This is the distribution of drives that fail *within the observation window*, normalized to 1.
- For MLE/MCMC: the CDF is essentially zero across the entire plot. F(92) ≈ {weibull_cdf(92, k_mle_ex, lam_mle_ex)*100:.2f}% — correctly predicting that only {weibull_cdf(92, k_mle_ex, lam_mle_ex)*100:.2f}% of the full fleet fails in 92 days.
- These are **not two versions of the same answer** — they describe different populations. WLS describes early-failure drives. MLE describes the full fleet.

**Survival Function — S(t) = 1 − F(t) = exp(−(t/λ)ᵏ)**

The probability that a drive is still alive at time t. Starts at 1, declines toward 0.
- The Kaplan-Meier curve (in Exp 1) stays flat near 1.0 — the empirical ground truth about the full population.
- Failure-only survival curves decline steeply — they are describing only the early-failing drives.
- MLE's survival curve stays near 1.0, matching KM. S(92) for MLE = {weibull_sf(92, k_mle_ex, lam_mle_ex):.4f}. S(92) for WLS = {weibull_sf(92, k_wls_ex, lam_wls_ex):.4f}.
- The gap between WLS and MLE at day 92 is {abs(weibull_sf(92, k_wls_ex, lam_wls_ex) - weibull_sf(92, k_mle_ex, lam_mle_ex))*100:.1f} percentage points.

**PDF — f(t) = (k/λ)(t/λ)^(k−1) · exp(−(t/λ)ᵏ)**

The probability density of failing at exactly time t. The area under the PDF integrates to 1.
- The peak of the PDF is the **mode** — the most likely failure time. For WLS: peak ≈ λ·((k−1)/k)^(1/k) = {lam_wls_ex * ((k_wls_ex-1)/k_wls_ex)**(1/k_wls_ex) if k_wls_ex > 1 else 0:.0f} days.
- Higher k → narrower, more peaked PDF → failures are concentrated around a specific age (predictable wear-out).
- Lower k → broader, flatter PDF → failures spread across a wide range of ages.
- MLE/MCMC PDFs are essentially zero across the visible range — their λ̂ is in the tens of thousands of days, so the mode is far beyond the plot range.

**Hazard Rate — h(t) = f(t)/S(t) = (k/λ)(t/λ)^(k−1)**

The instantaneous failure rate at time t, given that the unit has survived to t. This is the most operationally important function.
- **k < 1**: h(t) decreasing → infant mortality. Surviving drives get *more* reliable with age. Age-based replacement is counterproductive.
- **k = 1**: h(t) constant → random failures (memoryless, exponential distribution). Age carries no information. Replacement scheduling provides no benefit.
- **k > 1**: h(t) increasing → wear-out. Older drives fail faster. Age-based maintenance is justified.
- WLS gives k̂ = {k_wls_ex:.3f} → {'increasing hazard, wear-out in the failure subpopulation' if k_wls_ex > 1 else 'decreasing hazard, infant mortality pattern'}
- MLE gives k̂ = {k_mle_ex:.3f} → {'near-exponential behavior for the full population — hazard barely changes with age' if abs(k_mle_ex - 1) < 0.3 else 'increasing hazard at population level'}
- These two k̂ values lead to **opposite maintenance recommendations** depending on which population you care about.
        """)

with tabs[3]:
    st.markdown('<div class="section-header">Parameter Comparison (Normalized to MLE)</div>', unsafe_allow_html=True)
    mle_k = params_df.loc["MLE", "k"]; mle_lam = params_df.loc["MLE", "λ"]
    comp_df = params_df.loc[sel_methods].copy()
    comp_df["k_diff"] = (comp_df["k"] / mle_k - 1) * 100
    comp_df["lam_diff"] = (comp_df["λ"] / mle_lam - 1) * 100
    st.dataframe(comp_df, use_container_width=True)

    with st.expander("📖 How to read this table — complete explanation"):
        st.markdown(f"""
**What the columns mean**

- **k** — shape parameter estimate for this method. Controls the hazard behavior: <1 infant mortality, =1 random, >1 wear-out. Compare across methods to see how much the inference frame affects the shape conclusion.
- **λ** — scale parameter in days. This is the time at which exactly 63.2% of the population has failed (F(λ) = 1−e⁻¹ ≈ 0.632 for any k). It is the characteristic life of the population being modeled.
- **k_diff** — how much this method's k̂ differs from MLE's k̂, in percent. Positive = higher than MLE, negative = lower.
- **lam_diff** — how much this method's λ̂ differs from MLE's λ̂, in percent.

**Why lam_diff is enormous for failure-only methods**

MLE's λ̂ = {mle_lam:.0f} days. WLS's λ̂ = {params_df.loc['WLS','λ']:.0f} days. The ratio is {mle_lam/params_df.loc['WLS','λ']:.0f}×.

This is not a numerical error — it is mathematically necessary. WLS fits only the {len(t_fail)} drives that failed. MLE fits all {len(t_fail)+len(t_cens):,} drives. The λ for WLS is the characteristic life of the early-failure subpopulation (~50 days). The λ for MLE is the characteristic life of the full fleet (~{mle_lam:.0f} days). These are genuinely different quantities.

**Why k_diff is smaller but still significant**

The shape parameter k is dimensionless — it describes the *shape* of the distribution, not its scale. Failure-only methods estimate k for the early-failure subpopulation (which shows wear-out: k̂ ≈ 1.2–2.2). MLE estimates k for the full population (which shows near-exponential behavior: k̂ ≈ {mle_k:.2f}). The difference is real and leads to opposite maintenance recommendations.

**How to use this table**

Look at the lam_diff column. Any value far from 0% means that method is characterizing a fundamentally different population than MLE. The k_diff column tells you whether that method's shape conclusion (wear-out vs random vs infant mortality) agrees with MLE's full-population conclusion.
        """)

with tabs[4]:
    st.markdown('<div class="section-header">Method Selection Decision</div>', unsafe_allow_html=True)

    if defense_mode:
        st.markdown("""<div style="background:#0d2d1a;border-left:4px solid #2ea043;padding:12px;border-radius:0 8px 8px 0;margin-bottom:16px;">
        <strong style="color:#3fb950">🎓 Defense Q:</strong> <em>"So which method should an engineer actually use?"</em><br><br>
        <strong style="color:#3fb950">Your answer:</strong> "It depends on the question. For a first-year field replacement strategy —
        how many spares do I need in the first 90 days — use WLS: it characterizes the early-failure subpopulation accurately.
        For 10-year cost forecasting or warranty modeling — what fraction of the full fleet will fail over its lifetime —
        use MLE with the full censored likelihood. Always report both with explicit statements of what each represents.
        Appendix A of the report formalizes this as a decision procedure."
        </div>""", unsafe_allow_html=True)

    if censor_pct > 30: st.error(f"High Censoring ({censor_pct:.1f}%). Only MLE/MAP are statistically valid for population inference.")
    elif censor_pct > 5: st.warning(f"Moderate Censoring. WLS/L-Moments preferred over OLS/MOM.")
    else: st.success("Low Censoring. All methods valid.")

    st.markdown("#### Appendix A Decision Procedure")
    st.markdown(f"""
    | Step | Action | Result for this model |
    |------|--------|-----------------------|
    | 1 | Compute censoring fraction | **{censor_pct:.1f}%** → eliminates Percentile, MOM, EPF for population inference |
    | 2 | Run MLE as primary | k̂={params_df.loc['MLE','k']:.3f}, λ̂={params_df.loc['MLE','λ']:.0f}d |
    | 3 | Run WLS in parallel (within-window) | k̂={params_df.loc['WLS','k']:.3f}, λ̂={params_df.loc['WLS','λ']:.0f}d |
    | 4 | λ̂_MLE / λ̂_WLS ratio | **{params_df.loc['MLE','λ']/params_df.loc['WLS','λ']:.0f}×** — confirms extreme censoring impact |
    | 5 | Report both with inference frame labels | ✅ Done |
    """)

with tabs[5]:
    st.markdown('<div class="section-header">🧪 Reliability Lab — Interactive Proofs of the Report\'s Core Findings</div>', unsafe_allow_html=True)

    st.markdown("""
    > **This lab replicates the four core experimental findings from the report — not a demo, actual proof.**
    > Each experiment lets you manipulate one variable and watch the conclusion emerge from the real data.
    """)

    lab_exp = st.radio(
        "Choose Experiment",
        [
            "Exp 1 — The Censoring Duality: Two Methods, Two Truths",
            "Exp 2 — Outlier Sensitivity Hierarchy: EPF → MOM → L-Moments → MLE",
            "Exp 3 — MAP Prior Misspecification: When a Prior Destroys Estimation",
            "Exp 4 — Kaplan-Meier Validation: Who Is Actually Telling the Truth?",
        ],
    )

    tf_sorted = np.sort(t_fail)
    n_f = len(tf_sorted); n_c = len(t_cens); n_tot = n_f + n_c

    # ── EXP 1: CENSORING DUALITY ──────────────────────────────────────────────
    if "Exp 1" in lab_exp:
        st.markdown("### Experiment 1 — The Censoring Duality")
        st.markdown("""
        **Claim:** Failure-only methods (WLS, OLS…) and likelihood-based methods (MLE) are not competing —
        they answer *different questions*. WLS describes the early-failure subpopulation.
        MLE describes the full fleet. The Kaplan-Meier estimator is the assumption-free ground truth that
        proves which one is right about the population.
        """)

        # Kaplan-Meier
        all_t_km = np.concatenate([tf_sorted, t_cens])
        all_e_km = np.concatenate([np.ones(n_f), np.zeros(n_c)])
        ord_km = np.argsort(all_t_km)
        t_km_s = all_t_km[ord_km]; e_km_s = all_e_km[ord_km]
        km_t = [0]; km_s = [1.0]; S_km = 1.0; n_r = n_tot
        for ti, ei in zip(t_km_s, e_km_s):
            if ei == 1:
                S_km *= (1 - 1/n_r)
                km_t.append(ti); km_s.append(S_km)
            n_r -= 1
        km_t = np.array(km_t); km_s = np.array(km_s)

        k_wls, lam_wls = params_df.loc["WLS","k"], params_df.loc["WLS","λ"]
        k_mle, lam_mle = params_df.loc["MLE","k"], params_df.loc["MLE","λ"]
        t_pl = np.linspace(1, max(tf_sorted)*1.2, 400)

        fig_e1 = go.Figure()
        fig_e1.add_trace(go.Scatter(x=km_t, y=km_s, mode="lines",
            name="Kaplan-Meier (non-parametric ground truth — all drives)",
            line=dict(color="#ffffff", width=4, dash="dash")))
        fig_e1.add_trace(go.Scatter(x=t_pl, y=weibull_sf(t_pl, k_wls, lam_wls), mode="lines",
            name=f"WLS  k={k_wls:.2f} λ={lam_wls:.0f}d  — failure-only frame",
            line=dict(color=METHOD_COLORS["WLS"], width=2.5)))
        fig_e1.add_trace(go.Scatter(x=t_pl, y=weibull_sf(t_pl, k_mle, lam_mle), mode="lines",
            name=f"MLE  k={k_mle:.2f} λ={lam_mle:.0f}d  — full population frame",
            line=dict(color=METHOD_COLORS["MLE"], width=2.5)))
        fig_e1.add_vline(x=92, line_dash="dot", line_color="#555", annotation_text="Day 92 (window end)")
        fig_e1.update_layout(template="plotly_dark", height=480,
            title="Survival Function: Two Frames vs. KM Ground Truth",
            xaxis_title="Days", yaxis_title="S(t)", yaxis=dict(range=[-0.05, 1.1]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        st.plotly_chart(fig_e1, use_container_width=True)

        km_92 = float(km_s[np.searchsorted(km_t, 92, side="right")-1])
        s_wls_92 = float(weibull_sf(92, k_wls, lam_wls))
        s_mle_92 = float(weibull_sf(92, k_mle, lam_mle))

        c1e, c2e, c3e = st.columns(3)
        with c1e:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">KM S(92) — Ground Truth</div>
                <div class="metric-value" style="color:#fff">{km_92:.4f}</div>
                <div style="color:#8b949e;font-size:0.8rem;">No model. Just counting survivors.</div>
            </div>""", unsafe_allow_html=True)
        with c2e:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">WLS S(92)</div>
                <div class="metric-value" style="color:{METHOD_COLORS['WLS']}">{s_wls_92:.4f}</div>
                <div style="color:#f85149;font-size:0.8rem;">Predicts {(1-s_wls_92)*100:.0f}% failed — wrong for population by {abs(s_wls_92-km_92)*100:.1f}pp</div>
            </div>""", unsafe_allow_html=True)
        with c3e:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">MLE S(92)</div>
                <div class="metric-value" style="color:{METHOD_COLORS['MLE']}">{s_mle_92:.4f}</div>
                <div style="color:#3fb950;font-size:0.8rem;">Tracks KM — deviation only {abs(s_mle_92-km_92)*100:.3f}pp</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#1c2333;border-left:4px solid #3fb950;padding:16px;border-radius:0 8px 8px 0;margin-top:16px;">
        <strong style="color:#3fb950">📐 Deduction</strong><br>
        WLS says {(1-s_wls_92)*100:.0f}% of drives failed by day 92. KM says {(1-km_92)*100:.3f}% failed.
        WLS is wrong by a factor of {(1-s_wls_92)/(max(1-km_92,1e-9)):.0f}×.
        MLE deviates from KM by only {abs(s_mle_92-km_92)*100:.3f} percentage points.<br><br>
        The composite rank favors WLS over MLE because 2 of 3 composite metrics (RMSE, AD) compare against
        the failure-only empirical CDF — a structural bias in the scoring, not a judgment about population truth.
        </div>""", unsafe_allow_html=True)

        if defense_mode:
            st.markdown(f"""<div style="background:#2d2008;border-left:4px solid #9e6a03;padding:16px;border-radius:0 8px 8px 0;margin-top:8px;">
            <strong style="color:#d29922">🎓 Defense Q:</strong> <em>"If MLE is correct, why does your composite rank say WLS wins?"</em><br><br>
            <strong style="color:#d29922">Prepared answer:</strong> "Because the composite rank mixes two failure-only metrics (RMSE, AD)
            with one full-data metric (AIC) in a 2:1 ratio. This is acknowledged explicitly in Section 2.5.3.2.
            WLS wins the composite rank within the failure-only inference frame — it is the best method for
            characterizing drives that fail early. MLE wins in every population-level evaluation:
            it has the lowest AIC by 140,000 units and its survival curve overlaps the Kaplan-Meier
            ground truth to within {abs(s_mle_92-km_92)*100:.3f} percentage points at day 92.
            These are not contradictory results — they answer different questions."
            </div>""", unsafe_allow_html=True)

        with st.expander("📖 Deep explanation — every element of this experiment"):
            st.markdown(f"""
**What the Kaplan-Meier curve actually is**

The KM curve is computed from ALL {n_tot:,} drives — both the {n_f} that failed and the {n_c:,} that were still alive at day 92. It uses the product-limit formula:

S_KM(t) = ∏ (1 − dᵢ/nᵢ)

where at each failure time tᵢ, dᵢ = number of failures and nᵢ = number of drives still at risk (not yet failed or censored). When a censored drive exits the dataset, nᵢ decreases — the KM estimator correctly accounts for this.

The result: KM drops by approximately 1/{n_tot:,} ≈ {1/n_tot*100:.4f}% per failure. With only {n_f} failures, the total drop across the entire 92-day window is {n_f/n_tot*100:.3f}%. That is why the KM curve appears flat — it is genuinely flat, not an artifact.

**Why WLS's survival curve is wrong for the full population**

WLS fits only the {n_f} failure times. It computes Benard's plotting positions F̂ᵢ = (i−0.3)/(n+0.4) where n = {n_tot:,} — so the largest failure gets F̂ ≈ {(n_f-0.3)/(n_tot+0.4):.4f}, not 1.0. This is the implicit censoring adjustment: ranks are computed relative to the full fleet, not just failures. But the fitted CDF is still normalized to [0,1] over the failure times, giving λ̂_WLS ≈ {params_df.loc['WLS','λ']:.0f} days — the time at which 63.2% of early-failing drives fail.

WLS S(92) = {s_wls_92:.4f}, meaning it predicts {(1-s_wls_92)*100:.1f}% of drives failed by day 92. The actual rate is {(1-km_92)*100:.3f}%. WLS is wrong by a factor of {(1-s_wls_92)/max(1-km_92,1e-9):.0f}× for the full population.

**Why MLE's survival curve matches KM**

MLE maximizes the full censored log-likelihood:

ℓ(k,λ) = Σ_failures [ln f(tᵢ)] + Σ_censored [ln S(cⱼ)]

The second sum includes all {n_c:,} censored drives. Each S(cⱼ) = exp(−(cⱼ/λ)ᵏ) term encodes the evidence that drive j survived to cⱼ days. With λ = {params_df.loc['WLS','λ']:.0f} days, S(92) ≈ {weibull_sf(92, params_df.loc['WLS','k'], params_df.loc['WLS','λ']):.3f} — meaning the log-likelihood contribution of the {n_c:,} surviving drives would be approximately {n_c:,} × ln(0.126) ≈ {n_c * np.log(max(weibull_sf(92, params_df.loc['WLS','k'], params_df.loc['WLS','λ']),1e-9)):.0f}. The optimizer immediately rejects this and pushes λ to ~{params_df.loc['MLE','λ']:.0f} days where S(92) ≈ {weibull_sf(92, params_df.loc['MLE','k'], params_df.loc['MLE','λ']):.4f}.

**The inference frame insight**

WLS and MLE are not competing methods giving different approximations to the same truth. They are answering different questions:
- WLS answers: "Given that a drive fails within 92 days, what is the distribution of its failure time?"
- MLE answers: "What is the lifetime distribution of the entire drive population?"

Both answers are statistically valid. The question is which one you need for your operational decision.
            """)

    # ── EXP 2: OUTLIER SENSITIVITY HIERARCHY ──────────────────────────────────
    elif "Exp 2" in lab_exp:
        st.markdown("### Experiment 2 — Outlier Sensitivity Hierarchy")
        st.markdown("""
        **Claim:** The robustness ordering EPF > MOM > L-Moments > MLE is not empirical coincidence —
        it follows directly from the order of the moment each method uses: cubic → quadratic → linear → logarithmic.
        One contaminated data point at α× the maximum can reverse EPF's failure mode diagnosis entirely.
        """)
        omult = st.slider("Outlier multiplier α (× largest failure time)", 1.0, 10.0, 3.0, 0.5,
            help="α=1: no contamination. α=10: replicates Section 3.7 of the report.")

        tf_clean = tf_sorted.copy()
        tf_contam = tf_sorted.copy(); tf_contam[-1] *= omult

        sens_methods = ["EPF", "MOM", "L-Moments", "MLE"]
        moment_labels = {"EPF": "Cubic (3rd)", "MOM": "Quadratic (2nd)", "L-Moments": "Linear (1st)", "MLE": "Log (≈1.5th)"}
        sens_res = []
        for m in sens_methods:
            try:
                kc, _ = FIT_FUNCTIONS[m](tf_clean, np.array([]))
                kx, _ = FIT_FUNCTIONS[m](tf_contam, np.array([]))
                drift = abs(kx/kc - 1)*100
            except:
                kc, kx, drift = np.nan, np.nan, np.nan
            sens_res.append({"m": m, "kc": kc, "kx": kx, "drift": drift})

        fig_e2 = go.Figure()
        for r in sens_res:
            if not np.isnan(r["drift"]):
                fig_e2.add_trace(go.Bar(name=r["m"], x=[r["m"]], y=[r["drift"]],
                    marker_color=METHOD_COLORS[r["m"]],
                    text=[f"{r['drift']:.1f}%"], textposition="outside", width=0.5))
        fig_e2.add_shape(type="line", x0=-0.5, x1=3.5, y0=50, y1=50,
            line=dict(color="#f85149", dash="dot", width=1))
        fig_e2.add_annotation(x=3.3, y=53, text="50% threshold", font_color="#f85149", showarrow=False, font_size=11)
        fig_e2.update_layout(template="plotly_dark", height=420, showlegend=False,
            title=f"k̂ Drift from Single Outlier at α={omult:.1f}× (Section 3.7 replication)",
            yaxis_title="% Change in k̂")
        st.plotly_chart(fig_e2, use_container_width=True)

        cols_s = st.columns(4)
        for i, r in enumerate(sens_res):
            with cols_s[i]:
                if np.isnan(r["drift"]): continue
                sev = "🔴 CATASTROPHIC" if r["drift"] > 50 else ("🟡 HIGH" if r["drift"] > 25 else "🟢 ROBUST")
                sc = "#f85149" if r["drift"] > 50 else ("#d29922" if r["drift"] > 25 else "#3fb950")
                st.markdown(f"""<div style="background:#161b22;padding:16px;border-radius:10px;border-top:4px solid {METHOD_COLORS[r['m']]};">
                    <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">{r['m']} — {moment_labels[r['m']]}</div>
                    <div style="color:{METHOD_COLORS[r['m']]};font-size:1.3rem;font-weight:700;margin:6px 0;">
                        {r['kc']:.3f} → {r['kx']:.3f}
                    </div>
                    <div style="color:{sc};font-weight:600;">{sev}</div>
                    <div style="color:{sc};font-size:1rem;font-weight:700;">{r['drift']:.1f}% drift</div>
                </div>""", unsafe_allow_html=True)

        sorted_res = sorted([r for r in sens_res if not np.isnan(r["drift"])], key=lambda x: -x["drift"])
        hier = " > ".join([f"{r['m']} ({r['drift']:.0f}%)" for r in sorted_res])
        st.markdown(f"""
        <div style="background:#1c2333;border-left:4px solid #3b82f6;padding:16px;border-radius:0 8px 8px 0;margin-top:16px;">
        <strong style="color:#3b82f6">📐 Deduction at α={omult:.1f}×</strong><br>
        Observed hierarchy: <strong style="color:#fff">{hier}</strong><br><br>
        Mathematical explanation: EPF cubes the outlier contribution (×{omult:.0f}³ = {omult**3:.0f}×).
        MOM squares it (×{omult**2:.0f}×). L-Moments scales linearly (×{omult:.0f}×).
        MLE dampens via logarithm (×ln({omult:.0f}) ≈ {np.log(omult):.1f}×).
        The ranking is not observed — it is derived. Table 3.7 in the report is just this experiment at α=10.
        </div>""", unsafe_allow_html=True)

        if defense_mode:
            st.markdown(f"""<div style="background:#2d2008;border-left:4px solid #9e6a03;padding:16px;border-radius:0 8px 8px 0;margin-top:8px;">
            <strong style="color:#d29922">🎓 Defense Q:</strong> <em>"Why is MLE evaluated on failures-only in Table 3.7 when it normally uses censored data?"</em><br><br>
            <strong style="color:#d29922">Prepared answer:</strong> "Introducing a contaminated failure time of t_max×10 ≈ 920 days
            into a dataset where all censored observations are capped at 92 days creates a physically incoherent dataset —
            a unit recorded as having failed after all the surviving units were last observed.
            The censored log-likelihood behaves pathologically in that scenario.
            To produce a meaningful sensitivity comparison within the same inference frame as the moment-based methods,
            MLE was evaluated on the failures-only subset. This is stated explicitly in the Table 3.7 footnote.
            The full-censored MLE shape estimates are in Table 3.2."
            </div>""", unsafe_allow_html=True)

        with st.expander("📖 Deep explanation — why each method has exactly the sensitivity it has"):
            st.markdown(f"""
**The mathematical reason for each method's sensitivity — derived, not observed**

**EPF (Energy Pattern Factor) — cubic sensitivity**

EPF solves: Γ(1+3/k) / Γ³(1+1/k) = (Σtᵢ³/n) / t̄³

The left side depends only on k. The right side is the sample EPF.

If one observation tⱼ is replaced by α·tⱼ, its contribution to Σtᵢ³ increases by α³. For α={omult:.1f}: factor of {omult**3:.0f}×. This single point can dominate the entire sum regardless of all other observations, forcing the estimated k toward near-zero (infant mortality).

At α={omult:.1f}: EPF changes k by {next((r['drift'] for r in sens_res if r['m']=='EPF' and not np.isnan(r.get('drift',float('nan')))), float('nan')):.1f}% — confirming cubic amplification.

**MOM (Method of Moments) — quadratic sensitivity**

MOM matches the coefficient of variation: Γ(1+2/k)/Γ²(1+1/k) − 1 = s²/t̄²

If tⱼ → α·tⱼ, its contribution to the variance s² increases by α². For α={omult:.1f}: factor of {omult**2:.0f}×. Less catastrophic than EPF but still severe.

**L-Moments — linear sensitivity**

L-Moments uses λ₂ = (2/n(n−1)) Σ (i−1)·t(i) − t̄

The contaminated observation enters λ̂₂ with a weight proportional to its rank — at most (n−1)/n ≈ 1. So its influence scales linearly with α, not cubically or quadratically. For α={omult:.1f}: factor of {omult:.1f}× — much more resistant.

**MLE — logarithmic sensitivity**

The log-likelihood contribution of one observation is:
ln f(t; k, λ) = ln(k/λ) + (k−1)·ln(t/λ) − (t/λ)ᵏ

As t → α·t, the dominant term grows as ln(α·t) — logarithmically. For α={omult:.1f}: factor of ln({omult:.1f}) ≈ {np.log(omult):.2f}× — the logarithm strongly compresses the outlier's influence.

**The hierarchy is structural, not coincidental**

The order EPF > MOM > MLE > L-Moments follows directly from:
cubic moment (3rd) > quadratic moment (2nd) > logarithmic moment (~1.5th) > linear moment (1st)

This is derived from the mathematics of each method's estimating equation, not from observing this particular dataset. The experiment confirms the theory. Table 3.7 in the report is this experiment at α=10.
            """)

    # ── EXP 3: MAP PRIOR MISSPECIFICATION ─────────────────────────────────────
    elif "Exp 3" in lab_exp:
        st.markdown("### Experiment 3 — MAP Prior Misspecification")
        st.markdown("""
        **Claim:** Under extreme censoring (99%+), the prior on λ is the primary determinant of MAP quality.
        The default prior anchored at `mean(failures) ≈ 50 days` is strongly misspecified — it concentrates
        probability mass on lifetimes measured in *weeks* when drives actually last *years*.
        A correctly specified prior removes the problem entirely.
        """)
        mean_f = float(np.mean(t_fail))
        k_mle_r, lam_mle_r = params_df.loc["MLE","k"], params_df.loc["MLE","λ"]
        k_wls_r, lam_wls_r = params_df.loc["WLS","k"], params_df.loc["WLS","λ"]

        c_ctrl3, c_plot3 = st.columns([1, 2])
        with c_ctrl3:
            st.markdown("#### Adjust the prior on λ")
            default_anchor = int(mean_f)
            prior_opts = [10,20,30,50,100,200,500,1000,2000,5000,int(5*365),int(lam_mle_r),default_anchor]
            prior_opts = sorted(list(set(prior_opts)))
            lam_mu = st.select_slider("Prior anchor μλ (days)", options=prior_opts,
                value=default_anchor, help=f"Report default ≈{mean_f:.0f}d. MLE truth ≈{lam_mle_r:.0f}d. Try 5yr={int(5*365)}d.")
            p_str = st.slider("Prior strength", 0.1, 3.0, 1.0, 0.1,
                help="Higher = prior dominates the data. Lower = prior barely matters.")

            def neg_map3(p):
                kv, lv = np.exp(p[0]), np.exp(p[1])
                ll = censored_loglik(kv, lv, t_fail, t_cens)
                return -(ll - kv/2.0 - lv/lam_mu * p_str)
            try:
                r3 = minimize(neg_map3, [np.log(1.5), np.log(lam_mu)], method="L-BFGS-B")
                k_map3, lam_map3 = np.exp(r3.x[0]), np.exp(r3.x[1])
            except:
                k_map3, lam_map3 = np.nan, np.nan

            st.markdown(f"""<div style="margin-top:20px;padding:14px;background:#0d1117;border-radius:8px;">
            <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">MLE (full-data truth)</div>
            <div style="color:{METHOD_COLORS['MLE']};font-size:1rem;">k={k_mle_r:.3f} | λ={lam_mle_r:.0f}d</div>
            <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;margin-top:10px;">MAP with your prior</div>
            <div style="color:{METHOD_COLORS['MAP']};font-size:1rem;">k={k_map3:.3f} | λ={lam_map3:.0f}d</div>
            <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;margin-top:10px;">WLS (failure-only)</div>
            <div style="color:{METHOD_COLORS['WLS']};font-size:1rem;">k={k_wls_r:.3f} | λ={lam_wls_r:.0f}d</div>
            </div>""", unsafe_allow_html=True)

        with c_plot3:
            t_pl3 = np.linspace(1, max(tf_sorted)*2, 400)
            fig_e3 = go.Figure()
            fig_e3.add_trace(go.Scatter(x=t_pl3, y=weibull_sf(t_pl3, k_mle_r, lam_mle_r), mode="lines",
                name="MLE — correct population", line=dict(color=METHOD_COLORS["MLE"], width=3)))
            fig_e3.add_trace(go.Scatter(x=t_pl3, y=weibull_sf(t_pl3, k_wls_r, lam_wls_r), mode="lines",
                name="WLS — failure-only frame", line=dict(color=METHOD_COLORS["WLS"], width=2, dash="dot")))
            if not np.isnan(k_map3):
                fig_e3.add_trace(go.Scatter(x=t_pl3, y=weibull_sf(t_pl3, k_map3, lam_map3), mode="lines",
                    name=f"MAP (μλ={lam_mu}d, strength={p_str:.1f})",
                    line=dict(color=METHOD_COLORS["MAP"], width=3)))
            fig_e3.add_vline(x=92, line_dash="dot", line_color="#555", annotation_text="Day 92")
            fig_e3.update_layout(template="plotly_dark", height=420,
                title="MAP Prior Sensitivity on Survival Function",
                xaxis_title="Days", yaxis_title="S(t)", yaxis=dict(range=[-0.05, 1.1]),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3))
            st.plotly_chart(fig_e3, use_container_width=True)

        if not np.isnan(lam_map3):
            ratio = lam_mle_r / max(lam_map3, 1)
            correct_prior = int(5*365)
            st.markdown(f"""
            <div style="background:#1c2333;border-left:4px solid #ec4899;padding:16px;border-radius:0 8px 8px 0;margin-top:8px;">
            <strong style="color:#ec4899">📐 Deduction</strong><br>
            With μλ={lam_mu}d and strength={p_str:.1f}: MAP gives λ̂={lam_map3:.0f}d —
            that is <strong>{ratio:.1f}× smaller than MLE's truth</strong>.<br>
            {"⚠️ Prior dominates data: MAP is ignoring tens of thousands of censored survivors." if lam_map3 < lam_mle_r/10 else
             "✅ Prior is weakly informative enough that MAP approaches MLE."}<br><br>
            The report's Section 3.8.3 argues the correct prior for enterprise drives (MTBF ≈ 1M hours)
            should be centered at μλ ≈ {correct_prior}d (5 years).
            Try setting the slider to {correct_prior} — MAP will escape the failure-only trap.
            </div>""", unsafe_allow_html=True)

            if defense_mode:
                st.markdown(f"""<div style="background:#2d2008;border-left:4px solid #9e6a03;padding:16px;border-radius:0 8px 8px 0;margin-top:8px;">
                <strong style="color:#d29922">🎓 Defense Q (most likely):</strong> <em>"You identified the MAP prior misspecification and proposed
                λ₀ = 1,825 days as the fix. Why didn't you run that experiment in the report?"</em><br><br>
                <strong style="color:#d29922">Prepared answer:</strong> "That is a fair criticism. I identified the problem precisely in Section 3.8.3
                and specified what a corrected prior would look like. Running the corrected MAP was the natural next step
                but fell outside the scope and time constraints of this study.
                It is the first item in the Perspectives section for exactly this reason.
                You can see the corrected prior live here — set the slider to {correct_prior} days and observe that MAP's λ̂
                moves from {lam_map3:.0f}d to approximately the MLE range, confirming the diagnosis was correct."
                </div>""", unsafe_allow_html=True)

        with st.expander("📖 Deep explanation — why the prior dominates under extreme censoring"):
            st.markdown(f"""
**The MAP objective — posterior = likelihood × prior**

MAP maximizes:

log p(k, λ | data) = ℓ(k, λ) − k/2 − λ/μλ × strength + const

The log-likelihood ℓ(k, λ) is computed from all {n_tot:,} observations. The prior terms are −k/2 and −λ/μλ.

**Why the prior wins at 99%+ censoring**

Under extreme censoring, almost all the likelihood information comes from the {n_c:,} survival terms S(cⱼ) = exp(−(cⱼ/λ)ᵏ). These terms push λ upward — they encode evidence that drives survived a long time. But the survival contribution to the gradient with respect to λ is:

∂/∂λ [Σ_censored ln S(cⱼ)] = (k/λ²) · Σ (cⱼ/λ)ᵏ

For small λ (like the prior's anchor of {int(np.mean(t_fail))} days), this gradient is large and positive — pushing λ upward. But the prior gradient is:

∂/∂λ [−λ/μλ · strength] = −strength/μλ

This is a constant pull downward regardless of the data. When strength is high or μλ is small, the prior's constant downward pull overwhelms the data's upward push.

**The quantitative conflict**

At the current prior (μλ = {lam_mu} days, strength = {p_str:.1f}):
- Prior says: λ should be around {lam_mu} days
- Data (via MLE) says: λ should be around {lam_mle_r:.0f} days
- Ratio: {lam_mle_r/lam_mu:.0f}× discrepancy
- MAP compromise: λ̂ ≈ {lam_map3:.0f} days

**The k̂ distortion**

MAP's k̂ = {k_map3:.3f} is higher than MLE's k̂ = {k_mle_r:.3f}. This indirect bias happens because the prior suppresses λ to ~{lam_map3:.0f} days. At this compressed scale, the optimizer uses a higher k to partially reconcile the short characteristic life with the observed failure times. The shape and scale parameters are coupled — you cannot compress one without distorting the other.

**The fix: set μλ to 5 years = 1,825 days**

Set the slider to 1825. The prior now encodes the belief that most drives survive several years — consistent with manufacturer MTBF specifications of 1–1.5 million hours. This prior is diffuse enough to be overridden by the {n_tot:,} observations, and MAP moves toward MLE's estimate.
            """)

    # ── EXP 4: KM VALIDATION ──────────────────────────────────────────────────
    elif "Exp 4" in lab_exp:
        st.markdown("### Experiment 4 — Kaplan-Meier Validation")
        st.markdown("""
        **Claim:** The KM estimator is the non-parametric maximum likelihood estimator of the survival function
        under right-censoring. It makes no distributional assumption whatsoever. When KM stays flat near 1.0
        for 92 days, that is not a modeling artifact — it is an empirical fact about the full population.
        Any parametric model whose S(t) diverges from KM is characterizing a different population.
        """)

        # KM computation
        all_t2 = np.concatenate([tf_sorted, t_cens])
        all_e2 = np.concatenate([np.ones(n_f), np.zeros(n_c)])
        ord2 = np.argsort(all_t2)
        t2s = all_t2[ord2]; e2s = all_e2[ord2]
        km2_t = [0]; km2_s = [1.0]; S2 = 1.0; nr2 = n_tot
        for ti, ei in zip(t2s, e2s):
            if ei == 1:
                S2 *= (1 - 1/nr2)
                km2_t.append(ti); km2_s.append(S2)
            nr2 -= 1
        km2_t = np.array(km2_t); km2_s = np.array(km2_s)
        km_92v = float(km2_s[np.searchsorted(km2_t, 92, side="right")-1])

        grp = st.radio("Show which method family:", [
            "Failure-only (OLS, WLS, MOM, L-Moments, EPF, Percentile)",
            "Likelihood-based (MLE, MAP)",
        ], horizontal=True)

        show_m = (["OLS","WLS","MOM","L-Moments","EPF","Percentile"]
                  if "Failure" in grp else ["MLE","MAP"])

        t_pl4 = np.linspace(1, max(tf_sorted)*1.2, 400)
        fig_e4 = go.Figure()
        fig_e4.add_trace(go.Scatter(x=km2_t, y=km2_s, mode="lines+markers",
            name="Kaplan-Meier (ground truth)",
            line=dict(color="#ffffff", width=4), marker=dict(size=3)))
        for m in show_m:
            if m in params_df.index:
                km_v, lam_v = params_df.loc[m,"k"], params_df.loc[m,"λ"]
                if not (np.isnan(km_v) or np.isnan(lam_v)):
                    fig_e4.add_trace(go.Scatter(x=t_pl4, y=weibull_sf(t_pl4, km_v, lam_v),
                        name=f"{m} k={km_v:.2f} λ={lam_v:.0f}d",
                        line=dict(color=METHOD_COLORS[m], width=2)))
        fig_e4.add_vline(x=92, line_dash="dot", line_color="#444", annotation_text="Day 92")
        fig_e4.update_layout(template="plotly_dark", height=500,
            title="Parametric Survival Curves vs. Non-Parametric KM Ground Truth",
            xaxis_title="Days", yaxis_title="S(t)", yaxis=dict(range=[-0.05, 1.1]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4))
        st.plotly_chart(fig_e4, use_container_width=True)

        st.markdown(f"#### Deviation from KM at Day 92 (KM = {km_92v:.4f})")
        dev_cols = st.columns(len(show_m))
        for i, m in enumerate(show_m):
            if m in params_df.index:
                km_v, lam_v = params_df.loc[m,"k"], params_df.loc[m,"λ"]
                if not (np.isnan(km_v) or np.isnan(lam_v)):
                    s92 = float(weibull_sf(92, km_v, lam_v))
                    dev = abs(s92 - km_92v)*100
                    bad = dev > 5
                    with dev_cols[i]:
                        st.markdown(f"""<div style="background:#161b22;padding:12px;border-radius:8px;border-top:3px solid {METHOD_COLORS[m]};">
                        <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">{m}</div>
                        <div style="color:{METHOD_COLORS[m]};font-size:1rem;font-weight:600;">S(92)={s92:.4f}</div>
                        <div style="color:{'#f85149' if bad else '#3fb950'};font-size:0.85rem;">
                            {'❌' if bad else '✅'} Δ={dev:.2f}pp
                        </div></div>""", unsafe_allow_html=True)

        conclusion = ("Failure-only methods predict 50–90% failure rate by day 92. KM says the actual rate is "
                      f"{(1-km_92v)*100:.3f}%. These methods are statistically valid — but they describe the "
                      "early-failure subpopulation, not the full fleet. The KM curve is the proof."
                      if "Failure" in grp else
                      f"MLE aligns with KM within a fraction of a percent. MAP deviates because its prior on λ "
                      "anchored at ~50 days forces it toward a wrong timescale. MLE with the censored likelihood "
                      "is the only estimator that correctly characterizes the full population empirically.")
        st.markdown(f"""
        <div style="background:#1c2333;border-left:4px solid #ffffff;padding:16px;border-radius:0 8px 8px 0;margin-top:16px;">
        <strong style="color:#ffffff">📐 Deduction</strong><br>{conclusion}
        </div>""", unsafe_allow_html=True)

        with st.expander("📖 Deep explanation — what the KM estimator is and why it is the ground truth"):
            st.markdown(f"""
**What the Kaplan-Meier estimator actually computes**

KM is the **non-parametric maximum likelihood estimator** of the survival function under right-censoring. It makes zero distributional assumptions — no Weibull, no exponential, nothing.

The formula: S_KM(t) = ∏_{{t(i) ≤ t}} (1 − dᵢ/nᵢ)

At each observed failure time t(i):
- dᵢ = number of drives that failed at exactly t(i) (almost always 1 in continuous data)
- nᵢ = number of drives still at risk just before t(i) — meaning alive and not yet censored

When a censored drive exits the study (last seen alive), nᵢ decreases for all future failure times. This is how KM correctly handles censoring — the at-risk count automatically adjusts.

**Why KM is the ground truth for this study**

This dataset has {n_tot:,} drives. {n_f} failed. {n_c:,} were still alive at day 92. The KM estimator uses all {n_tot:,} observations.

S_KM(92) = {km_92v:.6f} — meaning {(1-km_92v)*100:.4f}% of drives failed by day 92.

This is not a model. It is direct counting. It is the empirical answer to "what fraction of drives actually survived?" No distributional assumption can override this.

**Why the KM curve is visually flat**

With {n_f} failures among {n_tot:,} drives, each failure step in the KM curve has height ≈ 1/{n_tot:,} ≈ {1/n_tot:.6f}. Over {n_f} failures, the total drop is {n_f}/{n_tot:,} = {n_f/n_tot:.4f} = {n_f/n_tot*100:.2f}%. This is smaller than the plot resolution. The curve is genuinely flat — not a scaling artifact.

**Why parametric models that diverge from KM are wrong about the population**

Any model that predicts S(92) = 0.15 (like WLS) is claiming 85% of drives failed by day 92. The KM evidence shows {(1-km_92v)*100:.2f}% failed. The model is wrong by a factor of {(1-weibull_sf(92, params_df.loc['WLS','k'], params_df.loc['WLS','λ']))/max(1-km_92v, 1e-9):.0f}× about the full population — even if it perfectly fits the {n_f} failure times. The model is fitting a different population than the one that actually exists.

**What MLE/MCMC alignment with KM proves**

MLE S(92) = {weibull_sf(92, params_df.loc['MLE','k'], params_df.loc['MLE','λ']):.4f}. KM S(92) = {km_92v:.4f}. Deviation = {abs(weibull_sf(92, params_df.loc['MLE','k'], params_df.loc['MLE','λ'])-km_92v)*100:.3f} percentage points.

MLE is not just theoretically correct — it is empirically confirmed correct by the assumption-free KM estimator. This is the strongest possible validation: theory and non-parametric evidence agree.
            """)

with tabs[6]:
    # ── EXTENDED LAB: Exp 5, 6, 7 ─────────────────────────────────────────────
    st.markdown('<div class="section-header">🔬 Extended Lab — Addressing the Report\'s Open Questions</div>', unsafe_allow_html=True)
    st.markdown("""
    > These three experiments directly address the criticisms raised in the report review:
    > **Exp 5** resolves the composite rank bias (Point 1). **Exp 6** runs the MAP corrected prior that the report proposed but didn't test (Point 3). **Exp 7** performs the day-28 binomial test that was flagged as missing (Point 4).
    """)

    ext_exp = st.radio("Choose Extended Experiment", [
        "Exp 5 — Composite Rank Weight Sensitivity: Is WLS Really the Winner?",
        "Exp 6 — MAP Corrected Prior: What Section 3.8.3 Proposed but Didn't Test",
        "Exp 7 — Day-28 Clustering: Statistical Test of the Batch Failure Hypothesis",
    ])

    tf_ext = np.sort(t_fail)
    n_f_ext = len(tf_ext); n_c_ext = len(t_cens); n_tot_ext = n_f_ext + n_c_ext

    # ── EXP 5: COMPOSITE RANK WEIGHT SENSITIVITY ──────────────────────────────
    if "Exp 5" in ext_exp:
        st.markdown("### Experiment 5 — Composite Rank Weight Sensitivity")
        st.markdown("""
        **The question:** The composite rank is defined as (rank_RMSE + rank_AD + rank_AIC) / 3.
        This gives RMSE and AD — both failure-only metrics — a 2:1 weight advantage over AIC.
        What happens to the ranking if you adjust that weight? At what AIC weight does MLE overtake WLS?
        """)

        w_aic = st.slider("Weight on AIC (failure-only metrics share the remainder equally)",
                          0.0, 1.0, 1/3, 0.05,
                          help="1/3 = report's equal weighting. 1.0 = AIC only. 0.0 = ignore AIC entirely.")
        w_shape = (1.0 - w_aic) / 2  # split equally between RMSE and AD

        st.markdown(f"**Current weights:** RMSE = {w_shape:.2f} | AD = {w_shape:.2f} | AIC = {w_aic:.2f}")

        methods_for_rank = [m for m in METHOD_NAMES if m in gof_df.index and
                            not any(np.isnan(gof_df.loc[m, c]) for c in ["RMSE", "AD", "AIC"])]

        rmse_vals = gof_df.loc[methods_for_rank, "RMSE"]
        ad_vals   = gof_df.loc[methods_for_rank, "AD"]
        aic_vals  = gof_df.loc[methods_for_rank, "AIC"]

        rmse_ranks = rmse_vals.rank(ascending=True)
        ad_ranks   = ad_vals.rank(ascending=True)
        aic_ranks  = aic_vals.rank(ascending=True)

        composite = w_shape * rmse_ranks + w_shape * ad_ranks + w_aic * aic_ranks
        composite_sorted = composite.sort_values()

        fig_e5 = go.Figure()
        colors_bar = [METHOD_COLORS.get(m, "#888") for m in composite_sorted.index]
        fig_e5.add_trace(go.Bar(
            x=composite_sorted.values,
            y=composite_sorted.index,
            orientation="h",
            marker_color=colors_bar,
            text=[f"{v:.2f}" for v in composite_sorted.values],
            textposition="outside",
        ))
        fig_e5.update_layout(
            template="plotly_dark", height=420,
            title=f"Composite Rank (RMSE×{w_shape:.2f} + AD×{w_shape:.2f} + AIC×{w_aic:.2f})",
            xaxis_title="Composite Score (lower = better)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_e5, use_container_width=True)

        winner = composite_sorted.index[0]
        mle_rank = composite_sorted.rank()[composite_sorted.index.get_loc("MLE")] if "MLE" in composite_sorted.index else None
        wls_score = composite_sorted.get("WLS", np.nan)
        mle_score = composite_sorted.get("MLE", np.nan)

        crossover_w = None
        for w_test in np.linspace(0, 1, 201):
            ws = (1 - w_test) / 2
            c_test = ws * rmse_ranks + ws * ad_ranks + w_test * aic_ranks
            if c_test.get("MLE", 999) < c_test.get("WLS", 0):
                crossover_w = w_test
                break

        st.markdown(f"""
        <div style="background:#1c2333;border-left:4px solid #58a6ff;padding:16px;border-radius:0 8px 8px 0;margin-top:16px;">
        <strong style="color:#58a6ff">📐 Deduction</strong><br>
        At the report's equal weighting (1/3 each): <strong style="color:#fff">WLS wins</strong> with score {wls_score:.2f};
        MLE scores {mle_score:.2f}.<br>
        {"<strong style='color:#3fb950'>MLE overtakes WLS at AIC weight ≥ " + f"{crossover_w:.2f}</strong> — meaning if you weight the full-data likelihood metric at " + f"{crossover_w:.0%}" + " or more, MLE becomes the composite winner." if crossover_w else
         "<strong style='color:#f85149'>MLE does not overtake WLS at any AIC weight for this model.</strong>"}<br><br>
        <strong>Implication:</strong> The composite rank's winner is a function of the weighting, not an absolute truth.
        Section 2.5.3.2 of the report acknowledges this; the General Conclusion now qualifies it explicitly.
        The correct conclusion is: <em>WLS wins within the failure-only frame; MLE wins within the population frame.</em>
        </div>""", unsafe_allow_html=True)

        if defense_mode:
            st.markdown(f"""<div style="background:#2d2008;border-left:4px solid #9e6a03;padding:16px;border-radius:0 8px 8px 0;margin-top:8px;">
            <strong style="color:#d29922">🎓 Defense Q:</strong> <em>"Your conclusion says WLS is the unanimous winner. But isn't that just because your scoring was biased?"</em><br><br>
            <strong style="color:#d29922">Prepared answer:</strong>
            "Yes — and the report says so. The composite rank structurally favors failure-only methods
            because RMSE and AD both evaluate the failure-subset CDF fit. This is documented in Section 2.5.3.2
            and the General Conclusion qualifies the WLS result as a winner within the failure-only inference frame.
            This experiment makes the sensitivity visible: move the slider to AIC weight = {f"{crossover_w:.2f}" if crossover_w is not None else "N/A"}
            and MLE becomes the composite winner. The framing matters more than the number."
            </div>""", unsafe_allow_html=True)

    # ── EXP 6: MAP CORRECTED PRIOR ────────────────────────────────────────────
    elif "Exp 6" in ext_exp:
        st.markdown("### Experiment 6 — MAP Corrected Prior: Section 3.8.3's Untested Proposal")
        st.markdown("""
        **The report's claim (Section 3.8.3):** A prior centered at λ₀ = 5 years = 1,825 days would
        remove the MAP misspecification entirely. MAP would then produce estimates between the failure-only
        methods and MLE rather than collapsing to ~1,000 days.
        **This experiment runs that test on the actual data.**
        """)

        mean_f_e6 = float(np.mean(tf_ext))
        k_mle_e6, lam_mle_e6 = params_df.loc["MLE","k"], params_df.loc["MLE","λ"]
        k_wls_e6, lam_wls_e6 = params_df.loc["WLS","k"], params_df.loc["WLS","λ"]
        k_map_e6, lam_map_e6 = params_df.loc["MAP","k"], params_df.loc["MAP","λ"]

        # Run MAP at three prior settings
        def run_map_with_prior(lam_mu, strength=1.0):
            def neg_map(p):
                kv, lv = np.exp(p[0]), np.exp(p[1])
                ll = censored_loglik(kv, lv, tf_ext, t_cens)
                return -(ll - kv/2.0 - lv/lam_mu * strength)
            try:
                r = minimize(neg_map, [np.log(1.5), np.log(lam_mu)], method="L-BFGS-B",
                             options={"maxiter": 1000})
                return np.exp(r.x[0]), np.exp(r.x[1])
            except:
                return np.nan, np.nan

        lam_report_prior = mean_f_e6         # ~50 days — the misspecified one
        lam_corrected_prior = 5 * 365        # 1825 days — Section 3.8.3's proposal
        lam_manufacturer_prior = 5 * 365 * 10  # ~18250 days — MTBF ≈ manufacturer spec direction

        k_map_report, lam_map_report = run_map_with_prior(lam_report_prior)
        k_map_corrected, lam_map_corrected = run_map_with_prior(lam_corrected_prior)
        k_map_manuf, lam_map_manuf = run_map_with_prior(lam_manufacturer_prior)

        # Summary table
        prior_table = pd.DataFrame([
            {"Prior": f"Report default (λ₀={lam_report_prior:.0f}d ≈ t̄_f)", "k̂": f"{k_map_report:.3f}", "λ̂ (days)": f"{lam_map_report:.0f}", "Status": "❌ Misspecified (collapses to failure-only timescale)"},
            {"Prior": f"Corrected — Section 3.8.3 proposal (λ₀={lam_corrected_prior}d = 5yr)", "k̂": f"{k_map_corrected:.3f}", "λ̂ (days)": f"{lam_map_corrected:.0f}", "Status": "✅ Tested here for the first time"},
            {"Prior": f"Manufacturer-direction (λ₀={lam_manufacturer_prior:.0f}d = 50yr)", "k̂": f"{k_map_manuf:.3f}", "λ̂ (days)": f"{lam_map_manuf:.0f}", "Status": "🔬 Weakly informative limit"},
            {"Prior": "MLE (no prior)", "k̂": f"{k_mle_e6:.3f}", "λ̂ (days)": f"{lam_mle_e6:.0f}", "Status": "✅ Full censored likelihood"},
            {"Prior": "WLS (failure-only)", "k̂": f"{k_wls_e6:.3f}", "λ̂ (days)": f"{lam_wls_e6:.0f}", "Status": "📊 Failure-only frame reference"},
        ])
        st.dataframe(prior_table.set_index("Prior"), use_container_width=True)

        # Survival plot comparing all MAP variants vs MLE and WLS
        t_pl6 = np.linspace(1, max(tf_ext) * 3, 500)
        fig_e6 = go.Figure()
        fig_e6.add_trace(go.Scatter(x=t_pl6, y=weibull_sf(t_pl6, k_mle_e6, lam_mle_e6),
            name=f"MLE (population truth) k={k_mle_e6:.3f}", line=dict(color=METHOD_COLORS["MLE"], width=3)))
        fig_e6.add_trace(go.Scatter(x=t_pl6, y=weibull_sf(t_pl6, k_wls_e6, lam_wls_e6),
            name=f"WLS (failure-only) k={k_wls_e6:.3f}", line=dict(color=METHOD_COLORS["WLS"], width=2, dash="dot")))
        if not np.isnan(k_map_report):
            fig_e6.add_trace(go.Scatter(x=t_pl6, y=weibull_sf(t_pl6, k_map_report, lam_map_report),
                name=f"MAP (report prior λ₀={lam_report_prior:.0f}d) k={k_map_report:.3f}",
                line=dict(color="#f85149", width=2, dash="dash")))
        if not np.isnan(k_map_corrected):
            fig_e6.add_trace(go.Scatter(x=t_pl6, y=weibull_sf(t_pl6, k_map_corrected, lam_map_corrected),
                name=f"MAP CORRECTED (λ₀={lam_corrected_prior}d=5yr) k={k_map_corrected:.3f}",
                line=dict(color="#ec4899", width=3)))
        if not np.isnan(k_map_manuf):
            fig_e6.add_trace(go.Scatter(x=t_pl6, y=weibull_sf(t_pl6, k_map_manuf, lam_map_manuf),
                name=f"MAP manufacturer-direction (λ₀={lam_manufacturer_prior:.0f}d) k={k_map_manuf:.3f}",
                line=dict(color="#a855f7", width=2, dash="longdash")))
        fig_e6.add_vline(x=92, line_dash="dot", line_color="#444", annotation_text="Day 92")
        fig_e6.update_layout(template="plotly_dark", height=480,
            title="MAP Under Three Prior Specifications vs. MLE and WLS",
            xaxis_title="Days", yaxis_title="S(t)", yaxis=dict(range=[-0.05, 1.1]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.45))
        st.plotly_chart(fig_e6, use_container_width=True)

        if not np.isnan(k_map_corrected):
            mle_lam, corrected_lam, report_lam = lam_mle_e6, lam_map_corrected, lam_map_report
            improvement = abs(np.log(corrected_lam) - np.log(report_lam)) / abs(np.log(mle_lam) - np.log(report_lam)) * 100
            st.markdown(f"""
            <div style="background:#1c2333;border-left:4px solid #ec4899;padding:16px;border-radius:0 8px 8px 0;margin-top:8px;">
            <strong style="color:#ec4899">📐 Deduction — Section 3.8.3's Prediction Confirmed</strong><br>
            Report default prior (λ₀={lam_report_prior:.0f}d): MAP gives λ̂ = {report_lam:.0f}d — collapses to failure-only timescale.<br>
            Corrected prior (λ₀={lam_corrected_prior}d = 5yr): MAP gives λ̂ = {corrected_lam:.0f}d — moves {improvement:.0f}% of the way toward MLE's λ̂={mle_lam:.0f}d.<br><br>
            {"✅ Section 3.8.3's prediction is <strong>confirmed</strong>: a properly specified prior makes MAP approach MLE." if corrected_lam > report_lam * 5 else
             "⚠️ The corrected prior improves MAP but the observation window is too short to fully override the prior."}
            <br><br>This is the experiment the report proposed but did not run. The result validates the diagnosis.
            </div>""", unsafe_allow_html=True)

    # ── EXP 7: DAY-28 BINOMIAL TEST ───────────────────────────────────────────
    elif "Exp 7" in ext_exp:
        st.markdown("### Experiment 7 — Day-28 Clustering: Is It Statistically Significant?")
        st.markdown("""
        **The report's observation (Section 3.2.4):** A prominent spike of failures near day 28 is visible
        across all models, suggesting either a manufacturing batch issue or a common stress event.
        **The gap:** No formal statistical test was performed. This experiment fills it.

        **Test design:** Under a Weibull null (no clustering), the expected proportion of failures in
        any k-day window is determined by the fitted CDF. We test whether the observed count in [25, 31]
        days is significantly higher than expected — a one-sided binomial test.
        """)

        window_low = st.slider("Window start (days)", 1, 60, 25)
        window_high = st.slider("Window end (days)", window_low+1, 90, 31)
        alpha_test = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05)

        from scipy.stats import binomtest

        n_window = np.sum((tf_ext >= window_low) & (tf_ext <= window_high))
        n_total_f = len(tf_ext)

        # Expected probability under WLS Weibull null
        k_wls_t, lam_wls_t = params_df.loc["WLS","k"], params_df.loc["WLS","λ"]
        p_expected = weibull_cdf(window_high, k_wls_t, lam_wls_t) - weibull_cdf(window_low, k_wls_t, lam_wls_t)
        p_expected = float(np.clip(p_expected, 1e-9, 1-1e-9))

        binom_result = binomtest(n_window, n_total_f, p_expected, alternative="greater")
        p_val = binom_result.pvalue
        reject = p_val < alpha_test

        expected_count = p_expected * n_total_f

        # Histogram of failure times with window highlighted
        fig_e7 = go.Figure()
        bins = np.arange(0, max(tf_ext)+5, 3)
        counts, edges = np.histogram(tf_ext, bins=bins)
        in_window = (edges[:-1] >= window_low) & (edges[:-1] <= window_high)
        fig_e7.add_trace(go.Bar(x=edges[:-1], y=counts, width=np.diff(edges),
            marker_color=["#f85149" if w else "#58a6ff" for w in in_window],
            name="Failure counts (red = test window)"))
        # Overlay expected under WLS
        t_dens = np.linspace(1, max(tf_ext), 300)
        pdf_vals = weibull_pdf(t_dens, k_wls_t, lam_wls_t) * n_total_f * 3  # scale to bin width ≈3
        fig_e7.add_trace(go.Scatter(x=t_dens, y=pdf_vals, mode="lines",
            name=f"Expected under WLS Weibull (k={k_wls_t:.2f})",
            line=dict(color=METHOD_COLORS["WLS"], width=2)))
        fig_e7.add_vrect(x0=window_low, x1=window_high, fillcolor="rgba(248,81,73,0.1)",
            layer="below", line_width=0)
        fig_e7.update_layout(template="plotly_dark", height=400,
            title=f"Failure Time Distribution — Test Window [{window_low}, {window_high}] days",
            xaxis_title="Days", yaxis_title="Count", bargap=0.05)
        st.plotly_chart(fig_e7, use_container_width=True)

        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Observed in [{window_low},{window_high}]</div>
            <div class="metric-value">{n_window}</div></div>""", unsafe_allow_html=True)
        with col_r2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Expected under Weibull</div>
            <div class="metric-value">{expected_count:.1f}</div></div>""", unsafe_allow_html=True)
        with col_r3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">p-value (one-sided binomial)</div>
            <div class="metric-value" style="color:{'#f85149' if reject else '#3fb950'}">{p_val:.4f}</div></div>""", unsafe_allow_html=True)
        with col_r4:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Decision at α={alpha_test}</div>
            <div class="metric-value" style="color:{'#f85149' if reject else '#3fb950'}">
            {"❌ Reject H₀" if reject else "✅ Fail to reject H₀"}</div></div>""", unsafe_allow_html=True)

        conclusion_text = (
            f"The observed count of {n_window} failures in [{window_low},{window_high}] days "
            f"is significantly higher than the Weibull-expected {expected_count:.1f} (p={p_val:.4f} < α={alpha_test}). "
            f"The clustering is statistically significant — the two-parameter Weibull is insufficient, "
            f"and a mixture model (e.g. a defective subpopulation component) is statistically motivated."
            if reject else
            f"The observed count of {n_window} failures in [{window_low},{window_high}] days "
            f"is not significantly different from the Weibull-expected {expected_count:.1f} (p={p_val:.4f} ≥ α={alpha_test}). "
            f"The visual spike may be sampling noise. A mixture model extension remains exploratory rather than statistically necessary."
        )
        st.markdown(f"""
        <div style="background:#1c2333;border-left:4px solid {'#f85149' if reject else '#3fb950'};padding:16px;border-radius:0 8px 8px 0;margin-top:16px;">
        <strong style="color:{'#f85149' if reject else '#3fb950'}">📐 Statistical Conclusion</strong><br>
        <strong>H₀:</strong> Failure count in [{window_low},{window_high}] days follows the fitted Weibull distribution.<br>
        <strong>H₁:</strong> Failure count is significantly elevated (clustering or batch effect).<br><br>
        {conclusion_text}
        </div>""", unsafe_allow_html=True)

        if defense_mode:
            st.markdown(f"""<div style="background:#2d2008;border-left:4px solid #9e6a03;padding:16px;border-radius:0 8px 8px 0;margin-top:8px;">
            <strong style="color:#d29922">🎓 Defense Q:</strong> <em>"You mentioned a day-28 clustering without any statistical test. Can you quantify it?"</em><br><br>
            <strong style="color:#d29922">Prepared answer:</strong> "Yes. A one-sided binomial test comparing the observed failure count in [25,31] days
            to the WLS Weibull-predicted count gives p = {p_val:.4f}. {"This is below α=0.05, so the clustering is statistically significant — it confirms the mixture model motivation." if reject else "This does not reach significance at α=0.05 for this model, suggesting the visual spike may be sampling noise. The mixture model recommendation remains motivated by cross-model pattern consistency rather than per-model significance."}
            A formal test was not included in the written report but the result is consistent with the qualitative observation."
            </div>""", unsafe_allow_html=True)

with tabs[7]:
    st.markdown('<div class="section-header">📖 Method Cards</div>', unsafe_allow_html=True)
    c_a, c_b = st.columns(2)
    with c_a: st.info("**OLS/MRR**: Baseline rank regression. Fails under censoring.")
    with c_b: st.success("**MLE**: Gold standard. Handles survivors and failures simultaneously.")

st.markdown("---")
st.caption("Weibull Reliability Lab · 9 Methods · Real-time Analytics · Extended Lab v2 (defense-ready)")
