import warnings
warnings.filterwarnings("ignore")

import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score

import umap
import hdbscan
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION PAGE
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alerte Fournisseurs Maroc",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.header-box {
    background: linear-gradient(135deg, #1F3864, #2E5496);
    padding: 20px 28px; border-radius: 10px; margin-bottom: 20px;
}
.header-box h1 { color: white; font-size: 1.7rem; margin: 0; }
.header-box p  { color: #BDD7EE; font-size: 0.9rem; margin: 4px 0 0; }
.kpi-card {
    border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 16px 10px; text-align: center;
    background: white; box-shadow: 0 2px 6px rgba(0,0,0,.07);
}
.kpi-val { font-size: 2.2rem; font-weight: 700; color: #1F3864; }
.kpi-lab { font-size: 0.8rem; color: #595959; margin-top: 4px; }
.alerte-rouge  { background:#ffcccc; border-left:5px solid #C00000;
                  padding:12px 18px; border-radius:6px; font-weight:700; margin:8px 0; }
.alerte-orange { background:#ffe0b2; border-left:5px solid #C55A11;
                  padding:12px 18px; border-radius:6px; font-weight:700; margin:8px 0; }
.alerte-vert   { background:#ccffcc; border-left:5px solid #375623;
                  padding:12px 18px; border-radius:6px; font-weight:700; margin:8px 0; }
.shap-bar {
    display:inline-block; height:11px; border-radius:3px;
    vertical-align:middle; margin-left:8px;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  CONSTANTES
# ──────────────────────────────────────────────────────────────
EXCLURE = [
    "ID_Fournisseur", "Nom_Fournisseur", "Secteur", "Region_Maroc",
    "Cluster_Reel", "Note_Risque_Pays", "Certification",
    "Niveau_Alerte", "Priorite_Action", "Score_Risque",
    "Alerte_ML", "Priorite_ML", "Cluster_HDBSCAN",
]
COULEURS = {"🟢 Vert": "#4CAF50", "🟠 Orange": "#FF9800", "🔴 Rouge": "#F44336"}

# ──────────────────────────────────────────────────────────────
#  BARRE LATERALE
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")

    # ── RÉSULTATS EN PREMIER (placeholder rempli après analyse) ──
    st.markdown("### 📊 Résultats de l'analyse")
    _sidebar_results = st.empty()   # sera rempli après le pipeline ML
    st.markdown("---")

    # ── Proposition de paramétrage (placeholder rempli après chargement du dataset) ──
    st.markdown("### 💡 Proposition de paramétrage")
    st.caption("Recommandations calculées sur votre dataset")
    _sidebar_param_guide = st.empty()  # sera rempli après chargement du dataset
    st.markdown("---")

    st.markdown("### UMAP")
    p_neighbors   = st.slider("n_neighbors",         5,  50, 15)
    st.caption("↑ Grand = structure globale · Petit = détail local des fournisseurs proches")
    p_min_dist    = st.slider("min_dist",           0.0, 0.5, 0.1, 0.05)
    st.caption("↑ Grand = clusters étalés · Petit = clusters compacts et denses")
    st.markdown("### HDBSCAN")
    p_min_cluster = st.slider("min_cluster_size",   3,  30, 10)
    st.caption("↑ Grand = moins de clusters, plus stables et robustes")
    p_min_samples = st.slider("min_samples",        1,  15,  5)
    st.caption("↑ Grand = plus conservateur, plus de points classés bruit")
    st.markdown("### Isolation Forest")
    p_contam      = st.slider("Contamination (%)",  1,  20,  5) / 100
    st.caption("↑ Grand = plus de fournisseurs détectés comme anormaux")
    st.markdown("### Poids score composite")
    p_w_cl = st.slider("Poids Cluster (%)",         10, 60, 35) / 100
    p_w_an = st.slider("Poids Anomalie (%)",        10, 60, 30) / 100
    p_w_te = st.slider("Poids Temporel (%)",         5, 40, 20) / 100
    p_w_dt = max(0.0, 1.0 - p_w_cl - p_w_an - p_w_te)
    st.info(f"Poids DTW (auto) : {p_w_dt*100:.0f}%")
    st.markdown(f"""
<div style='font-size:0.75rem;margin-top:4px'>
  <span style='background:#1F3864;color:white;padding:2px 6px;border-radius:3px'>Cl {p_w_cl*100:.0f}%</span>
  <span style='background:#C00000;color:white;padding:2px 6px;border-radius:3px'>An {p_w_an*100:.0f}%</span>
  <span style='background:#375623;color:white;padding:2px 6px;border-radius:3px'>Te {p_w_te*100:.0f}%</span>
  <span style='background:#7B2D8B;color:white;padding:2px 6px;border-radius:3px'>DT {p_w_dt*100:.0f}%</span>
</div>""", unsafe_allow_html=True)
    st.markdown("### SHAP")
    p_shap     = st.checkbox("Calculer SHAP", value=True)
    p_shap_top = st.slider("Top N variables", 5, 30, 15)
    st.caption("↑ Plus de variables expliquées, même temps de calcul")
    st.markdown("### Seuils d'alerte")
    p_seuil_vert   = st.slider("Seuil Vert / Orange",  10, 45, 29)
    p_seuil_orange = st.slider("Seuil Orange / Rouge", 40, 80, 59)
    st.markdown(f"""
<div style='font-size:0.75rem;margin-top:6px;padding:6px 8px;background:#f9f9f9;border-radius:5px'>
  🟢 0 – {p_seuil_vert} &nbsp;|&nbsp; 🟠 {p_seuil_vert+1} – {p_seuil_orange} &nbsp;|&nbsp; 🔴 {p_seuil_orange+1} – 100
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Université Mohammed V — Rabat — FSJES Agdal\nMaster M.I.E.L · 2025–2026")

# ──────────────────────────────────────────────────────────────
#  EN-TETE
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🔍 Système d'Alerte Précoce — Risque Fournisseurs Maroc</h1>
  <p>Pipeline ML : UMAP → HDBSCAN → Isolation Forest → VAE → SHAP</p>
  <p style='color:#93b8dc;font-size:0.8rem;margin-top:6px'>
    Université Mohammed V — Rabat — FSJES Agdal &nbsp;·&nbsp; Master M.I.E.L &nbsp;·&nbsp; 2025–2026
  </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  UPLOAD DU FICHIER
# ──────────────────────────────────────────────────────────────
st.markdown("### 📁 Étape 1 — Chargez votre dataset")

fichier = st.file_uploader(
    "Importez votre fichier Excel (.xlsx) ou CSV (.csv)",
    type=["xlsx", "csv"],
)

if fichier is None:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.info("**Étape 1** — Chargez votre dataset ci-dessus")
    c2.info("**Étape 2** — Ajustez les paramètres (barre gauche)")
    c3.info("**Étape 3** — Cliquez sur Lancer l'analyse")
    st.stop()

@st.cache_data(show_spinner=False)
def lire(contenu, nom):
    if nom.endswith(".xlsx"):
        # Détection automatique de la vraie ligne d'en-tête
        # (certains fichiers ont une ligne de titres de blocs avant les vrais noms de colonnes)
        df_raw = pd.read_excel(io.BytesIO(contenu), header=None, nrows=15)
        header_row = 0
        best_score = 0
        for i in range(len(df_raw)):
            row = df_raw.iloc[i]
            score = row.apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 1).sum()
            if score > best_score:
                best_score = score
                header_row = i
        df = pd.read_excel(io.BytesIO(contenu), header=header_row)
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all").reset_index(drop=True)
        return df
    df = pd.read_csv(io.BytesIO(contenu))
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
    return df

with st.spinner("Lecture du fichier…"):
    df = lire(fichier.read(), fichier.name)

st.success(f"✅  {fichier.name} — {df.shape[0]} fournisseurs × {df.shape[1]} colonnes")

with st.expander("👁️  Aperçu (5 premières lignes)"):
    st.dataframe(df.head(), use_container_width=True)

features = [c for c in df.columns
            if c not in EXCLURE
            and pd.api.types.is_numeric_dtype(df[c])]

with st.expander(f"🔢  {len(features)} variables numériques pour le ML"):
    cols = st.columns(4)
    for i, f in enumerate(features):
        cols[i % 4].markdown(f"- `{f}`")

if len(features) == 0:
    st.error(
        "❌ Aucune variable numérique détectée dans le fichier. "
        "Vérifiez que votre dataset contient des colonnes numériques "
        "non présentes dans la liste EXCLURE."
    )
    st.stop()

X_brut = df[features].copy()
X_brut = X_brut.apply(pd.to_numeric, errors="coerce").astype("float64")

if X_brut.isnull().all(axis=None):
    st.error("❌ Toutes les valeurs numériques sont manquantes (NaN). Impossible de lancer l'analyse.")
    st.stop()

# ── Injection du guide de paramétrage adapté au dataset ─────────────────────
_n_fournisseurs = len(df)
_n_features     = len(features)
_nan_global_pct = round(X_brut.isnull().sum().sum() / (_n_fournisseurs * _n_features) * 100, 1)

# Calcul des recommandations intelligentes basées sur le dataset
_rec_neighbors    = int(np.clip(int(np.sqrt(_n_fournisseurs)), 8, 30))
_rec_min_dist     = 0.05 if _n_fournisseurs < 150 else 0.10 if _n_fournisseurs < 500 else 0.15
_rec_min_cluster  = max(3, int(_n_fournisseurs * 0.04))   # ~4 % du dataset
_rec_min_samples  = max(1, _rec_min_cluster // 2)
_rec_contam       = 5 if _nan_global_pct < 10 else 8 if _nan_global_pct < 20 else 10
_has_temporal     = any(c in features for c in ["PSI_Score","Changepoint_PELT","Tendance_OTD_6M","OTD_Pct"])
_has_esg          = any(c in features for c in ["Score_ESG","Stabilite_Politique","Note_Risque_Pays"])
_rec_w_te         = 25 if _has_temporal else 10
_rec_w_cl         = 35
_rec_w_an         = 30
_rec_w_dt         = max(0, 100 - _rec_w_cl - _rec_w_an - _rec_w_te)
_rec_shap_top     = min(30, max(5, _n_features))

# Score composite fictif pour estimer les seuils (seront affinés après analyse)
_rec_seuil_vert   = 29
_rec_seuil_orange = 59

_color_contam = "#C00000" if _rec_contam >= 10 else "#FF9800" if _rec_contam >= 7 else "#375623"

_sidebar_param_guide.markdown(f"""
<div style='background:#EEF4FB;border-left:4px solid #2E5496;padding:10px 12px;border-radius:6px;font-size:0.77rem;line-height:1.55'>

<b>📌 Dataset détecté :</b> {_n_fournisseurs} fournisseurs × {_n_features} variables<br>
<b>NaN globaux :</b> {_nan_global_pct} % &nbsp;|&nbsp; <b>Variables temporelles :</b> {"✅ Oui" if _has_temporal else "❌ Non"}

<hr style='margin:7px 0;border:none;border-top:1px solid #cdd7e7'>

<b>🔵 UMAP</b><br>
→ <code>n_neighbors = <b>{_rec_neighbors}</b></code><br>
<i>≈ √{_n_fournisseurs} : règle empirique qui équilibre structure locale et globale pour ce volume.</i><br>
→ <code>min_dist = <b>{_rec_min_dist}</b></code><br>
<i>{"Petit dataset → clusters compacts (0.05)" if _n_fournisseurs < 150 else "Dataset moyen → valeur standard (0.10)" if _n_fournisseurs < 500 else "Grand dataset → légèrement étalé (0.15) pour lisibilité"}.</i>

<hr style='margin:7px 0;border:none;border-top:1px solid #cdd7e7'>

<b>🟢 HDBSCAN</b><br>
→ <code>min_cluster_size = <b>{_rec_min_cluster}</b></code><br>
<i>≈ 4 % × {_n_fournisseurs} = {_rec_min_cluster} fournisseurs minimum par groupe. Évite les micro-clusters instables.</i><br>
→ <code>min_samples = <b>{_rec_min_samples}</b></code><br>
<i>= ½ × min_cluster_size : compromis densité/bruit recommandé par McInnes et al. (2017).</i>

<hr style='margin:7px 0;border:none;border-top:1px solid #cdd7e7'>

<b>🔴 Isolation Forest</b><br>
→ <code>Contamination = <b>{_rec_contam} %</b></code><br>
<i>{"NaN faibles → données fiables → hypothèse conservatrice (5 %)." if _rec_contam == 5 else f"NaN élevés ({_nan_global_pct} %) → incertitude accrue → seuil relevé à {_rec_contam} %."}</i>

<hr style='margin:7px 0;border:none;border-top:1px solid #cdd7e7'>

<b>⚖️ Poids score composite</b><br>
→ Cluster <b>{_rec_w_cl} %</b> · Anomalie <b>{_rec_w_an} %</b> · Temporel <b>{_rec_w_te} %</b> · DTW <b>{_rec_w_dt} %</b><br>
<i>{"Variables temporelles détectées → poids Temporel renforcé à " + str(_rec_w_te) + " %." if _has_temporal else "Pas de variable temporelle → poids Temporel minimal (10 %), DTW compensé."}</i>

<hr style='margin:7px 0;border:none;border-top:1px solid #cdd7e7'>

<b>🧠 SHAP</b><br>
→ <code>Top N = <b>{_rec_shap_top}</b></code><br>
<i>= {_n_features} variables disponibles → afficher toutes pour une explicabilité complète.</i>

<hr style='margin:7px 0;border:none;border-top:1px solid #cdd7e7'>

<b>🚦 Seuils d'alerte</b><br>
→ <code>Vert/Orange = <b>{_rec_seuil_vert}</b></code> · <code>Orange/Rouge = <b>{_rec_seuil_orange}</b></code><br>
<i>Valeurs par défaut robustes (P30/P60). À recalibrer après analyse en visant ~60 % Vert, ~25 % Orange, ~15 % Rouge.</i>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  BOUTON LANCEMENT
# ──────────────────────────────────────────────────────────────
st.markdown("### 🚀 Étape 2 — Lancez l'analyse")

if not st.button("▶  Lancer l'analyse complète", type="primary", use_container_width=True):
    st.info("👆 Cliquez sur le bouton — durée estimée : 3 à 5 minutes")
    st.stop()

# ── Placeholders résultats : affichés EN PREMIER, remplis après le pipeline ──
_ph_kpi    = st.empty()   # cartes KPI
_ph_tabs   = st.empty()   # onglets principaux

# ──────────────────────────────────────────────────────────────
#  PIPELINE ML
# ──────────────────────────────────────────────────────────────
barre   = st.progress(0, text="Démarrage…")
log     = st.empty()
t_debut = time.time()

def avancer(msg, pct, ok=False):
    ic = "✅" if ok else "⏳"
    barre.progress(pct, text=f"{ic}  {msg}")
    log.info(f"{ic}  {msg}")

# 1. Prétraitement
avancer("Étape 1/7 — Prétraitement", 5)
imputer = SimpleImputer(strategy="median")
scaler  = RobustScaler()
X_imp   = imputer.fit_transform(X_brut.values)
X_sc    = scaler.fit_transform(X_imp)
avancer("Prétraitement terminé", 12, ok=True)

# Variables calculées ici, réutilisées dans t6 (Métriques Algorithmes)
nan_counts     = X_brut.isnull().sum()
nan_pct        = (nan_counts / len(X_brut) * 100).round(1)
total_nan      = int(nan_counts.sum())
total_cells    = X_brut.shape[0] * X_brut.shape[1]
pct_nan_global = round(total_nan / total_cells * 100, 2)
vars_sans_nan  = int((nan_counts == 0).sum())
vars_avec_nan  = int((nan_counts > 0).sum())
var_var        = X_brut.var().sort_values(ascending=False)
top_vars_norm  = list(var_var.head(min(6, len(features))).index)
X_brut_df      = pd.DataFrame(X_imp, columns=features)
X_sc_df        = pd.DataFrame(X_sc,  columns=features)

# 2. PCA
avancer("Étape 2/7 — PCA", 15)
pca   = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_sc)
n_pca = pca.n_components_
avancer(f"PCA : {X_sc.shape[1]}D → {n_pca}D", 22, ok=True)

# 3. UMAP
avancer("Étape 3/7 — UMAP (30 à 40 secondes…)", 25)
X_u3 = umap.UMAP(n_components=3, n_neighbors=p_neighbors,
                   min_dist=p_min_dist, random_state=42).fit_transform(X_pca)
X_u2 = umap.UMAP(n_components=2, n_neighbors=p_neighbors,
                   min_dist=p_min_dist, random_state=42).fit_transform(X_pca)
avancer("UMAP terminé", 42, ok=True)

# 4. HDBSCAN
avancer("Étape 4/7 — HDBSCAN", 46)
cl     = hdbscan.HDBSCAN(min_cluster_size=p_min_cluster,
                           min_samples=p_min_samples,
                           metric="euclidean", prediction_data=True)
labels = cl.fit_predict(X_u3)
n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
n_br   = int((labels == -1).sum())
mask   = labels != -1
sil    = silhouette_score(X_u3[mask], labels[mask]) if mask.sum() > 10 else 0.0
dbi    = davies_bouldin_score(X_u3[mask], labels[mask]) if mask.sum() > 10 else 9.9
avancer(f"HDBSCAN : {n_cl} clusters + {n_br} anomalies", 57, ok=True)

# 5. Isolation Forest
avancer("Étape 5/7 — Isolation Forest", 60)
iso    = IsolationForest(n_estimators=200, contamination=p_contam,
                          random_state=42, n_jobs=-1)
iso.fit(X_sc)
if_r   = iso.decision_function(X_sc)
if_sc  = 1 - (if_r - if_r.min()) / (if_r.max() - if_r.min())
n_anom = int((iso.predict(X_sc) == -1).sum())
avancer(f"Isolation Forest : {n_anom} anomalies", 68, ok=True)

# 6. VAE
avancer("Étape 6/7 — VAE", 70)

class VAE(nn.Module):
    def __init__(self, d, l=8):
        super().__init__()
        h = max(d // 2, l * 4)
        self.enc = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1),
            nn.Dropout(0.2), nn.Linear(h, h // 2), nn.LeakyReLU(0.1))
        self.mu  = nn.Linear(h // 2, l)
        self.lv  = nn.Linear(h // 2, l)
        self.dec = nn.Sequential(
            nn.Linear(l, h // 2), nn.LeakyReLU(0.1),
            nn.Linear(h // 2, h), nn.BatchNorm1d(h),
            nn.LeakyReLU(0.1), nn.Linear(h, d))
    def forward(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.dec(z), mu, lv

Xn  = torch.FloatTensor(X_sc[labels != -1])
ld  = DataLoader(TensorDataset(Xn), batch_size=32, shuffle=True)
vae = VAE(X_sc.shape[1])
opt = optim.Adam(vae.parameters(), lr=1e-3)
vae.train()
for _ in range(60):
    for (b,) in ld:
        opt.zero_grad()
        r, mu, lv = vae(b)
        loss = nn.functional.mse_loss(r, b, reduction="sum") \
             - 5e-4 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        loss.backward(); opt.step()

vae.eval()
with torch.no_grad():
    Xt = torch.FloatTensor(X_sc)
    r_all, mu_all, lv_all = vae(Xt)
    ve = nn.functional.mse_loss(r_all, Xt, reduction="none").mean(dim=1).numpy()
    vae_recon_loss = float(nn.functional.mse_loss(r_all, Xt).item())
    vae_kl_loss    = float(-0.5 * torch.sum(1 + lv_all - mu_all.pow(2) - lv_all.exp()).item() / len(Xt))
vae_sc = (ve - ve.min()) / (ve.max() - ve.min())
anom   = (if_sc + vae_sc) / 2
avancer("VAE terminé", 80, ok=True)

# 7. Score composite
avancer("Étape 7/7 — Score composite", 84)

cr = {c: float(if_sc[labels == c].mean()) if c != -1 else 1.0 for c in set(labels)}
cc = np.array([cr[c] for c in labels])

def gcol(n):
    return df[n].fillna(0).astype(float).values if n in df.columns else np.zeros(len(df))

derive = (0.4 * np.clip(gcol("PSI_Score") / 0.5, 0, 1)
        + 0.4 * gcol("Changepoint_PELT")
        + 0.2 * np.clip(np.abs(gcol("Tendance_OTD_6M")) / 10, 0, 1))

score_100 = np.clip(
    (p_w_cl * cc + p_w_an * anom + p_w_te * derive + p_w_dt * vae_sc) * 100,
    0, 100)

def alerte(s):
    if s <= p_seuil_vert:   return "🟢 Vert"
    if s <= p_seuil_orange: return "🟠 Orange"
    return "🔴 Rouge"

alertes  = np.array([alerte(s) for s in score_100])
n_vert   = int((alertes == "🟢 Vert").sum())
n_orange = int((alertes == "🟠 Orange").sum())
n_rouge  = int((alertes == "🔴 Rouge").sum())

df_res = df.copy()
df_res["Cluster_HDBSCAN"] = labels
df_res["Score_IF"]        = np.round(if_sc * 100, 1)
df_res["Score_VAE"]       = np.round(vae_sc * 100, 1)
df_res["Score_Anomalie"]  = np.round(anom * 100, 1)
df_res["Score_Risque_ML"] = np.round(score_100, 1)
df_res["Alerte_ML"]       = alertes
df_res["Priorite_ML"]     = [
    "IMMÉDIAT" if a == "🔴 Rouge" else "SURVEILLANCE" if a == "🟠 Orange" else "STANDARD"
    for a in alertes]

t_total = time.time() - t_debut
avancer(f"Pipeline complet en {t_total:.0f} secondes", 90, ok=True)

# SHAP
shap_values = None
shap_df     = None
if p_shap:
    avancer("Calcul SHAP…", 93)
    try:
        exp         = shap.TreeExplainer(iso)
        shap_values = exp.shap_values(X_sc)
        ma          = np.abs(shap_values).mean(axis=0)
        shap_df     = pd.DataFrame({
            "Variable": features,
            "SHAP_abs": ma,
            "SHAP_pct": (ma / ma.sum() * 100).round(1),
        }).sort_values("SHAP_abs", ascending=False).reset_index(drop=True)
        df_res["Top_Variable_SHAP"] = [
            features[int(np.argmax(np.abs(shap_values[i])))] for i in range(len(df))]
        avancer("SHAP terminé", 98, ok=True)
    except Exception as e:
        st.warning(f"SHAP non disponible : {e}")

barre.progress(100, text="✅  Analyse terminée !")
log.success(f"✅  {len(df)} fournisseurs analysés en {t_total:.0f} secondes")

# ── Injection des résultats dans la sidebar (placeholder défini plus haut) ──
_pct_vert   = round(n_vert   / len(df) * 100)
_pct_orange = round(n_orange / len(df) * 100)
_pct_rouge  = round(n_rouge  / len(df) * 100)
_sil_ok  = "✅" if sil > 0.50 else "⚠️"
_dbi_ok  = "✅" if dbi < 1.50 else "⚠️"
_sidebar_results.markdown(f"""
<div style='font-size:0.82rem;line-height:1.7'>
<div style='background:#f0f4fa;border-radius:7px;padding:10px 12px;margin-bottom:6px'>
  <b style='color:#1F3864'>🏭 {len(df)}</b> fournisseurs analysés<br>
  <b style='color:#1F3864'>🗂️ {n_cl}</b> clusters · <b>{n_br}</b> anomalies HDBSCAN<br>
  <b style='color:#1F3864'>⏱️ {t_total:.0f} s</b> · Score moyen : <b>{score_100.mean():.1f}/100</b>
</div>
<div style='background:#d4edda;border-left:4px solid #375623;border-radius:5px;padding:7px 10px;margin-bottom:4px'>
  🟢 <b>Vert</b> — {n_vert} fournisseurs ({_pct_vert} %)
</div>
<div style='background:#fff3cd;border-left:4px solid #C55A11;border-radius:5px;padding:7px 10px;margin-bottom:4px'>
  🟠 <b>Orange</b> — {n_orange} fournisseurs ({_pct_orange} %)
</div>
<div style='background:#f8d7da;border-left:4px solid #C00000;border-radius:5px;padding:7px 10px;margin-bottom:6px'>
  🔴 <b>Rouge</b> — {n_rouge} fournisseurs ({_pct_rouge} %)
</div>
<div style='background:#f9f9f9;border-radius:5px;padding:7px 10px;font-size:0.75rem;color:#444'>
  Silhouette : <b>{sil:.3f}</b> {_sil_ok} &nbsp;|&nbsp; Davies-Bouldin : <b>{dbi:.3f}</b> {_dbi_ok}<br>
  Anomalies IF : <b>{n_anom}</b> &nbsp;|&nbsp; PCA : <b>{n_pca}</b> composantes
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  REMPLISSAGE DES PLACEHOLDERS RÉSULTATS (affichés EN PREMIER)
# ──────────────────────────────────────────────────────────────

# ── KPI ──────────────────────────────────────────────────────
with _ph_kpi.container():
    st.markdown("### 📊 Résultats")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(df)}</div><div class="kpi-lab">Fournisseurs</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_cl}</div><div class="kpi-lab">Clusters ML</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card" style="border-color:#4CAF50"><div class="kpi-val" style="color:#375623">{n_vert}</div><div class="kpi-lab">🟢 Vert</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card" style="border-color:#FF9800"><div class="kpi-val" style="color:#C55A11">{n_orange}</div><div class="kpi-lab">🟠 Orange</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="kpi-card" style="border-color:#C00000"><div class="kpi-val" style="color:#C00000">{n_rouge}</div><div class="kpi-lab">🔴 Rouge</div></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="kpi-card"><div class="kpi-val">{score_100.mean():.1f}</div><div class="kpi-lab">Score moyen</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📏 Métriques de validation ML", expanded=False):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Silhouette",        f"{sil:.3f}", delta="✅ > 0.50" if sil > 0.50 else "⚠️")
        m2.metric("Davies-Bouldin",    f"{dbi:.3f}", delta="✅ < 1.50" if dbi < 1.50 else "⚠️")
        m3.metric("Anomalies IF",      n_anom)
        m4.metric("Anomalies HDBSCAN", n_br)

# ── ONGLETS ──────────────────────────────────────────────────
with _ph_tabs.container():
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📌 Clusters UMAP",
        "📊 Distribution",
        "🔬 Anomalies",
        "🧠 SHAP",
        "📋 Tableau",
        "📐 Métriques Algorithmes",
    ])

    dp = df.copy()
    dp["UMAP1"]   = X_u2[:, 0]
    dp["UMAP2"]   = X_u2[:, 1]
    dp["Cluster"] = ["Anomalie" if c == -1 else f"Cluster {c}" for c in labels]
    dp["Alerte"]  = alertes
    dp["Score"]   = np.round(score_100, 1)
    hv = [c for c in ["ID_Fournisseur", "Secteur", "Region_Maroc"] if c in dp.columns]

with t1:
    st.subheader("Projection UMAP 2D")
    choix = st.radio("Colorier par :", ["Alerte", "Cluster", "Score"], horizontal=True)
    if choix == "Alerte":
        fig = px.scatter(dp, x="UMAP1", y="UMAP2", color="Alerte",
                          color_discrete_map=COULEURS, hover_data=hv + ["Score"],
                          title="UMAP 2D — Niveaux d'alerte",
                          template="plotly_white", height=520)
    elif choix == "Cluster":
        fig = px.scatter(dp, x="UMAP1", y="UMAP2", color="Cluster",
                          hover_data=hv + ["Score"],
                          title="UMAP 2D — Clusters HDBSCAN",
                          template="plotly_white", height=520)
    else:
        fig = px.scatter(dp, x="UMAP1", y="UMAP2", color="Score",
                          color_continuous_scale="RdYlGn_r", hover_data=hv,
                          title="UMAP 2D — Score de risque",
                          template="plotly_white", height=520)
    fig.update_traces(marker=dict(size=8, opacity=0.82, line=dict(width=0.5, color="white")))
    fig.update_layout(font=dict(family="Arial", size=11), title_font=dict(size=14, color="#1F3864"))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("Distribution des alertes")
    ca, cb = st.columns(2)
    fig_h = px.histogram(x=score_100, nbins=40, color_discrete_sequence=["#2E5496"],
                          labels={"x": "Score (0–100)", "count": "Fréquence"},
                          title="Distribution du score composite", template="plotly_white")
    fig_h.add_vline(x=p_seuil_vert,    line_dash="dash", line_color="#4CAF50",
                     annotation_text=f"Vert ({p_seuil_vert})")
    fig_h.add_vline(x=p_seuil_orange,  line_dash="dash", line_color="#FF9800",
                     annotation_text=f"Orange ({p_seuil_orange})")
    fig_h.add_vline(x=score_100.mean(), line_dash="dot",  line_color="#C00000",
                     annotation_text=f"Moy {score_100.mean():.1f}")
    ca.plotly_chart(fig_h, use_container_width=True)
    fig_p = px.pie(names=["🟢 Vert", "🟠 Orange", "🔴 Rouge"],
                    values=[n_vert, n_orange, n_rouge],
                    color=["🟢 Vert", "🟠 Orange", "🔴 Rouge"],
                    color_discrete_map=COULEURS,
                    title="Répartition des alertes", hole=0.4)
    fig_p.update_traces(textinfo="label+percent+value", textposition="outside")
    cb.plotly_chart(fig_p, use_container_width=True)
    df_box = pd.DataFrame({
        "Cluster": ["Anomalie" if c == -1 else f"C{c}" for c in labels],
        "Score":   score_100})
    fig_b = px.box(df_box, x="Cluster", y="Score", color="Cluster",
                    title="Score par cluster", template="plotly_white",
                    height=400, points="outliers")
    fig_b.add_hline(y=p_seuil_vert,   line_dash="dash", line_color="#4CAF50")
    fig_b.add_hline(y=p_seuil_orange, line_dash="dash", line_color="#FF9800")
    fig_b.update_layout(showlegend=False, font=dict(family="Arial", size=11))
    st.plotly_chart(fig_b, use_container_width=True)

with t3:
    st.subheader("Scores d'anomalie")
    cc1, cc2 = st.columns(2)
    fig_if = px.scatter(x=X_u2[:, 0], y=X_u2[:, 1], color=if_sc * 100,
                         color_continuous_scale="RdYlGn_r",
                         labels={"x": "UMAP1", "y": "UMAP2", "color": "Score IF"},
                         title="Score Isolation Forest", template="plotly_white", height=420)
    fig_if.update_traces(marker=dict(size=7, opacity=0.8))
    cc1.plotly_chart(fig_if, use_container_width=True)
    fig_vae = px.scatter(x=X_u2[:, 0], y=X_u2[:, 1], color=vae_sc * 100,
                          color_continuous_scale="RdYlGn_r",
                          labels={"x": "UMAP1", "y": "UMAP2", "color": "Score VAE"},
                          title="Score VAE", template="plotly_white", height=420)
    fig_vae.update_traces(marker=dict(size=7, opacity=0.8))
    cc2.plotly_chart(fig_vae, use_container_width=True)
    fig_sc = px.scatter(x=if_sc * 100, y=vae_sc * 100, color=alertes,
                         color_discrete_map=COULEURS,
                         labels={"x": "Score IF (×100)", "y": "Score VAE (×100)"},
                         title="Corrélation IF vs VAE", template="plotly_white", height=420)
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                      line=dict(dash="dash", color="gray", width=1))
    fig_sc.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0.4, color="white")))
    st.plotly_chart(fig_sc, use_container_width=True)

with t4:
    st.subheader("Importance SHAP des variables")
    if shap_df is None:
        st.info("Cochez **Calculer SHAP** dans la barre latérale et relancez.")
    else:
        top = shap_df.head(p_shap_top)
        colors = ["#C00000" if v > 10 else "#C55A11" if v > 5 else "#2E5496"
                  for v in top["SHAP_pct"]]
        fig_s = go.Figure(go.Bar(
            x=top["SHAP_pct"], y=top["Variable"], orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in top["SHAP_pct"]], textposition="outside"))
        fig_s.update_layout(
            title=dict(text=f"Top {p_shap_top} variables", font=dict(size=14, color="#1F3864")),
            xaxis_title="Contribution (%)", yaxis=dict(autorange="reversed"),
            height=max(420, p_shap_top * 27), template="plotly_white",
            font=dict(family="Arial", size=11), margin=dict(l=10, r=80, t=50, b=40))
        st.plotly_chart(fig_s, use_container_width=True)
        st.dataframe(top[["Variable", "SHAP_pct"]].rename(
            columns={"SHAP_pct": "Contribution (%)"}),
            use_container_width=True, height=350)

with t5:
    st.subheader("Tableau des fournisseurs")
    f1, f2, f3 = st.columns(3)
    sel  = f1.multiselect("Alertes", ["🔴 Rouge", "🟠 Orange", "🟢 Vert"],
                            default=["🔴 Rouge", "🟠 Orange"])
    smin = f2.slider("Score min", 0, 100, 0)
    smax = f3.slider("Score max", 0, 100, 100)
    df_f = df_res[
        df_res["Alerte_ML"].isin(sel)
        & (df_res["Score_Risque_ML"] >= smin)
        & (df_res["Score_Risque_ML"] <= smax)
    ].sort_values("Score_Risque_ML", ascending=False)
    st.info(f"**{len(df_f)} fournisseurs** affichés sur {len(df_res)}")
    afficher = [c for c in [
        "ID_Fournisseur", "Secteur", "Region_Maroc",
        "Alerte_ML", "Score_Risque_ML", "Score_IF", "Score_VAE",
        "Cluster_HDBSCAN", "Priorite_ML", "Top_Variable_SHAP",
    ] if c in df_f.columns]
    st.dataframe(df_f[afficher], use_container_width=True, height=420)
    csv = df_res.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️  Télécharger les résultats (CSV)",
                        data=csv, file_name="resultats_alertes_fournisseurs.csv",
                        mime="text/csv", type="primary")

with t6:
    st.subheader("📐 Métriques détaillées par algorithme")
    st.markdown("Évaluation quantitative de chaque étape du pipeline ML — dans l'ordre d'exécution.")

    # ── 0. NETTOYAGE & NORMALISATION ────────────────────────────
    st.markdown("---")
    st.markdown("### 🧹 Étape 0 — Nettoyage & Normalisation des données")

    # KPIs NaN
    cv1, cv2, cv3, cv4 = st.columns(4)
    cv1.metric("Cellules manquantes",    f"{total_nan:,}",
               delta=f"{pct_nan_global} % du total",
               delta_color="inverse" if pct_nan_global > 5 else "normal")
    cv2.metric("Variables sans NaN",     vars_sans_nan,
               delta="✅ Complètes" if vars_sans_nan == len(features) else None)
    cv3.metric("Variables avec NaN",     vars_avec_nan,
               delta_color="inverse")
    cv4.metric("Stratégie d'imputation", "Médiane RobustScaler",
               delta="Robuste aux outliers")

    # Graphique NaN
    df_nan = pd.DataFrame({
        "Variable":     nan_counts.index,
        "Valeurs_NaN":  nan_counts.values,
        "Taux_NaN_pct": nan_pct.values,
    }).sort_values("Valeurs_NaN", ascending=False)

    if vars_avec_nan > 0:
        df_nan_filtre = df_nan[df_nan["Valeurs_NaN"] > 0]
        fig_nan = go.Figure(go.Bar(
            x=df_nan_filtre["Taux_NaN_pct"],
            y=df_nan_filtre["Variable"],
            orientation="h",
            marker_color=[
                "#C00000" if v > 20 else "#FF9800" if v > 5 else "#2E5496"
                for v in df_nan_filtre["Taux_NaN_pct"]
            ],
            text=[f"{v} %" for v in df_nan_filtre["Taux_NaN_pct"]],
            textposition="outside",
        ))
        fig_nan.update_layout(
            title="Taux de valeurs manquantes par variable (avant imputation)",
            xaxis_title="Taux de NaN (%)",
            xaxis_range=[0, max(df_nan_filtre["Taux_NaN_pct"]) * 1.25],
            yaxis=dict(autorange="reversed"),
            height=max(280, len(df_nan_filtre) * 28),
            template="plotly_white", font=dict(family="Arial", size=11),
            margin=dict(l=10, r=80, t=50, b=30),
        )
        fig_nan.add_vline(x=20, line_dash="dash", line_color="#C00000",
                          annotation_text="Seuil critique 20 %")
        fig_nan.add_vline(x=5,  line_dash="dot",  line_color="#FF9800",
                          annotation_text="Seuil attention 5 %")
        st.plotly_chart(fig_nan, use_container_width=True)
        st.caption("🔴 > 20 % : imputation médiane peut biaiser — envisager suppression  |  🟠 5–20 % : acceptable  |  🔵 < 5 % : négligeable")
    else:
        st.success("✅ Aucune valeur manquante — aucune imputation nécessaire.")

    st.markdown("#### Effet de la normalisation (RobustScaler)")
    st.caption("RobustScaler centre sur la médiane et échelonne par l'IQR → résistant aux valeurs extrêmes (contrairement au MinMaxScaler).")

    n_v = len(top_vars_norm)
    fig_norm = make_subplots(
        rows=n_v, cols=2,
        column_titles=["Avant normalisation (valeurs brutes)", "Après normalisation (RobustScaler)"],
        vertical_spacing=0.04,
        horizontal_spacing=0.08,
    )
    for i, var in enumerate(top_vars_norm, 1):
        fig_norm.add_trace(go.Histogram(
            x=X_brut_df[var], nbinsx=30,
            marker_color="#2E5496", opacity=0.75,
            name=var, showlegend=(i == 1),
        ), row=i, col=1)
        fig_norm.add_trace(go.Histogram(
            x=X_sc_df[var], nbinsx=30,
            marker_color="#375623", opacity=0.75,
            name=f"{var} (norm.)", showlegend=(i == 1),
        ), row=i, col=2)
        fig_norm.update_yaxes(title_text=var, row=i, col=1, title_font=dict(size=9))
    fig_norm.update_layout(
        title="Distributions avant / après RobustScaler (top 6 variables par variance)",
        height=max(500, n_v * 140),
        template="plotly_white",
        font=dict(family="Arial", size=10),
        margin=dict(l=10, r=20, t=60, b=30),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05),
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    st.markdown("#### Statistiques comparatives (avant vs après normalisation)")
    stats_before = X_brut_df[top_vars_norm].agg(["mean", "std", "min", "max"]).T.round(3)
    stats_after  = X_sc_df[top_vars_norm].agg(["mean", "std", "min", "max"]).T.round(3)
    stats_before.columns = ["Moy. brute", "Écart-type brut", "Min brut", "Max brut"]
    stats_after.columns  = ["Moy. norm.",  "Écart-type norm.",  "Min norm.",  "Max norm."]
    st.dataframe(pd.concat([stats_before, stats_after], axis=1), use_container_width=True)
    st.caption("Après normalisation : médiane ≈ 0, IQR ≈ 1. Les outliers sont préservés mais réduits en impact relatif.")

    st.markdown("#### Matrice de corrélation (données normalisées)")
    corr_mat = X_sc_df[top_vars_norm].corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.index.tolist(),
        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
        text=corr_mat.values.round(2),
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="r"),
    ))
    fig_corr.update_layout(
        title="Corrélations entre les top variables (données normalisées)",
        height=420, template="plotly_white",
        font=dict(family="Arial", size=10),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("Corrélations élevées (|r| > 0.8) → la PCA qui suit va les fusionner en composantes indépendantes.")

    # ── PCA ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔵 PCA — Réduction de dimensionnalité")
    pca_exp = pca.explained_variance_ratio_
    pca_cum = pca_exp.cumsum()
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Dimensions initiales",  X_sc.shape[1])
    p2.metric("Composantes retenues",  n_pca,
              delta=f"Réduction {(1-n_pca/X_sc.shape[1])*100:.0f}%")
    p3.metric("Variance expliquée",    f"{pca_cum[-1]*100:.1f}%",
              delta="✅ ≥ 95%" if pca_cum[-1] >= 0.95 else "⚠️ < 95%")
    p4.metric("Var. 1ère composante",  f"{pca_exp[0]*100:.1f}%")

    cp1, cp2 = st.columns(2)
    fig_scree = go.Figure()
    fig_scree.add_bar(x=list(range(1, len(pca_exp)+1)), y=pca_exp*100,
                      name="Var. individuelle", marker_color="#2E5496")
    fig_scree.add_scatter(x=list(range(1, len(pca_cum)+1)), y=pca_cum*100,
                          mode="lines+markers", name="Var. cumulée",
                          line=dict(color="#C00000", width=2), marker=dict(size=5))
    fig_scree.add_hline(y=95, line_dash="dash", line_color="#375623",
                        annotation_text="Seuil 95%")
    fig_scree.update_layout(title="Scree Plot — Variance expliquée",
                             xaxis_title="Composante", yaxis_title="Variance (%)",
                             template="plotly_white", height=360,
                             legend=dict(orientation="h", y=-0.25),
                             font=dict(family="Arial", size=11))
    cp1.plotly_chart(fig_scree, use_container_width=True)
    df_pca_tab = pd.DataFrame({
        "Composante": [f"PC{i+1}" for i in range(min(n_pca, 15))],
        "Var. individuelle (%)": (pca_exp[:15]*100).round(2),
        "Var. cumulée (%)":      (pca_cum[:15]*100).round(2),
    })
    cp2.markdown("**Variance par composante (top 15)**")
    cp2.dataframe(df_pca_tab, use_container_width=True, height=340)

    # ── UMAP ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🟣 UMAP — Projection non-linéaire")
    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Dimensions entrée",  n_pca)
    u2.metric("Dimensions sortie",  "2D / 3D")
    u3.metric("n_neighbors",        p_neighbors)
    u4.metric("min_dist",           p_min_dist)

    cu1, cu2 = st.columns(2)
    fig_dens = go.Figure(go.Histogram2dContour(
        x=X_u2[:, 0], y=X_u2[:, 1],
        colorscale="Blues", ncontours=18, showscale=True))
    fig_dens.update_layout(title="Densité de la projection UMAP 2D",
                            xaxis_title="UMAP1", yaxis_title="UMAP2",
                            template="plotly_white", height=360,
                            font=dict(family="Arial", size=11))
    cu1.plotly_chart(fig_dens, use_container_width=True)
    fig_ud = make_subplots(rows=1, cols=2,
                            subplot_titles=["Distribution UMAP1","Distribution UMAP2"])
    fig_ud.add_trace(go.Histogram(x=X_u2[:,0], nbinsx=30,
                                   marker_color="#7B2D8B", name="UMAP1"), row=1, col=1)
    fig_ud.add_trace(go.Histogram(x=X_u2[:,1], nbinsx=30,
                                   marker_color="#2E5496", name="UMAP2"), row=1, col=2)
    fig_ud.update_layout(template="plotly_white", height=360, showlegend=False,
                          font=dict(family="Arial", size=11))
    cu2.plotly_chart(fig_ud, use_container_width=True)

    # ── HDBSCAN ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🟠 HDBSCAN — Clustering")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Clusters trouvés",     n_cl)
    h2.metric("Points bruit",         n_br,
              delta=f"{n_br/len(labels)*100:.1f}% du total")
    h3.metric("Silhouette Score",     f"{sil:.3f}",
              delta="✅ > 0.5" if sil > 0.5 else ("⚠️ Moyen" if sil > 0.25 else "❌ Faible"))
    h4.metric("Davies-Bouldin Index", f"{dbi:.3f}",
              delta="✅ < 1.5" if dbi < 1.5 else ("⚠️ Moyen" if dbi < 2.5 else "❌ Élevé"))
    h5.metric("min_cluster_size",     p_min_cluster)

    ch1, ch2 = st.columns(2)
    cl_sizes = pd.Series(labels).value_counts().reset_index()
    cl_sizes.columns = ["Cluster","Taille"]
    cl_sizes["Label"]   = cl_sizes["Cluster"].apply(lambda x: "Bruit (-1)" if x==-1 else f"C{x}")
    cl_sizes["Couleur"] = cl_sizes["Cluster"].apply(lambda x: "#C00000" if x==-1 else "#2E5496")
    fig_cls = go.Figure(go.Bar(
        x=cl_sizes["Label"], y=cl_sizes["Taille"],
        marker_color=cl_sizes["Couleur"],
        text=cl_sizes["Taille"], textposition="outside"))
    fig_cls.update_layout(title="Taille de chaque cluster HDBSCAN",
                           xaxis_title="Cluster", yaxis_title="Nb fournisseurs",
                           template="plotly_white", height=360,
                           font=dict(family="Arial", size=11))
    ch1.plotly_chart(fig_cls, use_container_width=True)
    cl_sc = pd.DataFrame({"Cluster": labels, "Score": score_100})
    cl_sc["Label"] = cl_sc["Cluster"].apply(lambda x: "Bruit (-1)" if x==-1 else f"C{x}")
    cl_agg = cl_sc.groupby("Label")["Score"].agg(
        ["mean","std","min","max"]).round(2).reset_index()
    cl_agg.columns = ["Cluster","Moy.","Écart-type","Min","Max"]
    ch2.markdown("**Score composite par cluster**")
    ch2.dataframe(cl_agg, use_container_width=True, height=340)

    # Jauges
    fig_sil_g = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=sil,
        delta={"reference":0.5,"increasing":{"color":"#375623"},"decreasing":{"color":"#C00000"}},
        gauge={"axis":{"range":[-1,1]},"bar":{"color":"#2E5496"},
               "steps":[{"range":[-1,0],"color":"#ffcccc"},{"range":[0,0.25],"color":"#ffe0b2"},
                         {"range":[0.25,0.5],"color":"#fff9c4"},{"range":[0.5,1],"color":"#ccffcc"}],
               "threshold":{"line":{"color":"#C00000","width":3},"thickness":0.8,"value":0.5}},
        title={"text":"Silhouette Score<br><span style='font-size:.8em'>[-1→1 | Bon > 0.5]</span>"},
        number={"font":{"size":40}}))
    fig_sil_g.update_layout(height=300, font=dict(family="Arial", size=12))

    fig_dbi_g = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=dbi,
        delta={"reference":1.5,"increasing":{"color":"#C00000"},"decreasing":{"color":"#375623"}},
        gauge={"axis":{"range":[0,5]},"bar":{"color":"#C55A11"},
               "steps":[{"range":[0,1.5],"color":"#ccffcc"},{"range":[1.5,2.5],"color":"#fff9c4"},
                         {"range":[2.5,5],"color":"#ffcccc"}],
               "threshold":{"line":{"color":"#C00000","width":3},"thickness":0.8,"value":1.5}},
        title={"text":"Davies-Bouldin Index<br><span style='font-size:.8em'>[0→∞ | Bon < 1.5]</span>"},
        number={"font":{"size":40}}))
    fig_dbi_g.update_layout(height=300, font=dict(family="Arial", size=12))

    gg1, gg2 = st.columns(2)
    gg1.plotly_chart(fig_sil_g, use_container_width=True)
    gg2.plotly_chart(fig_dbi_g, use_container_width=True)

    # ── Isolation Forest ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔴 Isolation Forest — Détection d'anomalies")
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Estimateurs (arbres)",   200)
    i2.metric("Contamination",          f"{p_contam*100:.0f}%")
    i3.metric("Anomalies détectées",    n_anom,
              delta=f"{n_anom/len(df)*100:.1f}% du total")
    i4.metric("Score IF moyen",         f"{if_sc.mean()*100:.1f}/100")

    ci1, ci2 = st.columns(2)
    fig_ifh = go.Figure()
    fig_ifh.add_trace(go.Histogram(x=if_sc*100, nbinsx=40,
                                    marker_color="#C00000", opacity=0.75))
    fig_ifh.add_vline(x=if_sc.mean()*100, line_dash="dash", line_color="#1F3864",
                       annotation_text=f"Moy {if_sc.mean()*100:.1f}")
    fig_ifh.update_layout(title="Distribution scores Isolation Forest",
                           xaxis_title="Score IF (0–100)", yaxis_title="Fréquence",
                           template="plotly_white", height=360,
                           font=dict(family="Arial", size=11))
    ci1.plotly_chart(fig_ifh, use_container_width=True)
    df_if_cl = pd.DataFrame({
        "Label": ["Bruit (-1)" if l==-1 else f"C{l}" for l in labels],
        "Score_IF": if_sc*100})
    fig_ifb = px.box(df_if_cl, x="Label", y="Score_IF", color="Label",
                      title="Score IF par cluster HDBSCAN",
                      template="plotly_white", height=360, points="outliers")
    fig_ifb.update_layout(showlegend=False, font=dict(family="Arial", size=11))
    ci2.plotly_chart(fig_ifb, use_container_width=True)

    # ── VAE ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🟡 VAE — Variational Autoencoder")
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Dim. entrée / latente",   f"{X_sc.shape[1]} / 8")
    v2.metric("Époques",                 60)
    v3.metric("Loss reconstruction",     f"{vae_recon_loss:.4f}",
              delta="✅ Bonne" if vae_recon_loss < 0.5 else "⚠️ Élevée")
    v4.metric("Loss KL (moy.)",          f"{vae_kl_loss:.2f}")

    cv1, cv2 = st.columns(2)
    fig_vaeh = go.Figure()
    fig_vaeh.add_trace(go.Histogram(x=vae_sc*100, nbinsx=40,
                                     marker_color="#C55A11", opacity=0.75))
    fig_vaeh.add_vline(x=vae_sc.mean()*100, line_dash="dash", line_color="#1F3864",
                        annotation_text=f"Moy {vae_sc.mean()*100:.1f}")
    fig_vaeh.update_layout(title="Distribution erreurs de reconstruction VAE",
                            xaxis_title="Score VAE (0–100)", yaxis_title="Fréquence",
                            template="plotly_white", height=360,
                            font=dict(family="Arial", size=11))
    cv1.plotly_chart(fig_vaeh, use_container_width=True)
    corr_val = float(np.corrcoef(if_sc, vae_sc)[0, 1])
    fig_corr = px.scatter(
        x=if_sc*100, y=vae_sc*100,
        color=["Bruit" if l==-1 else f"C{l}" for l in labels],
        labels={"x":"Score IF (×100)","y":"Score VAE (×100)","color":"Cluster"},
        title="Corrélation IF ↔ VAE par cluster",
        template="plotly_white", height=360, opacity=0.7)
    fig_corr.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                        line=dict(dash="dash", color="gray", width=1))
    fig_corr.add_annotation(x=80, y=10, text=f"r = {corr_val:.3f}", showarrow=False,
                             font=dict(size=14, color="#1F3864"),
                             bgcolor="white", bordercolor="#1F3864", borderwidth=1)
    cv2.plotly_chart(fig_corr, use_container_width=True)

    # ── Score composite ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏆 Score composite — Synthèse")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Score moyen",  f"{score_100.mean():.1f}/100")
    s2.metric("Score médian", f"{np.median(score_100):.1f}/100")
    s3.metric("Score max",    f"{score_100.max():.1f}/100")
    s4.metric("Écart-type",   f"{score_100.std():.1f}")
    s5.metric("% Rouge",      f"{(score_100 > p_seuil_orange).mean()*100:.1f}%")

    pw1, pw2 = st.columns([1, 2])
    df_poids = pd.DataFrame({
        "Composante": ["Cluster HDBSCAN","Anomalie (IF+VAE)","Temporel","VAE seul"],
        "Poids (%)":  [f"{p_w_cl*100:.0f}%", f"{p_w_an*100:.0f}%",
                       f"{p_w_te*100:.0f}%", f"{p_w_dt*100:.0f}%"],
        "Source":     ["HDBSCAN","IF + VAE","PSI / PELT / OTD","VAE reconstruction"],
    })
    pw1.markdown("**Poids du score composite**")
    pw1.dataframe(df_poids, use_container_width=True, hide_index=True)
    fig_pw = go.Figure(go.Pie(
        labels=df_poids["Composante"],
        values=[p_w_cl, p_w_an, p_w_te, p_w_dt],
        hole=0.45,
        marker=dict(colors=["#2E5496","#C00000","#375623","#C55A11"]),
        textinfo="label+percent"))
    fig_pw.update_layout(title="Répartition des poids",
                          height=280, template="plotly_white",
                          font=dict(family="Arial", size=11),
                          showlegend=False, margin=dict(t=40,b=10))
    pw2.plotly_chart(fig_pw, use_container_width=True)

    # ── Tableau récapitulatif ─────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Récapitulatif — Qualité du pipeline")
    recap = {
        "Algorithme": ["PCA","UMAP","HDBSCAN","Isolation Forest","VAE"],
        "Rôle": ["Réduction dimensionnalité","Projection non-linéaire 2D/3D",
                 "Clustering densité","Détection anomalies (arbres)",
                 "Détection anomalies (deep learning)"],
        "Métrique clé": [
            f"Variance expliquée : {pca_cum[-1]*100:.1f}%",
            f"n_neighbors={p_neighbors}, min_dist={p_min_dist}",
            f"Silhouette={sil:.3f} | DBI={dbi:.3f}",
            f"Anomalies : {n_anom} ({n_anom/len(df)*100:.1f}%)",
            f"Recon. loss={vae_recon_loss:.4f} | KL={vae_kl_loss:.2f}",
        ],
        "Statut": [
            "✅" if pca_cum[-1] >= 0.95 else "⚠️",
            "✅",
            "✅" if sil > 0.5 and dbi < 1.5 else ("⚠️" if sil > 0.25 else "❌"),
            "✅" if n_anom > 0 else "⚠️",
            "✅" if vae_recon_loss < 0.5 else "⚠️",
        ],
    }
    st.dataframe(pd.DataFrame(recap), use_container_width=True, hide_index=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
#  RAPPORTS D'INTERPRÉTATION
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#1F3864,#2E5496);
padding:16px 24px;border-radius:10px;margin-bottom:18px'>
<h2 style='color:white;margin:0;font-size:1.35rem'>📑 Rapports d'Interprétation</h2>
<p style='color:#BDD7EE;margin:4px 0 0;font-size:0.85rem'>
Interprétation Globale du pipeline · Fiche détaillée par fournisseur</p>
</div>
""", unsafe_allow_html=True)

ri1, ri2 = st.tabs([
    "🌐 Interprétation Globale",
    "📄 Interprétation par Fournisseur",
])

# ── INTERPRÉTATION GLOBALE ────────────────────────────────────
with ri1:
    from datetime import date as _date
    _TODAY = _date.today().strftime("%d/%m/%Y")
    _pct_rouge  = n_rouge  / len(df) * 100
    _pct_orange = n_orange / len(df) * 100
    _pct_vert   = n_vert   / len(df) * 100
    _corr_iv    = float(np.corrcoef(if_sc, vae_sc)[0, 1])
    _pca_exp    = pca.explained_variance_ratio_
    _pca_cum    = _pca_exp.cumsum()

    st.markdown(f"""
    <div style='background:#F0F4FF;border:1px solid #BDD7EE;border-radius:10px;
    padding:18px 26px;margin-bottom:16px'>
    <h2 style='color:#1F3864;margin:0'>🌐 Rapport d'Interprétation Globale</h2>
    <p style='color:#595959;margin:5px 0 0;font-size:0.88rem'>
    Université Mohammed V — Rabat — FSJES Agdal &nbsp;|&nbsp; Master M.I.E.L · 2025–2026 &nbsp;|&nbsp; {_TODAY}<br>
    Dataset : <b>{len(df)} fournisseurs</b> × <b>{len(features)} variables</b></p>
    </div>""", unsafe_allow_html=True)

    # Diagnostic global
    st.markdown("### 1. Diagnostic global du portefeuille")
    if _pct_rouge > 20:
        _niv_css, _niv_lbl = "alerte-rouge",  "🔴 SITUATION CRITIQUE"
        _niv_msg = f"<b>{_pct_rouge:.1f}%</b> du portefeuille ({n_rouge} fournisseurs) est en alerte rouge. Une revue urgente s'impose."
    elif _pct_rouge > 10 or _pct_orange > 30:
        _niv_css, _niv_lbl = "alerte-orange", "🟠 SITUATION PRÉOCCUPANTE"
        _niv_msg = f"<b>{_pct_rouge:.1f}%</b> rouge et <b>{_pct_orange:.1f}%</b> orange. Des actions correctives ciblées sont nécessaires."
    else:
        _niv_css, _niv_lbl = "alerte-vert",   "🟢 SITUATION MAÎTRISÉE"
        _niv_msg = f"Seulement <b>{_pct_rouge:.1f}%</b> en rouge. Le portefeuille est globalement sous contrôle."
    st.markdown(f'<div class="{_niv_css}"><b>{_niv_lbl}</b> — {_niv_msg}</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    _gi1, _gi2, _gi3 = st.columns(3)
    _gi1.markdown(f'<div class="kpi-card" style="border-color:#4CAF50"><div class="kpi-val" style="color:#375623">{_pct_vert:.0f}%</div><div class="kpi-lab">🟢 Stables ({n_vert})</div></div>', unsafe_allow_html=True)
    _gi2.markdown(f'<div class="kpi-card" style="border-color:#FF9800"><div class="kpi-val" style="color:#C55A11">{_pct_orange:.0f}%</div><div class="kpi-lab">🟠 Surveillance ({n_orange})</div></div>', unsafe_allow_html=True)
    _gi3.markdown(f'<div class="kpi-card" style="border-color:#C00000"><div class="kpi-val" style="color:#C00000">{_pct_rouge:.0f}%</div><div class="kpi-lab">🔴 Action immédiate ({n_rouge})</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Interprétation par algorithme
    st.markdown("### 2. Interprétation par algorithme")

    with st.expander("🔵 PCA — Ce que révèle la réduction de dimensionnalité", expanded=True):
        st.markdown(f"""
**Résultat :** {X_sc.shape[1]} variables → **{n_pca} composantes** conservant **{_pca_cum[-1]*100:.1f}%** de la variance.

- {"✅ Excellent : presque toute l'information est préservée, la réduction est fiable." if _pca_cum[-1]>=0.95 else "⚠️ Variance < 95% : certaines nuances peuvent être perdues."}
- La 1ère composante explique **{_pca_exp[0]*100:.1f}%** de la variance → {"une structure dominante claire existe." if _pca_exp[0]>0.3 else "l'information est bien répartie entre plusieurs dimensions."}
- **Impact pratique :** UMAP et HDBSCAN travaillent sur un espace épuré, améliorant la qualité du clustering.
        """)

    with st.expander("🟣 UMAP — Ce que révèle la projection", expanded=True):
        st.markdown(f"""
**Résultat :** Projection en 2D/3D avec n_neighbors={p_neighbors}, min_dist={p_min_dist}.

- **n_neighbors={p_neighbors}** → {"Équilibre local/global bien calibré." if 10<=p_neighbors<=20 else "Favorise la structure globale (clusters larges)." if p_neighbors>20 else "Favorise les micro-structures locales."}
- **min_dist={p_min_dist}** → {"Clusters compacts et bien séparés visuellement." if p_min_dist<0.2 else "Clusters plus étalés, transitions douces entre groupes."}
- **Impact pratique :** Une bonne projection UMAP se traduit directement par un meilleur Silhouette Score HDBSCAN.
        """)

    with st.expander("🟠 HDBSCAN — Ce que révèle le clustering", expanded=True):
        _sil_q = "excellente" if sil>0.7 else "bonne" if sil>0.5 else "modérée" if sil>0.25 else "faible"
        _dbi_q = "très bonne" if dbi<1.0 else "bonne" if dbi<1.5 else "moyenne" if dbi<2.5 else "faible"
        st.markdown(f"""
**Résultat :** **{n_cl} cluster(s)** identifiés + **{n_br} points de bruit** ({n_br/len(df)*100:.1f}%).

**Silhouette = {sil:.3f}** → séparation {_sil_q}
- {"✅ Clusters bien définis et fiables." if sil>0.5 else "⚠️ Chevauchement partiel — interpréter avec prudence." if sil>0.25 else "❌ Clustering peu fiable. Ajustez min_cluster_size."}

**Davies-Bouldin = {dbi:.3f}** → cohésion {_dbi_q}
- {"✅ Clusters denses et bien séparés." if dbi<1.5 else "⚠️ Cohésion moyenne." if dbi<2.5 else "❌ Clusters peu cohérents. Revoir les paramètres."}

**{n_br} points de bruit** → comportements atypiques, **priorité de surveillance élevée**.
        """)

    with st.expander("🔴 Isolation Forest — Ce que révèle la détection d'anomalies", expanded=True):
        st.markdown(f"""
**Résultat :** **{n_anom} fournisseurs** ({n_anom/len(df)*100:.1f}%) détectés comme anomalies (contamination {p_contam*100:.0f}%).

- Score moyen : **{if_sc.mean()*100:.1f}/100** → {"🔴 Risque général élevé." if if_sc.mean()*100>60 else "🟠 Niveau modéré, quelques fournisseurs préoccupants." if if_sc.mean()*100>35 else "🟢 Majorité du portefeuille saine."}
- Corrélation IF ↔ VAE : **r = {_corr_iv:.3f}** → {"Les deux détecteurs convergent : anomalies robustes." if abs(_corr_iv)>0.6 else "Signaux complémentaires : IF et VAE détectent des anomalies différentes."}
- **Impact pratique :** Score IF élevé = fournisseur dont le profil s'écarte fortement de la norme.
        """)

    with st.expander("🟡 VAE — Ce que révèle l'autoencoder variationnel", expanded=True):
        st.markdown(f"""
**Résultat :** Entraîné sur {mask.sum()} fournisseurs normaux — Loss reconstruction = **{vae_recon_loss:.4f}**, KL = **{vae_kl_loss:.2f}**.

- **Loss reconstruction ({vae_recon_loss:.4f})** : {"✅ Excellente reconstruction." if vae_recon_loss<0.1 else "✅ Bonne reconstruction." if vae_recon_loss<0.5 else "⚠️ Reconstruction imparfaite — certaines anomalies peuvent être manquées."}
- **KL Divergence ({vae_kl_loss:.2f})** : {"Régularisation bien calibrée." if vae_kl_loss<10 else "Régularisation forte — espace latent très contraint."}
- **Impact pratique :** Score VAE élevé = le fournisseur ne ressemble à aucun profil connu → signal d'alerte fort.
        """)

    with st.expander("🧠 SHAP — Ce que révèle l'explicabilité", expanded=True):
        if shap_df is not None:
            _top3 = shap_df.head(3)["Variable"].tolist()
            _top3p = shap_df.head(3)["SHAP_pct"].tolist()
            st.markdown(f"""
**Variables les plus influentes :** {", ".join([f"**{v}** ({p:.1f}%)" for v,p in zip(_top3,_top3p)])}.

- **{_top3[0]}** concentre **{_top3p[0]:.1f}%** du pouvoir explicatif → variable critique à surveiller en priorité.
- Les variables SHAP positives (↑ Aggrave) sont les leviers d'action pour réduire le risque.
- **Impact pratique :** Les équipes peuvent cibler leurs audits sur les variables SHAP les plus contributives.
            """)
            _top8 = shap_df.head(8)
            _fig_sg = go.Figure(go.Bar(
                x=_top8["SHAP_pct"], y=_top8["Variable"], orientation="h",
                marker_color=["#C00000" if v>10 else "#C55A11" if v>5 else "#2E5496" for v in _top8["SHAP_pct"]],
                text=[f"{v:.1f}%" for v in _top8["SHAP_pct"]], textposition="outside"))
            _fig_sg.update_layout(title="Top 8 variables — Contribution SHAP globale",
                                   xaxis_title="Contribution (%)", yaxis=dict(autorange="reversed"),
                                   template="plotly_white", height=300,
                                   font=dict(family="Arial", size=11))
            st.plotly_chart(_fig_sg, use_container_width=True)
        else:
            st.info("Activez SHAP dans la barre latérale pour voir cette section.")

    # Score composite
    st.markdown("### 3. Interprétation du score composite")
    st.markdown(f"""
| Composante | Poids | Interprétation |
|---|---|---|
| Cluster HDBSCAN | **{p_w_cl*100:.0f}%** | Risque lié au groupe d'appartenance |
| Anomalie (IF+VAE) | **{p_w_an*100:.0f}%** | Déviance comportementale |
| Temporel (PSI/PELT/OTD) | **{p_w_te*100:.0f}%** | Dérive et instabilité dans le temps |
| VAE seul | **{p_w_dt*100:.0f}%** | Anomalie sémantique profonde |

**Score moyen : {score_100.mean():.1f}/100** → {"🔴 Risque élevé." if score_100.mean()>p_seuil_orange else "🟠 Risque modéré." if score_100.mean()>p_seuil_vert else "🟢 Risque faible."}
    """)

    # Recommandations
    st.markdown("### 4. Recommandations stratégiques")
    st.markdown(f"""
<div class="alerte-rouge">
🔴 <b>ACTIONS IMMÉDIATES — {n_rouge} fournisseurs</b><br>
Activer le plan de contingence sous 2 semaines · Identifier des fournisseurs alternatifs ·
Exiger un audit financier et opérationnel · Réduire la dépendance si &gt;30% du volume.
</div>
<div class="alerte-orange">
🟠 <b>SURVEILLANCE RENFORCÉE — {n_orange} fournisseurs</b><br>
Planifier un audit dans le mois · Reporting hebdomadaire ·
KPIs de suivi spécifiques · Préparer un plan de continuité.
</div>
<div class="alerte-vert">
🟢 <b>MONITORING STANDARD — {n_vert} fournisseurs</b><br>
Suivi mensuel habituel · Revue trimestrielle des indicateurs.
</div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Limites
    st.markdown("### 5. Limites et précautions")
    st.warning("""
⚠️ **Points de vigilance :**
- Le modèle est **non supervisé** : les scores reflètent des déviances statistiques, pas un risque avéré.
- La qualité des résultats dépend directement de la **qualité des données** d'entrée.
- Les paramètres (contamination IF, min_cluster_size) influencent les résultats — à calibrer avec les experts métier.
- Le VAE est entraîné sur les fournisseurs "normaux" : sa performance dépend de la représentativité de cet ensemble.
- Un score élevé doit déclencher une **investigation humaine**, pas une décision automatique.
    """)
    _csv_g = df_res.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Télécharger les résultats complets (CSV)",
                        data=_csv_g, file_name="rapport_global_fournisseurs.csv",
                        mime="text/csv", type="primary", use_container_width=True)

# ── INTERPRÉTATION PAR FOURNISSEUR ───────────────────────────
with ri2:
    st.markdown("""
    <div style='background:#F0F4FF;border:1px solid #BDD7EE;border-radius:10px;
    padding:18px 26px;margin-bottom:16px'>
    <h2 style='color:#1F3864;margin:0'>📄 Rapport d'Interprétation par Fournisseur</h2>
    <p style='color:#595959;margin:5px 0 0;font-size:0.88rem'>
    Sélectionnez un fournisseur pour obtenir son rapport complet d'interprétation.</p>
    </div>""", unsafe_allow_html=True)

    _opts = (df_res["ID_Fournisseur"].tolist()
             if "ID_Fournisseur" in df_res.columns else list(df_res.index))
    _id_sel = st.selectbox("🔍 Sélectionnez un fournisseur", _opts, key="sel_interp")
    _idx  = (df_res[df_res["ID_Fournisseur"] == _id_sel].index[0]
             if "ID_Fournisseur" in df_res.columns else _id_sel)
    _lig  = df_res.loc[_idx]
    _scv  = float(_lig["Score_Risque_ML"])
    _alv  = str(_lig["Alerte_ML"])
    _fid  = str(_lig.get("ID_Fournisseur", f"Index {_idx}"))
    _sect = str(_lig.get("Secteur", "N/A"))
    _reg  = str(_lig.get("Region_Maroc", "N/A"))
    _fcl  = int(_lig["Cluster_HDBSCAN"])
    _sif  = float(_lig["Score_IF"])
    _svae = float(_lig["Score_VAE"])
    _san  = float(_lig["Score_Anomalie"])
    _css  = ("alerte-rouge"  if "Rouge"  in _alv else
             "alerte-orange" if "Orange" in _alv else "alerte-vert")
    _cll  = "Anomalie / Bruit" if _fcl == -1 else f"Cluster {_fcl}"

    # En-tête
    st.markdown(f'<div class="{_css}">🏭 <b>{_fid}</b> &nbsp;|&nbsp; {_sect} &nbsp;|&nbsp; {_reg} &nbsp;|&nbsp; {_cll}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="{_css}">Alerte : <b>{_alv}</b> &nbsp;·&nbsp; Score global : <b>{_scv:.1f}/100</b></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # KPIs
    _km1, _km2, _km3, _km4 = st.columns(4)
    _km1.metric("Score global",   f"{_scv:.1f}/100",
                delta="🔴 Élevé" if _scv>p_seuil_orange else "🟠 Modéré" if _scv>p_seuil_vert else "🟢 Faible")
    _km2.metric("Score IF",       f"{_sif:.1f}/100",
                delta="↑ Anomalie" if _sif>50 else "✓ Normal")
    _km3.metric("Score VAE",      f"{_svae:.1f}/100",
                delta="↑ Anomalie" if _svae>50 else "✓ Normal")
    _km4.metric("Score Anomalie", f"{_san:.1f}/100",
                delta="↑ Combiné" if _san>50 else "✓ Normal")
    st.markdown("<br>", unsafe_allow_html=True)

    # Interprétation du score
    st.markdown("#### 📊 Interprétation du score global")
    if "Rouge" in _alv:
        st.markdown(f'<div class="alerte-rouge">🔴 <b>ALERTE ROUGE — Score {_scv:.1f}/100</b><br>Ce fournisseur présente un risque <b>critique</b> (seuil rouge = {p_seuil_orange}). Une <b>action immédiate</b> est requise.</div>', unsafe_allow_html=True)
    elif "Orange" in _alv:
        st.markdown(f'<div class="alerte-orange">🟠 <b>ALERTE ORANGE — Score {_scv:.1f}/100</b><br>Risque <b>modéré à élevé</b>. Score entre seuil vert ({p_seuil_vert}) et rouge ({p_seuil_orange}). <b>Surveillance renforcée</b> recommandée.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alerte-vert">🟢 <b>ALERTE VERTE — Score {_scv:.1f}/100</b><br>Risque <b>faible</b> (seuil vert = {p_seuil_vert}). <b>Monitoring standard</b> suffisant.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Signal par signal
    st.markdown("#### 🔬 Interprétation signal par signal")
    _ca, _cb = st.columns(2)

    _if_interp = ("🔴 Très élevé — profil fortement atypique." if _sif>70 else
                  "🟠 Élevé — comportement déviant détecté." if _sif>50 else
                  "🟡 Modéré — légère déviance par rapport à la norme." if _sif>30 else
                  "🟢 Normal — profil conforme aux fournisseurs sains.")
    _ca.markdown(f"""
**🔴 Score Isolation Forest : {_sif:.1f}/100**

{_if_interp}

Mesure à quel point le fournisseur s'**isole** dans l'espace multi-dimensionnel. Un score élevé signifie que ses caractéristiques sont rares ou inhabituelles par rapport au reste du portefeuille.
    """)

    _vae_interp = ("🔴 Très élevé — le modèle ne reconnaît pas ce profil comme normal." if _svae>70 else
                   "🟠 Élevé — reconstruction difficile, profil atypique." if _svae>50 else
                   "🟡 Modéré — légères anomalies sémantiques détectées." if _svae>30 else
                   "🟢 Normal — profil bien reconnu par le VAE.")
    _cb.markdown(f"""
**🟡 Score VAE : {_svae:.1f}/100**

{_vae_interp}

Mesure l'**erreur de reconstruction** du VAE. Un score élevé signifie que le modèle n'arrive pas à reproduire le profil de ce fournisseur — signe d'une anomalie profonde non-linéaire.
    """)

    # Signal cluster
    st.markdown("---")
    if _fcl == -1:
        st.markdown(f"""
**🟠 Appartenance au cluster : Bruit / Anomalie structurelle**

Ce fournisseur est classé comme **point de bruit** par HDBSCAN : il n'appartient à aucun groupe cohérent. Son comportement est suffisamment atypique pour ne correspondre à aucun profil connu dans le portefeuille.

→ **Recommandation :** Analyser en priorité ses caractéristiques (données manquantes, valeurs extrêmes, situation exceptionnelle).
        """)
    else:
        _cl_scores = score_100[labels == _fcl]
        _cl_mean   = _cl_scores.mean()
        _cl_rank   = float(np.sum(_cl_scores < _scv) / len(_cl_scores) * 100)
        st.markdown(f"""
**🔵 Appartenance au Cluster {_fcl}** ({len(_cl_scores)} fournisseurs, score moyen {_cl_mean:.1f}/100)

- Score personnel ({_scv:.1f}) **{"au-dessus" if _scv>_cl_mean else "en-dessous"}** de la moyenne du cluster ({_cl_mean:.1f}).
- **{_cl_rank:.0f}e percentile** dans son cluster → {"parmi les plus risqués du groupe." if _cl_rank>70 else "dans la moyenne du groupe." if _cl_rank>30 else "parmi les moins risqués du groupe."}
- {"⚠️ Ce fournisseur tire son cluster vers le haut en termes de risque." if _scv>_cl_mean+10 else "✅ Son profil est cohérent avec le reste de son cluster."}
        """)

    # SHAP
    st.markdown("---")
    st.markdown("#### 🧠 Facteurs SHAP — Pourquoi ce score ?")
    _aggrave_vars = []
    if shap_values is not None:
        _pos  = df_res.index.get_loc(_idx)
        _sv   = shap_values[_pos]
        _top5 = np.argsort(np.abs(_sv))[::-1][:5]
        _tot  = np.abs(_sv).sum() + 1e-10
        _aggrave_vars = [features[j] for j in _top5 if _sv[j] > 0]
        _reduit_vars  = [features[j] for j in _top5 if _sv[j] < 0]

        _rows = ""
        for _rk, _j in enumerate(_top5, 1):
            _ct = abs(_sv[_j]) / _tot * 100
            _di = "↑ Aggrave" if _sv[_j]>0 else "↓ Réduit"
            _dc = "#C00000" if _sv[_j]>0 else "#375623"
            _vn = features[_j]
            try:    _vb = round(float(df.loc[_idx, _vn]), 3)
            except: _vb = "N/A"
            _bw = int(_ct * 3)
            _bc = "#C00000" if _sv[_j]>0 else "#375623"
            _bg = "#FFF8F8" if _sv[_j]>0 else "#F8FFF8"
            _rows += (
                f"<tr style='background:{_bg};border-bottom:1px solid #eee'>"
                f"<td style='padding:8px;text-align:center;font-weight:700'>{_rk}</td>"
                f"<td style='padding:8px;font-weight:700;color:#1F3864'>{_vn}</td>"
                f"<td style='padding:8px;text-align:center'><code>{_vb}</code></td>"
                f"<td style='padding:8px;text-align:center;font-weight:700'>{_ct:.1f}%</td>"
                f"<td style='padding:8px;color:{_dc};font-weight:600'>{_di}</td>"
                f"<td style='padding:8px'>"
                f"<span class='shap-bar' style='width:{_bw}px;background:{_bc}'></span></td></tr>")
        st.markdown(
            "<table style='width:100%;border-collapse:collapse;font-family:Arial;font-size:13px'>"
            "<tr style='background:#1F3864;color:white'>"
            "<th style='padding:8px'>Rang</th><th>Variable</th><th>Valeur actuelle</th>"
            "<th>Contribution</th><th>Direction</th><th>Intensité</th></tr>"
            + _rows + "</table>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        _top1_var = features[_top5[0]]
        _top1_ct  = abs(_sv[_top5[0]]) / _tot * 100
        _top1_dir = "aggrave" if _sv[_top5[0]]>0 else "réduit"
        try:    _top1_val = round(float(df.loc[_idx, _top1_var]), 3)
        except: _top1_val = "N/A"

        st.info(f"""
**Lecture SHAP pour {_fid} :**

La variable **{_top1_var}** (valeur = `{_top1_val}`) est la plus déterminante : elle contribue à **{_top1_ct:.1f}%** de l'explication du score et **{_top1_dir}** le risque.

{"- Variables qui **aggravent** le risque : " + ", ".join([f"**{v}**" for v in _aggrave_vars]) if _aggrave_vars else ""}
{"- Variables qui **réduisent** le risque : " + ", ".join([f"**{v}**" for v in _reduit_vars]) if _reduit_vars else ""}

→ Pour améliorer le profil de ce fournisseur, concentrez les actions sur : **{_aggrave_vars[0] if _aggrave_vars else "les variables ci-dessus"}**.
        """)
    else:
        st.info("Activez SHAP dans la barre latérale et relancez l'analyse.")

    # Radar
    _rv = [v for v in ["OTD_Pct","Altman_ZScore","Score_ESG","Stabilite_Politique",
                        "Score_IF","Current_Ratio","Dependance_Mono"] if v in df_res.columns]
    if len(_rv) >= 3:
        st.markdown("---")
        st.markdown("#### 📡 Profil radar — Positionnement relatif")
        _ra1, _ra2 = st.columns([3, 2])
        _vn_r = []
        for _v in _rv:
            _d = df_res[_v].dropna()
            _vn_r.append(float(np.clip(
                (float(df_res.loc[_idx, _v]) - _d.min()) / (_d.max()-_d.min()+1e-10), 0, 1)*100))
        _fig_rad = go.Figure(go.Scatterpolar(
            r=_vn_r+[_vn_r[0]], theta=_rv+[_rv[0]], fill="toself",
            fillcolor="rgba(46,84,150,0.15)", line=dict(color="#2E5496", width=2)))
        _fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100])),
            title=dict(text=f"Profil normalisé — {_fid}", font=dict(color="#1F3864",size=13)),
            height=380, showlegend=False, font=dict(family="Arial",size=10))
        _ra1.plotly_chart(_fig_rad, use_container_width=True)
        _df_rad = pd.DataFrame({
            "Variable": _rv,
            "Valeur brute": [round(float(df_res.loc[_idx,_v]),2) if _v in df_res.columns else "N/A" for _v in _rv],
            "Score normalisé": [round(_v2,1) for _v2 in _vn_r],
            "Niveau": ["🔴 Élevé" if _v2>70 else "🟠 Modéré" if _v2>40 else "🟢 Faible" for _v2 in _vn_r],
        })
        _ra2.markdown("**Détail des dimensions radar**")
        _ra2.dataframe(_df_rad, use_container_width=True, hide_index=True)

    # Recommandation finale
    st.markdown("---")
    st.markdown("#### ✅ Recommandation d'action")
    _var_cible = _aggrave_vars[0] if _aggrave_vars else "variables SHAP critiques"
    if "Rouge" in _alv:
        st.markdown(f"""
<div class="alerte-rouge">
🔴 <b>ACTION IMMÉDIATE requise pour {_fid}</b><br><br>
1. Convoquer une réunion de crise avec le fournisseur sous 72h.<br>
2. Activer le plan de contingence et identifier des alternatives.<br>
3. Réduire les commandes en cours si possible.<br>
4. Exiger un audit financier et opérationnel complet dans les 2 semaines.<br>
5. KPI prioritaire à surveiller : <b>{_var_cible}</b>.
</div>""", unsafe_allow_html=True)
    elif "Orange" in _alv:
        st.markdown(f"""
<div class="alerte-orange">
🟠 <b>SURVEILLANCE RENFORCÉE pour {_fid}</b><br><br>
1. Planifier un audit de performance dans le mois.<br>
2. Passer le reporting à une fréquence hebdomadaire.<br>
3. Mettre en place des alertes sur les variables critiques.<br>
4. Préparer un plan de continuité en cas de dégradation.<br>
5. KPI prioritaire à surveiller : <b>{_var_cible}</b>.
</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="alerte-vert">
🟢 <b>MONITORING STANDARD pour {_fid}</b><br><br>
1. Maintenir le suivi mensuel habituel.<br>
2. Revue des indicateurs clés trimestriellement.<br>
3. Ce fournisseur peut servir de référence pour les autres.<br>
4. Continuer à monitorer : <b>{_rv[0] if _rv else "indicateurs habituels"}</b>.
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _csv_ind = df_res.loc[[_idx]].to_csv(index=False).encode("utf-8-sig")
    st.download_button(f"⬇️ Télécharger la fiche de {_fid} (CSV)",
                        data=_csv_ind, file_name=f"rapport_{_fid}.csv",
                        mime="text/csv", use_container_width=True)

st.markdown("---")
st.caption("Université Mohammed V — Rabat — FSJES Agdal · Master M.I.E.L · 2025–2026")

options = df_res["ID_Fournisseur"].tolist() if "ID_Fournisseur" in df_res.columns else list(df_res.index)
id_sel  = st.selectbox("Sélectionnez un fournisseur", options)
idx     = df_res[df_res["ID_Fournisseur"] == id_sel].index[0] if "ID_Fournisseur" in df_res.columns else id_sel
ligne   = df_res.loc[idx]
sc_v    = float(ligne["Score_Risque_ML"])
al_v    = str(ligne["Alerte_ML"])
fid     = ligne.get("ID_Fournisseur", f"Index {idx}")
fsect   = ligne.get("Secteur", "N/A")
freg    = ligne.get("Region_Maroc", "N/A")
fcl     = int(ligne["Cluster_HDBSCAN"])

css = "alerte-rouge" if "Rouge" in al_v else "alerte-orange" if "Orange" in al_v else "alerte-vert"
st.markdown(f'<div class="{css}">🏭 <b>{fid}</b> · {fsect} · {freg}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="{css}">Alerte : <b>{al_v}</b> · Score : <b>{sc_v:.1f}/100</b> · Cluster : <b>{"Anomalie" if fcl==-1 else f"C{fcl}"}</b></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Score de risque", f"{sc_v:.1f}/100")
r2.metric("Score IF",        f"{ligne['Score_IF']:.1f}/100")
r3.metric("Score VAE",       f"{ligne['Score_VAE']:.1f}/100")
r4.metric("Score Anomalie",  f"{ligne['Score_Anomalie']:.1f}/100")

if shap_values is not None:
    st.markdown("#### 🧠 Facteurs de risque (SHAP)")
    pos  = df_res.index.get_loc(idx)
    sv   = shap_values[pos]
    top5 = np.argsort(np.abs(sv))[::-1][:5]
    tot  = np.abs(sv).sum() + 1e-10
    rows = ""
    for rk, j in enumerate(top5, 1):
        ct  = abs(sv[j]) / tot * 100
        di  = "↑ Aggrave" if sv[j] > 0 else "↓ Réduit"
        dc  = "#C00000" if sv[j] > 0 else "#375623"
        vn  = features[j]
        try:    vb = round(float(df.loc[idx, vn]), 3)
        except: vb = "N/A"
        bw  = int(ct * 3)
        bc  = "#C00000" if sv[j] > 0 else "#375623"
        bg  = "#FFF8F8" if sv[j] > 0 else "#F8FFF8"
        rows += (
            f"<tr style='background:{bg};border-bottom:1px solid #eee'>"
            f"<td style='padding:8px;text-align:center;font-weight:700'>{rk}</td>"
            f"<td style='padding:8px;font-weight:700;color:#1F3864'>{vn}</td>"
            f"<td style='padding:8px;text-align:center'><code>{vb}</code></td>"
            f"<td style='padding:8px;text-align:center;font-weight:700'>{ct:.1f}%</td>"
            f"<td style='padding:8px;color:{dc};font-weight:600'>{di}</td>"
            f"<td style='padding:8px'>"
            f"<span class='shap-bar' style='width:{bw}px;background:{bc}'></span></td></tr>")
    st.markdown(
        "<table style='width:100%;border-collapse:collapse;font-family:Arial;font-size:13px'>"
        "<tr style='background:#1F3864;color:white'>"
        "<th style='padding:8px'>Rang</th><th>Variable</th><th>Valeur</th>"
        "<th>Contribution</th><th>Direction</th><th>Barre</th></tr>"
        + rows + "</table>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

if "Rouge" in al_v:
    rec_css = "alerte-rouge"
    rec_txt = "🔴 <b>ACTION IMMÉDIATE</b> — Activer le plan de contingence dans les 2 semaines."
elif "Orange" in al_v:
    rec_css = "alerte-orange"
    rec_txt = "🟠 <b>SURVEILLANCE RENFORCÉE</b> — Planifier un audit dans le mois."
else:
    rec_css = "alerte-vert"
    rec_txt = "🟢 <b>SURVEILLANCE STANDARD</b> — Monitoring mensuel, aucune action immédiate."

st.markdown(f'<div class="{rec_css}"><b>Recommandation :</b> {rec_txt}</div>',
            unsafe_allow_html=True)

rv = [v for v in ["OTD_Pct", "Altman_ZScore", "Score_ESG", "Stabilite_Politique",
                   "Score_IF", "Current_Ratio", "Dependance_Mono"] if v in df_res.columns]
if len(rv) >= 4:
    with st.expander("📡  Profil radar"):
        vn = []
        for v in rv:
            d = df_res[v].dropna()
            vn.append(float(np.clip(
                (float(df_res.loc[idx, v]) - d.min()) / (d.max() - d.min() + 1e-10),
                0, 1) * 100))
        fig_r = go.Figure(go.Scatterpolar(
            r=vn + [vn[0]], theta=rv + [rv[0]], fill="toself",
            fillcolor="rgba(46,84,150,0.15)", line=dict(color="#2E5496", width=2)))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=dict(text=f"Profil — {fid}", font=dict(color="#1F3864", size=13)),
            height=420, showlegend=False, font=dict(family="Arial", size=10))
        st.plotly_chart(fig_r, use_container_width=True)

st.markdown("---")
st.caption("Université Mohammed V — Rabat — FSJES Agdal · Master M.I.E.L · 2025–2026")
