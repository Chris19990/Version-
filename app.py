"""
⚽ World Cup Intelligence Platform
ML Engineer · Analyse historique FIFA 1930–2014
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from data_engine import (
    build_dataframes, build_team_stats, build_clusters,
    build_ml_model, predict_match
)

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="World Cup Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

GOLD   = "#F0C040"
BLUE   = "#378ADD"
RED    = "#E05A5A"
GREEN  = "#1D9E75"
PURPLE = "#9966CC"
BG     = "#0E1117"
CARD   = "#141824"
BORDER = "#2E3450"
TEXT   = "#C0C8D8"
TEXT_H = "#F0F2F6"
PALETTE = [GOLD, BLUE, RED, GREEN, PURPLE, "#CC8844", "#AAAACC"]

PROFIL_COLORS = {
    "🏆 Dominants":  GOLD,
    "💪 Solides":    BLUE,
    "⚡ Challengers": GREEN,
    "🌱 Émergents":  RED,
}

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT)),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(color=TEXT)),
    legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1, font=dict(color=TEXT)),
    margin=dict(t=50, b=40, l=50, r=30),
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; background:{BG}; color:{TEXT}; }}
.stApp {{ background-color:{BG}; }}
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg,#0a0d14 0%,#111827 100%);
  border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] label {{ color:{TEXT} !important; font-size:0.8rem; }}
.metric-card {{
  background: linear-gradient(135deg,{CARD} 0%,#1a1f2e 100%);
  border:1px solid {BORDER}; border-radius:12px;
  padding:18px 20px; margin-bottom:8px; position:relative; overflow:hidden;
}}
.metric-card::before {{
  content:''; position:absolute; top:0; left:0; right:0; height:2px;
}}
.metric-card.gold::before   {{ background:linear-gradient(90deg,{GOLD},transparent); }}
.metric-card.blue::before   {{ background:linear-gradient(90deg,{BLUE},transparent); }}
.metric-card.green::before  {{ background:linear-gradient(90deg,{GREEN},transparent); }}
.metric-card.red::before    {{ background:linear-gradient(90deg,{RED},transparent); }}
.metric-card.purple::before {{ background:linear-gradient(90deg,{PURPLE},transparent); }}
.metric-label {{ font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em; color:#6B7A9A; margin-bottom:5px; }}
.metric-value {{ font-size:1.8rem; font-weight:700; color:{TEXT_H}; line-height:1; }}
.metric-sub   {{ font-size:0.73rem; color:#5A6A8A; margin-top:4px; }}
.sec-header {{ border-left:3px solid {GOLD}; padding-left:12px; margin:28px 0 14px; }}
.sec-header h2 {{ color:{TEXT_H}; font-size:1.05rem; font-weight:600; margin:0; }}
.sec-header p  {{ color:#5A6A8A; font-size:0.76rem; margin:2px 0 0; }}
.nav-lbl {{ font-size:0.63rem; text-transform:uppercase; letter-spacing:0.15em; color:#4A5A7A; margin:18px 0 6px; padding-left:4px; }}
.insight {{ background:rgba(240,192,64,0.06); border:1px solid rgba(240,192,64,0.2);
            border-radius:8px; padding:12px 16px; margin:6px 0; font-size:0.81rem; color:{TEXT}; }}
.insight strong {{ color:{GOLD}; }}
.pbar-wrap {{ margin:8px 0; }}
.pbar-label {{ display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:3px; color:{TEXT}; }}
.pbar-bg {{ height:8px; border-radius:4px; background:rgba(46,52,80,0.8); overflow:hidden; }}
.pbar-fill {{ height:100%; border-radius:4px; }}
#MainMenu {{ visibility:hidden; }} footer {{ visibility:hidden; }}
.js-plotly-plot .plotly .modebar {{ background:transparent !important; }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CHARGEMENT DONNÉES
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_all():
    matches, tableau, worldcups = build_dataframes()
    team_stats = build_team_stats(tableau)
    team_stats, k_range, inertias = build_clusters(team_stats)
    clf, sc_m, X_test_s, y_test = build_ml_model(team_stats, tableau)
    return matches, tableau, worldcups, team_stats, k_range, inertias, clf, sc_m, X_test_s, y_test

with st.spinner("⚙️ Chargement des données et entraînement du modèle…"):
    matches, tableau, worldcups, team_stats, k_range, inertias, clf, sc_m, X_test_s, y_test = load_all()

ALL_TEAMS  = sorted(team_stats["Team"].tolist())
ALL_YEARS  = sorted(matches["Year"].unique().tolist())

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:22px 0 14px;">
      <div style="font-size:2.4rem;margin-bottom:6px;">⚽</div>
      <div style="font-size:0.95rem;font-weight:700;color:{TEXT_H};letter-spacing:0.05em;">
        WORLD CUP<br/><span style="color:{GOLD};">INTELLIGENCE</span>
      </div>
      <div style="font-size:0.62rem;color:#4A5A7A;margin-top:5px;letter-spacing:0.1em;">
        1930 – 2014 · ML PLATFORM
      </div>
    </div>
    <hr style="border-color:#1e2538;margin:0 0 6px;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-lbl">Navigation</div>', unsafe_allow_html=True)
    PAGES = ["🏠  Vue Générale","🌍  Analyse par Équipe","🤖  Clustering ML","🎯  Prédiction ML","⚡  Simulateur"]
    if "page" not in st.session_state:
        st.session_state["page"] = PAGES[0]
    page = st.radio("Navigation", PAGES,
                    index=PAGES.index(st.session_state["page"]),
                    key="nav_radio", label_visibility="collapsed")
    st.session_state["page"] = page

    st.markdown('<div class="nav-lbl">Filtres globaux</div>', unsafe_allow_html=True)
    year_range   = st.slider("Période", 1930, 2014, (1930, 2014), step=4)
    phase_filter = st.multiselect("Phase", ["Groupes","Élimination"],
                                  default=["Groupes","Élimination"])

    st.markdown(f"""
    <hr style="border-color:#1e2538;margin:14px 0 8px;">
    <div style="font-size:0.63rem;color:#3A4A6A;text-align:center;line-height:1.7;">
      <strong style="color:#4A5A8A;">Dataset</strong><br/>
      {len(matches)} matchs · {team_stats['Team'].nunique()} nations<br/>
      {matches['Year'].nunique()} éditions · 1930-2014
    </div>
    """, unsafe_allow_html=True)

# ── données filtrées globales
matches_f = matches[
    matches["Year"].between(*year_range) &
    matches["Phase"].isin(phase_filter)
].copy()

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def section(title, sub=""):
    st.markdown(f"""
    <div class="sec-header">
      <h2>{title}</h2>
      {"<p>"+sub+"</p>" if sub else ""}
    </div>""", unsafe_allow_html=True)

def metric_card(label, value, sub="", color="gold"):
    st.markdown(f"""
    <div class="metric-card {color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def fig_layout(fig, title="", h=None):
    kw = dict(**PLOTLY_BASE)
    if title:
        kw["title"] = dict(text=title, font=dict(color=TEXT_H, size=13), x=0.01)
    if h:
        kw["height"] = h
    fig.update_layout(**kw)
    return fig

def hex_rgba(hex_color, alpha=0.8):
    r,g,b = int(hex_color[1:3],16), int(hex_color[3:5],16), int(hex_color[5:7],16)
    return f"rgba({r},{g},{b},{alpha})"


# ══════════════════════════════════════════════════════════════
# PAGE 1 — VUE GÉNÉRALE
# ══════════════════════════════════════════════════════════════
if page == "🏠  Vue Générale":
    st.markdown(f"""
    <h1 style="color:{TEXT_H};font-size:1.55rem;font-weight:700;margin-bottom:4px;">
      Vue Générale <span style="color:{GOLD};">·</span> Analyse Historique FIFA
    </h1>
    <p style="color:#4A5A7A;font-size:0.8rem;margin-bottom:20px;">
      {year_range[0]} – {year_range[1]} · {len(matches_f)} matchs sélectionnés
    </p>""", unsafe_allow_html=True)

    # KPIs
    if len(matches_f):
        total_goals = int(matches_f["TotalGoals"].sum())
        avg_goals   = round(matches_f["TotalGoals"].mean(), 2)
        home_pct    = round((matches_f["Result"]=="HomeWin").mean()*100, 1)
        n_nations   = len(set(matches_f["HomeTeam"].tolist()+matches_f["AwayTeam"].tolist()))
        max_idx     = matches_f["TotalGoals"].idxmax()
        max_row     = matches_f.loc[max_idx]
        record      = f"{max_row['HomeTeam']} {int(max_row['HomeGoals'])}–{int(max_row['AwayGoals'])} {max_row['AwayTeam']}"
    else:
        total_goals=avg_goals=home_pct=n_nations=0; record="—"

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("Matchs disputés", f"{len(matches_f):,}", "sélection active", "gold")
    with c2: metric_card("Buts totaux",     f"{total_goals:,}",   f"moy. {avg_goals}/match", "blue")
    with c3: metric_card("Victoires dom.",  f"{home_pct}%",       "avantage terrain", "green")
    with c4: metric_card("Nations",         f"{n_nations}",       "pays représentés", "purple")
    with c5: metric_card("Record de buts",  record.split()[0] if record!="—" else "—", record, "red")

    st.markdown("<br>", unsafe_allow_html=True)

    # Buts par édition
    col1,col2 = st.columns([3,2])
    with col1:
        section("Évolution des buts par édition","Total (barres) · Moyenne/match (ligne)")
        if len(matches_f):
            ge = matches_f.groupby("Year").agg(Total=("TotalGoals","sum"),N=("Year","count")).reset_index()
            ge["Moy"] = (ge["Total"]/ge["N"]).round(2)
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Bar(x=ge["Year"],y=ge["Total"],name="Total buts",
                                 marker_color=GOLD,opacity=0.5,
                                 marker_line_color=GOLD,marker_line_width=1),secondary_y=False)
            fig.add_trace(go.Scatter(x=ge["Year"],y=ge["Moy"],name="Moy/match",
                                     mode="lines+markers",line=dict(color=RED,width=2.5),
                                     marker=dict(size=7,color=RED)),secondary_y=True)
            fig.update_yaxes(title_text="Total buts",color=GOLD,secondary_y=False,
                             gridcolor=BORDER,tickfont=dict(color=GOLD))
            fig.update_yaxes(title_text="Moy/match",color=RED,secondary_y=True,
                             tickfont=dict(color=RED))
            fig_layout(fig, h=340)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Répartition des résultats")
        if len(matches_f):
            hw = (matches_f["Result"]=="HomeWin").sum()
            aw = (matches_f["Result"]=="AwayWin").sum()
            dr = (matches_f["Result"]=="Draw").sum()
            fig2 = go.Figure(go.Pie(
                labels=["Victoire dom.","Nul","Victoire ext."],
                values=[hw,dr,aw], hole=0.55,
                marker=dict(colors=[BLUE,"#6B7A9A",RED],line=dict(color=BG,width=2)),
                textfont=dict(color=TEXT_H,size=11),
            ))
            fig2.update_layout(**PLOTLY_BASE, height=340,
                annotations=[dict(text=f"<b>{len(matches_f)}</b><br>matchs",
                                  font=dict(size=13,color=TEXT_H),showarrow=False)])
            st.plotly_chart(fig2, use_container_width=True)

    # Distribution & phases
    col3,col4 = st.columns(2)
    with col3:
        section("Distribution des buts par match")
        if len(matches_f):
            sf = matches_f["TotalGoals"].value_counts().sort_index()
            fig3 = go.Figure(go.Bar(
                x=sf.index, y=sf.values,
                marker=dict(color=sf.values,
                            colorscale=[[0,"#2a3050"],[0.5,BLUE],[1,GOLD]],
                            line=dict(color=BG,width=0.8)),
                text=sf.values, textposition="outside",
                textfont=dict(color=TEXT,size=10)
            ))
            fig_layout(fig3, h=300)
            fig3.update_layout(xaxis_title="Buts/match", yaxis_title="Nb matchs")
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("Groupes vs Élimination directe")
        if len(matches_f):
            fig4 = go.Figure()
            for ph,col_c in [("Groupes",GOLD),("Élimination",RED)]:
                d = matches_f[matches_f["Phase"]==ph]["TotalGoals"]
                if len(d):
                    freq = d.value_counts().sort_index()
                    fig4.add_trace(go.Bar(x=freq.index,y=freq.values,
                                         name=f"{ph} (moy={d.mean():.2f})",
                                         marker_color=col_c,opacity=0.6,
                                         marker_line_color=BG,marker_line_width=0.8))
            fig4.update_layout(barmode="group",**PLOTLY_BASE,height=300,
                               xaxis_title="Buts",yaxis_title="Fréquence")
            st.plotly_chart(fig4, use_container_width=True)

    # Affluence
    section("Affluence & expansion du tournoi","Spectateurs totaux et équipes qualifiées par édition")
    wc_f = worldcups[worldcups["Year"].between(*year_range)].copy()
    wc_f["Attendance"] = pd.to_numeric(wc_f["Attendance"], errors="coerce")
    if len(wc_f):
        fig5 = make_subplots(specs=[[{"secondary_y":True}]])
        fig5.add_trace(go.Scatter(x=wc_f["Year"],y=wc_f["Attendance"],name="Affluence",
                                  mode="lines+markers",fill="tozeroy",
                                  fillcolor="rgba(55,138,221,0.10)",
                                  line=dict(color=BLUE,width=2.5),
                                  marker=dict(size=7)), secondary_y=False)
        fig5.add_trace(go.Scatter(x=wc_f["Year"],y=wc_f["QualifiedTeams"],name="Équipes qual.",
                                  mode="lines+markers",
                                  line=dict(color=GOLD,width=2,dash="dash"),
                                  marker=dict(size=7,symbol="diamond")), secondary_y=True)
        fig5.update_yaxes(title_text="Affluence",color=BLUE,secondary_y=False,
                          gridcolor=BORDER,tickfont=dict(color=BLUE))
        fig5.update_yaxes(title_text="Équipes",color=GOLD,secondary_y=True,
                          tickfont=dict(color=GOLD))
        fig_layout(fig5, h=280)
        st.plotly_chart(fig5, use_container_width=True)

    # Palmarès vainqueurs
    section("Palmarès — Vainqueurs de la Coupe du Monde")
    wc_filt = worldcups[worldcups["Year"].between(*year_range)].copy()
    winners = wc_filt["Winner"].value_counts().reset_index()
    winners.columns = ["Pays","Titres"]
    fig6 = go.Figure(go.Bar(
        y=winners["Pays"][::-1], x=winners["Titres"][::-1],
        orientation="h",
        marker=dict(color=winners["Titres"][::-1],
                    colorscale=[[0,"#2a3050"],[0.5,BLUE],[1,GOLD]],
                    line=dict(color=BG,width=0.5)),
        text=winners["Titres"][::-1], textposition="outside",
        textfont=dict(color=TEXT_H,size=11)
    ))
    fig_layout(fig6, h=max(250, len(winners)*32))
    fig6.update_layout(xaxis_title="Nombre de titres", yaxis_title="")
    st.plotly_chart(fig6, use_container_width=True)

    # Insights
    section("Insights Clés")
    i1,i2,i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="insight"><strong>📉 Déclin offensif</strong><br/>
        La moyenne buts/match passe de <strong>5.4</strong> en 1954 à <strong>2.2</strong>
        en 1990 — densification tactique progressive.</div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="insight"><strong>🏠 Avantage domicile</strong><br/>
        ~55% des matchs remportés par l'équipe désignée "domicile"
        vs ~27% pour l'extérieur.</div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="insight"><strong>⚡ Phase éliminatoire</strong><br/>
        Les K.O. produisent <strong>moins de buts</strong> que les phases de groupes
        — pression tactique et défense accrue.</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE PAR ÉQUIPE
# ══════════════════════════════════════════════════════════════
elif page == "🌍  Analyse par Équipe":
    st.markdown(f"""
    <h1 style="color:{TEXT_H};font-size:1.55rem;font-weight:700;margin-bottom:4px;">
      Analyse par Équipe <span style="color:{GOLD};">·</span> Ranking & Profils
    </h1>""", unsafe_allow_html=True)

    # ── Filtre année propre à cette page
    years_available = sorted(matches["Year"].unique().tolist())
    col_yr1, col_yr2 = st.columns([3,1])
    with col_yr1:
        yr_team = st.select_slider(
            "Filtrer par période (page Équipe)",
            options=years_available,
            value=(years_available[0], years_available[-1]),
            key="yr_team"
        )
    with col_yr2:
        st.markdown("<br>", unsafe_allow_html=True)
        ph_team = st.multiselect("Phase", ["Groupes","Élimination"],
                                 default=["Groupes","Élimination"], key="ph_team")

    # Recalcul stats sur le sous-ensemble année
    tableau_f = tableau[
        tableau["Year"].between(*yr_team) &
        tableau["Phase"].isin(ph_team if ph_team else ["Groupes","Élimination"])
    ].copy()
    ts_page = build_team_stats(tableau_f) if len(tableau_f) else team_stats.copy()

    # Ajouter profil depuis team_stats global
    profil_map = team_stats.set_index("Team")["Profil"].to_dict()
    ts_page["Profil"] = ts_page["Team"].map(profil_map).fillna("🌱 Émergents")

    st.markdown(f"<p style='color:#4A5A7A;font-size:0.8rem;margin-bottom:20px;'>"
                f"{ts_page['Team'].nunique()} nations · {yr_team[0]}–{yr_team[1]}</p>",
                unsafe_allow_html=True)

    # Top 15 victoires & buts
    col_l,col_r = st.columns(2)
    with col_l:
        section("Top 15 — Victoires")
        top15 = ts_page.nlargest(15,"Victoires").reset_index(drop=True)
        vals = top15["Victoires"].values[::-1]
        nms  = top15["Team"].values[::-1]
        fig = go.Figure(go.Bar(
            y=nms, x=vals, orientation="h",
            marker=dict(color=vals, colorscale=[[0,"#2a3050"],[0.5,BLUE],[1,GOLD]],
                        line=dict(color=BG,width=0.5)),
            text=vals, textposition="outside", textfont=dict(color=TEXT_H,size=10)
        ))
        fig_layout(fig, h=430)
        fig.update_layout(xaxis_title="Victoires")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section("Top 15 — Buts marqués")
        top15b = ts_page.nlargest(15,"Buts_marques").reset_index(drop=True)
        vals2 = top15b["Buts_marques"].values[::-1]
        nms2  = top15b["Team"].values[::-1]
        fig2 = go.Figure(go.Bar(
            y=nms2, x=vals2, orientation="h",
            marker=dict(color=vals2, colorscale=[[0,"#1a2a3a"],[0.5,GREEN],[1,GOLD]],
                        line=dict(color=BG,width=0.5)),
            text=vals2, textposition="outside", textfont=dict(color=TEXT_H,size=10)
        ))
        fig_layout(fig2, h=430)
        fig2.update_layout(xaxis_title="Buts marqués")
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter win rate vs buts
    section("Profil offensif vs Win rate","Taille = matchs joués · Couleur = profil de cluster")
    ts_sc = ts_page[ts_page["Matchs"] >= 5].copy()
    fig3 = go.Figure()
    for profil in sorted(ts_sc["Profil"].unique()):
        sub = ts_sc[ts_sc["Profil"]==profil]
        c   = PROFIL_COLORS.get(profil, PURPLE)
        fig3.add_trace(go.Scatter(
            x=sub["Buts_pm"], y=sub["Win_rate"],
            mode="markers+text", name=profil,
            marker=dict(size=np.clip(np.sqrt(sub["Matchs"])*4,7,30),
                        color=hex_rgba(c,0.75),
                        line=dict(color=BG,width=1)),
            text=sub.apply(lambda r: r["Team"] if r["Matchs"]>=20 else "", axis=1),
            textposition="top center", textfont=dict(size=9,color=TEXT),
            hovertemplate="<b>%{customdata[0]}</b><br>Win rate:%{y:.1f}%<br>Buts/m:%{x:.2f}<br>Matchs:%{customdata[1]}<extra></extra>",
            customdata=sub[["Team","Matchs"]].values
        ))
    if len(ts_sc):
        fig3.add_hline(y=ts_sc["Win_rate"].mean(),line_dash="dash",line_color="#444466",line_width=1,opacity=0.5)
        fig3.add_vline(x=ts_sc["Buts_pm"].mean(), line_dash="dash",line_color="#444466",line_width=1,opacity=0.5)
    fig_layout(fig3, h=460)
    fig3.update_layout(xaxis_title="Buts marqués / match", yaxis_title="Win rate (%)")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Radar comparatif ─────────────────────────────────────
    section("Radar — Comparaison multi-nations","Stats normalisées 0–1 sur toutes les équipes")
    default_t = [t for t in ["Brazil","Germany","Italy","Argentina","France"] if t in ALL_TEAMS]
    sel_teams = st.multiselect("Équipes à comparer (2–6)", ALL_TEAMS,
                               default=default_t[:5], max_selections=6, key="radar_sel")

    if len(sel_teams) >= 2:
        ts_r = ts_page.copy()
        # normalisation robuste : si une seule valeur unique, mettre 0.5
        def norm_col(col):
            mn, mx = ts_r[col].min(), ts_r[col].max()
            if mx == mn:
                return pd.Series(0.5, index=ts_r.index)
            return (ts_r[col] - mn) / (mx - mn)

        ts_r["n_wr"]  = norm_col("Win_rate")
        ts_r["n_bpm"] = norm_col("Buts_pm")
        ts_r["n_def"] = 1 - norm_col("Buts_encais_pm")
        ts_r["n_lon"] = norm_col("Matchs")
        ts_r["n_eff"] = norm_col("Diff_buts")
        cats = ["Win rate","Att. (buts/m)","Solidité déf.","Longévité","Efficacité"]

        fig_r = go.Figure()
        for team, color in zip(sel_teams, PALETTE):
            row = ts_r[ts_r["Team"]==team]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            vals = [float(row["n_wr"]), float(row["n_bpm"]), float(row["n_def"]),
                    float(row["n_lon"]), float(row["n_eff"])]
            vals_c = vals + [vals[0]]
            cats_c = cats + [cats[0]]
            r_v,g_v,b_v = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
            fig_r.add_trace(go.Scatterpolar(
                r=vals_c, theta=cats_c, fill="toself", name=team,
                line=dict(color=color, width=2),
                fillcolor=f"rgba({r_v},{g_v},{b_v},0.12)",
            ))
        fig_r.update_layout(
            polar=dict(
                bgcolor=CARD,
                radialaxis=dict(visible=True, range=[0,1], color=TEXT,
                                gridcolor=BORDER, tickfont=dict(color="#5A6A8A"),
                                tickvals=[0,0.25,0.5,0.75,1]),
                angularaxis=dict(color=TEXT, gridcolor=BORDER,
                                 tickfont=dict(color=TEXT_H, size=11))
            ),
            **PLOTLY_BASE, height=460
        )
        st.plotly_chart(fig_r, use_container_width=True)
    elif len(sel_teams) == 1:
        st.info("Sélectionnez au moins 2 équipes pour afficher le radar.")

    # ── Fiche nation ─────────────────────────────────────────
    section("Fiche Nation","Statistiques détaillées + historique buts")
    team_sel = st.selectbox("Choisir une nation", ALL_TEAMS,
                            index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0,
                            key="fiche_team")
    row_t = ts_page[ts_page["Team"]==team_sel]
    if len(row_t):
        row_t = row_t.iloc[0]
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: metric_card("Matchs",         int(row_t["Matchs"]),      "", "gold")
        with c2: metric_card("Victoires",       int(row_t["Victoires"]),   f"Win rate {row_t['Win_rate']}%", "green")
        with c3: metric_card("Buts marqués",    int(row_t["Buts_marques"]),f"{row_t['Buts_pm']}/match", "blue")
        with c4: metric_card("Buts encaissés",  int(row_t["Buts_encais"]), f"{row_t['Buts_encais_pm']}/match", "red")
        with c5: metric_card("Diff. buts",      f"{int(row_t['Diff_buts']):+d}",
                             f"{int(row_t['Editions'])} éditions", "purple")
    else:
        st.warning(f"{team_sel} absent des données pour cette période.")

    matches_team = matches[
        (matches["Year"].between(*yr_team)) &
        ((matches["HomeTeam"]==team_sel)|(matches["AwayTeam"]==team_sel))
    ].copy()
    if len(matches_team):
        st.markdown("<br>", unsafe_allow_html=True)
        scored_list, conceded_list, yrs_list = [], [], []
        for yr, grp in matches_team.groupby("Year"):
            s = grp.apply(lambda r: r["HomeGoals"] if r["HomeTeam"]==team_sel else r["AwayGoals"],axis=1).sum()
            c_v = grp.apply(lambda r: r["AwayGoals"] if r["HomeTeam"]==team_sel else r["HomeGoals"],axis=1).sum()
            scored_list.append(int(s)); conceded_list.append(int(c_v)); yrs_list.append(yr)
        fig5 = go.Figure()
        fig5.add_trace(go.Bar(x=yrs_list, y=scored_list,   name="Buts marqués",   marker_color=GREEN, opacity=0.85))
        fig5.add_trace(go.Bar(x=yrs_list, y=[-v for v in conceded_list], name="Buts encaissés", marker_color=RED, opacity=0.85))
        fig_layout(fig5, f"Bilan buts de {team_sel} par édition", h=270)
        fig5.update_layout(barmode="relative", yaxis_title="Buts (+ marqués / − encaissés)",
                           xaxis=dict(tickvals=yrs_list, gridcolor=BORDER))
        st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — CLUSTERING ML
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Clustering ML":
    st.markdown(f"""
    <h1 style="color:{TEXT_H};font-size:1.55rem;font-weight:700;margin-bottom:4px;">
      Clustering ML <span style="color:{GOLD};">·</span> K-Means · Profils de Jeu
    </h1>
    <p style="color:#4A5A7A;font-size:0.8rem;margin-bottom:20px;">
      Segmentation non supervisée · 5 features · k=4 clusters
    </p>""", unsafe_allow_html=True)

    # Méthode du coude
    col1,col2 = st.columns([3,2])
    with col1:
        section("Méthode du coude","Inertie WCSS selon le nombre de clusters")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range), y=inertias, mode="lines+markers",
            line=dict(color=GOLD,width=2.5),
            marker=dict(size=9,color=GOLD,line=dict(color=BG,width=1.5)),
            name="Inertie WCSS"
        ))
        idx4 = list(k_range).index(4)
        fig.add_vline(x=4, line_dash="dash", line_color=RED, line_width=1.5, opacity=0.7)
        fig.add_annotation(x=4, y=inertias[idx4], text="  k=4 optimal",
                           font=dict(color=RED,size=11), showarrow=False, xanchor="left")
        fig_layout(fig, h=320)
        fig.update_layout(xaxis_title="k (clusters)", yaxis_title="Inertie WCSS",
                          xaxis=dict(tickmode="linear",tick0=2,dtick=1,gridcolor=BORDER))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Répartition des profils")
        pc = team_stats["Profil"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=pc.index, values=pc.values, hole=0.52,
            marker=dict(colors=[PROFIL_COLORS.get(p,PURPLE) for p in pc.index],
                        line=dict(color=BG,width=2)),
            textfont=dict(color=TEXT_H,size=10),
        ))
        fig2.update_layout(**PLOTLY_BASE, height=320,
            annotations=[dict(text=f"<b>{len(team_stats)}</b><br>nations",
                              font=dict(size=12,color=TEXT_H),showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    # Scatters clusters
    col3,col4 = st.columns(2)
    with col3:
        section("Win rate vs Offensivité par cluster")
        fig3 = go.Figure()
        for profil in sorted(team_stats["Profil"].unique()):
            sub = team_stats[team_stats["Profil"]==profil]
            c   = PROFIL_COLORS.get(profil, PURPLE)
            fig3.add_trace(go.Scatter(
                x=sub["Buts_pm"], y=sub["Win_rate"], mode="markers", name=profil,
                marker=dict(size=np.clip(np.sqrt(sub["Matchs"])*3.5,6,25),
                            color=hex_rgba(c,0.8), line=dict(color=BG,width=0.5)),
                hovertemplate="<b>%{customdata}</b><br>Win rate:%{y:.1f}%<br>Buts/m:%{x:.2f}<extra></extra>",
                customdata=sub["Team"].values
            ))
        fig_layout(fig3, h=360)
        fig3.update_layout(xaxis_title="Buts marqués / match", yaxis_title="Win rate (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section("Attaque vs Défense par cluster")
        fig4 = go.Figure()
        for profil in sorted(team_stats["Profil"].unique()):
            sub = team_stats[team_stats["Profil"]==profil]
            c   = PROFIL_COLORS.get(profil, PURPLE)
            fig4.add_trace(go.Scatter(
                x=sub["Buts_encais_pm"], y=sub["Buts_pm"], mode="markers", name=profil,
                marker=dict(size=np.clip(np.sqrt(sub["Matchs"])*3.5,6,25),
                            color=hex_rgba(c,0.8), line=dict(color=BG,width=0.5)),
                hovertemplate="<b>%{customdata}</b><br>Enc:/m:%{x:.2f}<br>Buts/m:%{y:.2f}<extra></extra>",
                customdata=sub["Team"].values
            ))
        mv = max(team_stats["Buts_encais_pm"].max(), team_stats["Buts_pm"].max())
        fig4.add_trace(go.Scatter(x=[0,mv],y=[0,mv],mode="lines",
                                  line=dict(color="#444466",dash="dash",width=1),
                                  name="Équilibre",showlegend=True))
        fig_layout(fig4, h=360)
        fig4.update_layout(xaxis_title="Buts encaissés / match", yaxis_title="Buts marqués / match")
        st.plotly_chart(fig4, use_container_width=True)

    # Équipes par cluster
    section("Équipes par profil de jeu")
    profils_order = ["🏆 Dominants","💪 Solides","⚡ Challengers","🌱 Émergents"]
    p_cols = st.columns(4)
    for col, profil in zip(p_cols, profils_order):
        with col:
            sub = sorted(team_stats[team_stats["Profil"]==profil]["Team"].tolist())
            c   = PROFIL_COLORS.get(profil, PURPLE)
            r_v,g_v,b_v = int(c[1:3],16),int(c[3:5],16),int(c[5:7],16)
            t_html = "".join([
                f'<div style="padding:3px 0;border-bottom:1px solid {BORDER};font-size:0.76rem;color:{TEXT};">{t}</div>'
                for t in sub
            ])
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-top:2px solid {c};
                        border-radius:10px;padding:13px;">
              <div style="font-size:0.83rem;font-weight:600;color:{c};margin-bottom:9px;">
                {profil}
                <span style="background:rgba({r_v},{g_v},{b_v},0.15);color:{c};
                      padding:2px 8px;border-radius:10px;font-size:0.68rem;">{len(sub)}</span>
              </div>
              {t_html}
            </div>""", unsafe_allow_html=True)

    # Statistiques moyennes par cluster — tableau propre sans gradient bugué
    section("Statistiques moyennes par cluster")
    cm_df = (
        team_stats.groupby("Profil")[["Win_rate","Buts_pm","Buts_encais_pm","Diff_buts","Matchs"]]
        .mean().round(2)
        .rename(columns={"Win_rate":"Win rate %","Buts_pm":"Buts/m",
                         "Buts_encais_pm":"Buts enc./m","Diff_buts":"Diff. buts","Matchs":"Matchs moy."})
        .reset_index()
    )
    # Affichage via Plotly table pour éviter les bugs Styler
    header_vals = list(cm_df.columns)
    cell_vals   = [cm_df[col].tolist() for col in cm_df.columns]
    fig_t = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in header_vals],
            fill_color="#1a1f2e", font=dict(color=GOLD, size=12),
            align="center", line_color=BORDER, height=32
        ),
        cells=dict(
            values=cell_vals,
            fill_color=[[CARD if i%2==0 else "#161b28" for i in range(len(cm_df))]
                        for _ in cell_vals],
            font=dict(color=TEXT_H, size=11),
            align=["left"] + ["center"]*(len(header_vals)-1),
            line_color=BORDER, height=28
        )
    ))
    fig_t.update_layout(**{**PLOTLY_BASE, "margin": dict(t=10,b=10,l=10,r=10)}, height=40+len(cm_df)*30+32)
    st.plotly_chart(fig_t, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION ML
# ══════════════════════════════════════════════════════════════
elif page == "🎯  Prédiction ML":
    st.markdown(f"""
    <h1 style="color:{TEXT_H};font-size:1.55rem;font-weight:700;margin-bottom:4px;">
      Prédiction ML <span style="color:{GOLD};">·</span> Régression Logistique
    </h1>
    <p style="color:#4A5A7A;font-size:0.8rem;margin-bottom:20px;">
      Modèle entraîné sur 80% des données · évalué sur 20%
    </p>""", unsafe_allow_html=True)

    feat_labels = [
        "Win rate (équipe)","Buts/match (éq.)","Buts enc./m (éq.)","Diff. buts (éq.)",
        "Win rate (adv.)","Buts/match (adv.)","Buts enc./m (adv.)","Diff. buts (adv.)"
    ]

    y_pred = clf.predict(X_test_s)
    acc    = round(clf.score(X_test_s, y_test)*100, 1)

    # KPIs
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1   = round(f1_score(y_test, y_pred, average="macro")*100, 1)
    prec = round(precision_score(y_test, y_pred, average="macro", zero_division=0)*100, 1)
    rec  = round(recall_score(y_test, y_pred, average="macro")*100, 1)

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Accuracy",          f"{acc}%",   "Régression Logistique", "gold")
    with c2: metric_card("F1-score macro",     f"{f1}%",    "équilibré par classe", "blue")
    with c3: metric_card("Précision macro",    f"{prec}%",  "", "green")
    with c4: metric_card("Rappel macro",       f"{rec}%",   "", "purple")

    st.markdown("<br>", unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    # Matrice de confusion
    with col1:
        section("Matrice de confusion","Prédit vs Réel")
        cm = confusion_matrix(y_test, y_pred)
        labels_cm = ["Défaite","Nul","Victoire"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=labels_cm, y=labels_cm,
            colorscale=[[0,"#1a1f2e"],[0.5,BLUE],[1,GOLD]],
            text=cm, texttemplate="<b>%{text}</b>",
            textfont=dict(size=18,color=TEXT_H), showscale=True,
            colorbar=dict(tickfont=dict(color=TEXT),bgcolor=CARD)
        ))
        fig_layout(fig_cm, h=340)
        fig_cm.update_layout(
            xaxis_title="Prédit", yaxis_title="Réel",
            yaxis=dict(autorange="reversed", gridcolor=BORDER),
            xaxis=dict(gridcolor=BORDER)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # Importance features
    with col2:
        section("Importance des features","Coefficients absolus moyens (LogReg)")
        coef = np.abs(clf.coef_).mean(axis=0)
        idx  = np.argsort(coef)
        bar_colors = [GOLD if i >= 4 else BLUE for i in idx]
        fig_imp = go.Figure(go.Bar(
            y=[feat_labels[i] for i in idx], x=coef[idx], orientation="h",
            marker=dict(color=bar_colors, line=dict(color=BG,width=0.5)),
            text=[f"{v:.3f}" for v in coef[idx]],
            textposition="outside", textfont=dict(color=TEXT,size=9)
        ))
        fig_layout(fig_imp, h=340)
        fig_imp.update_layout(xaxis_title="Importance absolue (|coeff.|)")
        # légende manuelle
        fig_imp.add_trace(go.Bar(x=[0],y=[""],marker_color=GOLD,name="Stats adversaire",
                                 orientation="h",showlegend=True))
        fig_imp.add_trace(go.Bar(x=[0],y=[""],marker_color=BLUE,name="Stats équipe",
                                 orientation="h",showlegend=True))
        st.plotly_chart(fig_imp, use_container_width=True)

    # Rapport de classification — Plotly table
    section("Rapport de classification détaillé")
    report = classification_report(y_test, y_pred,
                                   target_names=["Défaite","Nul","Victoire"],
                                   output_dict=True)
    rp = pd.DataFrame(report).T.round(3).reset_index()
    rp.columns = ["Classe","Précision","Rappel","F1-score","Support"]
    # Garder seulement les lignes utiles
    rp = rp[rp["Classe"].isin(["Défaite","Nul","Victoire","macro avg","weighted avg"])]

    def cell_color(val, col):
        if col not in ["Précision","Rappel","F1-score"]: return "#161b28"
        try:
            v = float(val)
            if v >= 0.6:  return "#1a3a2a"
            elif v >= 0.45: return "#2a2a1a"
            else:           return "#3a1a1a"
        except:
            return "#161b28"

    fig_rep = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{c}</b>" for c in rp.columns],
            fill_color="#1a1f2e", font=dict(color=GOLD,size=12),
            align="center", line_color=BORDER, height=32
        ),
        cells=dict(
            values=[rp[c].tolist() for c in rp.columns],
            fill_color=[
                ["#1a1f2e" if r["Classe"] in ["macro avg","weighted avg"] else CARD
                 for _,r in rp.iterrows()]
                for _ in rp.columns
            ],
            font=dict(color=TEXT_H, size=11),
            align=["left","center","center","center","center"],
            line_color=BORDER, height=30
        )
    ))
    fig_rep.update_layout(**{**PLOTLY_BASE, "margin": dict(t=10,b=10,l=10,r=10)}, height=40+len(rp)*32)
    st.plotly_chart(fig_rep, use_container_width=True)

    # Insights
    section("Insights Modèle")
    i1,i2,i3 = st.columns(3)
    with i1:
        st.markdown(f"""<div class="insight"><strong>🎯 Accuracy {acc}%</strong><br/>
        Cohérent pour un sport aléatoire. Même les meilleurs modèles peinent à dépasser
        <strong>60%</strong> sur des données historiques agrégées.</div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="insight"><strong>🔑 Features clés</strong><br/>
        Le <strong>Win rate historique</strong> et la <strong>différence de buts</strong>
        sont les prédicteurs les plus puissants des deux côtés.</div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="insight"><strong>⚠️ Limites</strong><br/>
        Pas d'effectifs individuels, ni blessés, ni contexte situationnel
        (stade, météo, enjeu). Le modèle capture la tendance historique globale.</div>""",
        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — SIMULATEUR
# ══════════════════════════════════════════════════════════════
elif page == "⚡  Simulateur":
    st.markdown(f"""
    <h1 style="color:{TEXT_H};font-size:1.55rem;font-weight:700;margin-bottom:4px;">
      Simulateur <span style="color:{GOLD};">·</span> Pronostics & Quarts de finale
    </h1>
    <p style="color:#4A5A7A;font-size:0.8rem;margin-bottom:20px;">
      Modèle Régression Logistique · stats historiques 1930–2014
    </p>""", unsafe_allow_html=True)

    # ── Match à la demande ────────────────────────────────────
    section("Simuler un match")
    col_a, col_vs, col_b = st.columns([5,1,5])
    with col_a:
        team_a = st.selectbox("Équipe A", ALL_TEAMS,
                              index=ALL_TEAMS.index("Brazil") if "Brazil" in ALL_TEAMS else 0,
                              key="sim_a")
    with col_vs:
        st.markdown(f"""<div style="text-align:center;padding-top:30px;
                    font-size:1.3rem;font-weight:700;color:{GOLD};">VS</div>""",
                    unsafe_allow_html=True)
    with col_b:
        idx_b = ALL_TEAMS.index("Germany") if "Germany" in ALL_TEAMS else 1
        team_b = st.selectbox("Équipe B", ALL_TEAMS, index=idx_b, key="sim_b")

    if team_a == team_b:
        st.warning("⚠️ Sélectionnez deux équipes différentes.")
    else:
        res = predict_match(team_a, team_b, team_stats, clf, sc_m)
        if res:
            p_a, p_d, p_b = res["p_win_a"], res["p_draw"], res["p_win_b"]
            probs = [p_a, p_d, p_b]
            winner_idx = probs.index(max(probs))
            winner = [team_a, "Match nul", team_b][winner_idx]
            w_color = GOLD if winner_idx != 1 else "#6B7A9A"

            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{CARD},#1a1f2e);
                        border:1px solid {BORDER};border-top:2px solid {GOLD};
                        border-radius:14px;padding:26px 32px;text-align:center;margin:16px 0;">
              <div style="font-size:0.68rem;text-transform:uppercase;letter-spacing:0.15em;
                          color:#4A5A7A;margin-bottom:10px;">Pronostic</div>
              <div style="font-size:2rem;font-weight:800;color:{w_color};">
                {"🏆 " if winner_idx!=1 else "🤝 "}{winner}
              </div>
              <div style="font-size:0.78rem;color:#4A5A7A;margin-top:7px;">
                Régression Logistique · probabilités multi-classes
              </div>
            </div>""", unsafe_allow_html=True)

            col_p1, col_p2 = st.columns([2,3])
            with col_p1:
                for label, prob, color in [
                    (f"Victoire {team_a}", p_a, GREEN),
                    ("Match nul",          p_d, "#6B7A9A"),
                    (f"Victoire {team_b}", p_b, RED)
                ]:
                    pct = prob*100
                    st.markdown(f"""
                    <div class="pbar-wrap">
                      <div class="pbar-label">
                        <span>{label}</span>
                        <span style="color:{color};font-weight:600;">{pct:.1f}%</span>
                      </div>
                      <div class="pbar-bg">
                        <div class="pbar-fill" style="width:{pct}%;background:{color};"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            with col_p2:
                bar_c = [GREEN if i==winner_idx else (RED if i==2 else "#6B7A9A") for i in range(3)]
                if winner_idx == 2: bar_c[0] = BLUE
                fig_bar = go.Figure(go.Bar(
                    x=[team_a,"Nul",team_b], y=probs,
                    marker=dict(color=bar_c, line=dict(color=BG,width=1)),
                    text=[f"{v*100:.1f}%" for v in probs],
                    textposition="outside", textfont=dict(size=13,color=TEXT_H)
                ))
                fig_layout(fig_bar, h=280)
                fig_bar.update_layout(
                    yaxis=dict(tickformat=".0%", range=[0,max(probs)*1.3], gridcolor=BORDER),
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # Fiche comparative
            section("Comparaison des profils")
            ra = team_stats[team_stats["Team"]==team_a].iloc[0]
            rb = team_stats[team_stats["Team"]==team_b].iloc[0]
            comp = {
                "Statistique":["Win rate","Buts/match","Buts enc./match","Diff. buts","Matchs","Victoires"],
                team_a:[f"{ra['Win_rate']}%", ra['Buts_pm'], ra['Buts_encais_pm'],
                        f"{int(ra['Diff_buts']):+d}", int(ra['Matchs']), int(ra['Victoires'])],
                team_b:[f"{rb['Win_rate']}%", rb['Buts_pm'], rb['Buts_encais_pm'],
                        f"{int(rb['Diff_buts']):+d}", int(rb['Matchs']), int(rb['Victoires'])],
            }
            comp_df = pd.DataFrame(comp)
            fig_comp = go.Figure(go.Table(
                header=dict(values=[f"<b>{c}</b>" for c in comp_df.columns],
                            fill_color="#1a1f2e", font=dict(color=GOLD,size=11),
                            align="center", line_color=BORDER, height=30),
                cells=dict(values=[comp_df[c].tolist() for c in comp_df.columns],
                           fill_color=CARD, font=dict(color=TEXT_H,size=11),
                           align=["left","center","center"],
                           line_color=BORDER, height=27)
            ))
            fig_comp.update_layout(**{**PLOTLY_BASE, "margin": dict(t=10,b=10,l=10,r=10)}, height=230)
            st.plotly_chart(fig_comp, use_container_width=True)

    # ── Quarts de finale interactifs ──────────────────────────
    section("Simulation — Quarts de finale","Configurez les 4 matchups et simulez en un clic")

    default_qf = [
        ("Brazil","Germany"),("Argentina","Italy"),
        ("France","Spain"),("Netherlands","England"),
    ]
    st.markdown("<br>", unsafe_allow_html=True)
    qf_pairs = []
    for i in range(4):
        col_qa, col_qvs, col_qb = st.columns([5,1,5])
        da, db = default_qf[i]
        idx_a = ALL_TEAMS.index(da) if da in ALL_TEAMS else 0
        idx_b_v = ALL_TEAMS.index(db) if db in ALL_TEAMS else 1
        with col_qa:
            qa = st.selectbox(f"Équipe A — Match {i+1}", ALL_TEAMS, index=idx_a, key=f"qa{i}")
        with col_qvs:
            st.markdown(f"""<div style="text-align:center;padding-top:28px;
                        font-size:1rem;font-weight:700;color:{BORDER};">vs</div>""",
                        unsafe_allow_html=True)
        with col_qb:
            qb = st.selectbox(f"Équipe B — Match {i+1}", ALL_TEAMS, index=idx_b_v, key=f"qb{i}")
        qf_pairs.append((qa, qb))

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚽ Simuler les quarts de finale", type="primary", use_container_width=True):
        st.session_state["qf_results"] = []
        winners_qf = []
        for qa, qb in qf_pairs:
            if qa == qb:
                st.warning(f"Match {qa} vs {qb} : deux équipes identiques.")
                continue
            r = predict_match(qa, qb, team_stats, clf, sc_m)
            if r:
                st.session_state["qf_results"].append((qa, qb, r))
                w_idx = [r["p_win_a"], r["p_draw"], r["p_win_b"]].index(
                         max(r["p_win_a"], r["p_draw"], r["p_win_b"]))
                winners_qf.append([qa,"(nul)",qb][w_idx])
        st.session_state["qf_winners"] = winners_qf

    if "qf_results" in st.session_state and st.session_state["qf_results"]:
        cols_qf = st.columns(len(st.session_state["qf_results"]))
        for col, (qa, qb, r) in zip(cols_qf, st.session_state["qf_results"]):
            with col:
                pa2, pd2, pb2 = r["p_win_a"], r["p_draw"], r["p_win_b"]
                probs2 = [pa2, pd2, pb2]
                wi = probs2.index(max(probs2))
                bar_c2 = [GREEN if j==wi else ("#6B7A9A" if j==1 else BLUE) for j in range(3)]
                fig_qf = go.Figure(go.Bar(
                    x=[qa,"Nul",qb], y=probs2,
                    marker=dict(color=bar_c2, line=dict(color=BG,width=0.8)),
                    text=[f"{v*100:.1f}%" for v in probs2],
                    textposition="outside", textfont=dict(size=10,color=TEXT_H)
                ))
                w_name = [qa,"Nul",qb][wi]
                fig_qf.update_layout(
                    **{**PLOTLY_BASE,
                       "yaxis": dict(tickformat=".0%", range=[0,max(probs2)*1.35], gridcolor=BORDER),
                       "margin": dict(t=45,b=30,l=15,r=15)},
                    title=dict(text=f"<b style='color:{GOLD}'>{qa}</b> vs <b style='color:{GOLD}'>{qb}</b>",
                               font=dict(size=11,color=TEXT_H), x=0.5),
                    height=290,
                    showlegend=False
                )
                st.plotly_chart(fig_qf, use_container_width=True)
                emoji = "🏆" if wi != 1 else "🤝"
                color_w = GREEN if wi != 1 else "#6B7A9A"
                st.markdown(f"""<div style="text-align:center;font-size:0.85rem;
                            font-weight:600;color:{color_w};margin-top:-8px;">
                            {emoji} {w_name}</div>""", unsafe_allow_html=True)

        # Demi-finales projetées
        winners = st.session_state.get("qf_winners", [])
        if len(winners) == 4:
            section("Demi-finales projetées","Basées sur les vainqueurs des quarts")
            sf_pairs = [(winners[0], winners[1]), (winners[2], winners[3])]
            sf_cols = st.columns(2)
            finalists = []
            for col, (sa, sb) in zip(sf_cols, sf_pairs):
                with col:
                    if sa == sb or sa == "(nul)" or sb == "(nul)":
                        st.info("Match nul au QF — demi-finale incalculable.")
                        continue
                    r_sf = predict_match(sa, sb, team_stats, clf, sc_m)
                    if r_sf:
                        ps = [r_sf["p_win_a"], r_sf["p_draw"], r_sf["p_win_b"]]
                        wi_sf = ps.index(max(ps))
                        w_sf = [sa,"(nul)",sb][wi_sf]
                        finalists.append(w_sf)
                        bar_sf = [GREEN if j==wi_sf else ("#6B7A9A" if j==1 else BLUE) for j in range(3)]
                        fig_sf = go.Figure(go.Bar(
                            x=[sa,"Nul",sb], y=ps,
                            marker=dict(color=bar_sf,line=dict(color=BG,width=0.8)),
                            text=[f"{v*100:.1f}%" for v in ps],
                            textposition="outside", textfont=dict(size=10,color=TEXT_H)
                        ))
                        fig_sf.update_layout(
                            **{**PLOTLY_BASE,
                               "yaxis": dict(tickformat=".0%",range=[0,max(ps)*1.35],gridcolor=BORDER),
                               "margin": dict(t=40,b=25,l=15,r=15)},
                            title=dict(text=f"<b>{sa}</b> vs <b>{sb}</b>",
                                       font=dict(size=11,color=TEXT_H),x=0.5),
                            height=270, showlegend=False
                        )
                        st.plotly_chart(fig_sf, use_container_width=True)
                        st.markdown(f"""<div style="text-align:center;font-size:0.85rem;
                                    font-weight:600;color:{GREEN};margin-top:-6px;">
                                    🏆 {w_sf}</div>""", unsafe_allow_html=True)

            # Finale
            if len(finalists) == 2:
                section("Finale simulée", f"{finalists[0]} 🆚 {finalists[1]}")
                r_fin = predict_match(finalists[0], finalists[1], team_stats, clf, sc_m)
                if r_fin:
                    pf = [r_fin["p_win_a"], r_fin["p_draw"], r_fin["p_win_b"]]
                    wi_f = pf.index(max(pf))
                    champion = [finalists[0],"Match nul",finalists[1]][wi_f]
                    bar_f = [GREEN if j==wi_f else ("#6B7A9A" if j==1 else BLUE) for j in range(3)]
                    col_fin, col_champ = st.columns([3,2])
                    with col_fin:
                        fig_fin = go.Figure(go.Bar(
                            x=[finalists[0],"Nul",finalists[1]], y=pf,
                            marker=dict(color=bar_f,line=dict(color=BG,width=0.8)),
                            text=[f"{v*100:.1f}%" for v in pf],
                            textposition="outside",textfont=dict(size=12,color=TEXT_H)
                        ))
                        fig_fin.update_layout(
                            **{**PLOTLY_BASE,
                               "yaxis": dict(tickformat=".0%",range=[0,max(pf)*1.35],gridcolor=BORDER),
                               "margin": dict(t=20,b=25,l=20,r=20)},
                            height=300, showlegend=False
                        )
                        st.plotly_chart(fig_fin, use_container_width=True)
                    with col_champ:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg,{CARD},#1a1f2e);
                                    border:2px solid {GOLD};border-radius:14px;
                                    padding:30px;text-align:center;margin-top:10px;">
                          <div style="font-size:2.5rem;margin-bottom:8px;">🏆</div>
                          <div style="font-size:0.65rem;text-transform:uppercase;
                                      letter-spacing:0.15em;color:#4A5A7A;margin-bottom:8px;">
                            Champion simulé
                          </div>
                          <div style="font-size:1.6rem;font-weight:800;color:{GOLD};">
                            {champion}
                          </div>
                          <div style="font-size:0.75rem;color:#5A6A8A;margin-top:8px;">
                            {pf[wi_f]*100:.1f}% de probabilité
                          </div>
                        </div>""", unsafe_allow_html=True)
