"""
data_engine.py — Chargement des vraies données + pipeline ML
Source : world_cup_results.xlsx (3 feuilles)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_PATH = "world_cup_results.xlsx"

TEAM_UNIFICATION = {
    "Germany FR":        "Germany",
    "Czechoslovakia":    "Czech Republic",
    "Soviet Union":      "Russia",
    "Yugoslavia":        "Serbia",
    "Dutch East Indies": "Indonesia",
    "East Germany":      "Germany",
    "Zaire":             "DR Congo",
    "Bohemia":           "Czech Republic",
}

# ─────────────────────────────────────────────────────────────
def _clean_attendance(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(".", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan

def _phase(round_str):
    r = str(round_str).lower()
    group_kw = ["group", "first round", "preliminary", "round 1"]
    return "Groupes" if any(k in r for k in group_kw) else "Élimination"


# ─────────────────────────────────────────────────────────────
def build_dataframes():
    xl = pd.read_excel(DATA_PATH, sheet_name=None)

    # ── WorldCupMatches ──────────────────────────────────────
    matches = xl["WorldCupMatches"].copy()
    matches["HomeTeam"] = matches["HomeTeam"].str.strip().replace(TEAM_UNIFICATION)
    matches["AwayTeam"] = matches["AwayTeam"].str.strip().replace(TEAM_UNIFICATION)
    matches = matches.dropna(subset=["HomeGoals", "AwayGoals"]).copy()
    matches["HomeGoals"] = matches["HomeGoals"].astype(int)
    matches["AwayGoals"] = matches["AwayGoals"].astype(int)
    matches["TotalGoals"] = matches["HomeGoals"] + matches["AwayGoals"]
    matches["Margin"]     = (matches["HomeGoals"] - matches["AwayGoals"]).abs()
    matches["Result"]     = matches.apply(
        lambda r: "HomeWin" if r["HomeGoals"] > r["AwayGoals"]
        else ("AwayWin" if r["HomeGoals"] < r["AwayGoals"] else "Draw"), axis=1)
    matches["Phase"] = matches["Round"].apply(_phase)

    # ── Tableau format ───────────────────────────────────────
    tableau = xl["World Cup - Tableau format"].copy()
    tableau["Team"]     = tableau["Team"].str.strip().replace(TEAM_UNIFICATION)
    tableau["Opponent"] = tableau["Opponent"].str.strip().replace(TEAM_UNIFICATION)
    tableau = tableau.dropna(subset=["Team G", "Opponent G"]).copy()
    tableau["Team G"]     = tableau["Team G"].astype(int)
    tableau["Opponent G"] = tableau["Opponent G"].astype(int)
    tableau["Result"] = tableau.apply(
        lambda r: "Win"  if r["Team G"] > r["Opponent G"]
        else ("Loss" if r["Team G"] < r["Opponent G"] else "Draw"), axis=1)
    tableau["Phase"] = tableau["Round"].apply(_phase)

    # ── WorldCups ────────────────────────────────────────────
    worldcups = xl["WorldCups"].copy()
    worldcups["Winner"] = worldcups["Winner"].str.strip().replace(TEAM_UNIFICATION)
    worldcups["Attendance"] = worldcups["Attendance"].apply(_clean_attendance)

    return matches, tableau, worldcups


# ─────────────────────────────────────────────────────────────
def build_team_stats(tableau):
    ts = tableau.groupby("Team").agg(
        Matchs       = ("Result", "count"),
        Victoires    = ("Result", lambda x: (x == "Win").sum()),
        Nuls         = ("Result", lambda x: (x == "Draw").sum()),
        Defaites     = ("Result", lambda x: (x == "Loss").sum()),
        Buts_marques = ("Team G", "sum"),
        Buts_encais  = ("Opponent G", "sum"),
        Editions     = ("Year", "nunique"),
    ).reset_index()
    ts["Win_rate"]       = (ts["Victoires"] / ts["Matchs"] * 100).round(1)
    ts["Buts_pm"]        = (ts["Buts_marques"] / ts["Matchs"]).round(2)
    ts["Buts_encais_pm"] = (ts["Buts_encais"]  / ts["Matchs"]).round(2)
    ts["Diff_buts"]      = ts["Buts_marques"] - ts["Buts_encais"]
    return ts


# ─────────────────────────────────────────────────────────────
def build_clusters(team_stats):
    features = ["Win_rate", "Buts_pm", "Buts_encais_pm", "Diff_buts", "Matchs"]
    X = team_stats[features].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    inertias = []
    k_range  = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xs)
        inertias.append(km.inertia_)

    K = 4
    km4 = KMeans(n_clusters=K, random_state=42, n_init=10)
    ts  = team_stats.copy()
    ts["Cluster"] = km4.fit_predict(Xs)

    means = ts.groupby("Cluster")[features].mean()
    names = {}
    for c in range(K):
        wr = means.loc[c, "Win_rate"]
        if wr >= 55:   names[c] = "🏆 Dominants"
        elif wr >= 42: names[c] = "💪 Solides"
        elif wr >= 28: names[c] = "⚡ Challengers"
        else:          names[c] = "🌱 Émergents"

    ts["Profil"] = ts["Cluster"].map(names)
    return ts, list(k_range), inertias


# ─────────────────────────────────────────────────────────────
def build_ml_model(team_stats, tableau):
    stat_cols = ["Team", "Win_rate", "Buts_pm", "Buts_encais_pm", "Diff_buts"]

    df = tableau.merge(team_stats[stat_cols], on="Team", how="left")
    df = df.merge(
        team_stats[stat_cols].rename(columns={
            "Team": "Opponent",
            "Win_rate":       "Opp_wr",
            "Buts_pm":        "Opp_bpm",
            "Buts_encais_pm": "Opp_bencais",
            "Diff_buts":      "Opp_diff",
        }),
        on="Opponent", how="left"
    )
    df["label"] = df["Result"].map({"Win": 2, "Draw": 1, "Loss": 0})

    feats = ["Win_rate", "Buts_pm", "Buts_encais_pm", "Diff_buts",
             "Opp_wr",   "Opp_bpm", "Opp_bencais",    "Opp_diff"]
    df = df.dropna(subset=feats + ["label"])
    X  = df[feats]
    y  = df["label"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)

    return clf, sc, X_te_s, y_te


# ─────────────────────────────────────────────────────────────
def predict_match(team_a, team_b, team_stats, clf, sc):
    teams = team_stats["Team"].values
    if team_a not in teams or team_b not in teams:
        return None
    ra = team_stats[team_stats["Team"] == team_a].iloc[0]
    rb = team_stats[team_stats["Team"] == team_b].iloc[0]
    vec = [[
        ra["Win_rate"], ra["Buts_pm"], ra["Buts_encais_pm"], ra["Diff_buts"],
        rb["Win_rate"], rb["Buts_pm"], rb["Buts_encais_pm"], rb["Diff_buts"],
    ]]
    vec_s = sc.transform(pd.DataFrame(vec, columns=[
        "Win_rate","Buts_pm","Buts_encais_pm","Diff_buts",
        "Opp_wr","Opp_bpm","Opp_bencais","Opp_diff"
    ]))
    p = clf.predict_proba(vec_s)[0]
    return {"team_a": team_a, "team_b": team_b,
            "p_win_a": p[2], "p_draw": p[1], "p_win_b": p[0]}
