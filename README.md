# ⚽ World Cup Intelligence Platform

Application Streamlit d'analyse ML sur les données FIFA Coupe du Monde 1930–2014.

## Lancement rapide

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages

| Page | Contenu |
|------|---------|
| 🏠 Vue Générale | KPIs, évolution buts, affluence, distribution scores |
| 🌍 Analyse par Équipe | Top 15, scatter, radar comparatif, fiche nation |
| 🤖 Clustering ML | K-Means k=4, méthode du coude, profils de jeu |
| 🎯 Prédiction ML | LogReg + HGB, matrices de confusion, importance features |
| ⚡ Simulateur | Pronostic match à la demande + quarts de finale scénarisés |

## Stack technique
- **ML** : scikit-learn (KMeans, LogisticRegression, HistGradientBoosting)
- **Viz** : Plotly (interactif, dark theme)
- **UI** : Streamlit + CSS custom
- **Data** : 354 matchs · 64 nations · 20 éditions (1930–2014)
