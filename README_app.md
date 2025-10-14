# 📈 Portfolio Analysis & Rebalancing Tool - Application Interactive

## 🎯 Description

Cette application Streamlit interactive permet d'analyser et de comparer différentes stratégies de portefeuille avec des curseurs pour modifier les paramètres en temps réel.

## ✨ Fonctionnalités

### 🎛️ Contrôles Interactifs
- **Sélection d'actions** : Choisissez parmi AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, NFLX
- **Période d'analyse** : Définissez la date de début et de fin
- **Fenêtre de lookback** : Ajustez la période d'estimation (60-500 jours)
- **Coûts de transaction** : Modifiez les coûts de transaction (0-50 bps)
- **Poids maximum** : Limitez le poids maximum par action (10-100%)

### 📊 Stratégies Disponibles
- **Equal Weight** : Répartition égale entre toutes les actions
- **Minimum Variance** : Optimisation pour minimiser la variance
- **Benchmark SPY** : Comparaison avec l'ETF SPY

### 📈 Visualisations
- **Courbes d'équité** : Évolution de la valeur du portefeuille
- **Métriques de performance** : Rendement, volatilité, Sharpe, drawdown
- **Évolution des poids** : Répartition du portefeuille dans le temps
- **Matrice de corrélation** : Corrélations entre les actions

## 🚀 Comment lancer l'application

### Méthode 1 : Script de lancement
```bash
cd /Users/nathanheimendinger/Downloads
python3 run_app.py
```

### Méthode 2 : Commande directe
```bash
cd /Users/nathanheimendinger/Downloads
streamlit run portfolio_app.py
```

### Méthode 3 : Avec paramètres personnalisés
```bash
cd /Users/nathanheimendinger/Downloads
streamlit run portfolio_app.py --server.port 8501 --server.address localhost
```

## 🌐 Accès à l'application

Une fois lancée, l'application sera accessible à l'adresse :
- **URL locale** : http://localhost:8501
- **URL réseau** : http://[votre-ip]:8501

## 🎮 Guide d'utilisation

### 1. Configuration initiale
- Sélectionnez les actions dans la barre latérale
- Choisissez la période d'analyse
- Ajustez les paramètres de backtest

### 2. Exploration interactive
- Modifiez les curseurs pour voir l'impact en temps réel
- Comparez différentes stratégies
- Analysez les métriques de performance

### 3. Analyse des résultats
- Consultez les métriques de performance
- Examinez les courbes d'équité
- Analysez l'évolution des poids du portefeuille

## 📋 Prérequis

- Python 3.9+
- Packages installés :
  - streamlit
  - plotly
  - pandas
  - numpy
  - yfinance
  - cvxpy
  - pyarrow

## 🔧 Dépannage

### Problème de port
Si le port 8501 est occupé, utilisez :
```bash
streamlit run portfolio_app.py --server.port 8502
```

### Problème de données
- Vérifiez votre connexion internet
- Assurez-vous que les tickers sont valides
- Vérifiez que la période sélectionnée contient des données

### Problème de performance
- Réduisez la période d'analyse
- Diminuez le nombre d'actions
- Augmentez la fenêtre de lookback

## 💡 Conseils d'utilisation

1. **Commencez simple** : Utilisez 2-3 actions pour comprendre l'interface
2. **Explorez les paramètres** : Modifiez les curseurs pour voir l'impact
3. **Comparez les stratégies** : Activez plusieurs stratégies simultanément
4. **Analysez les corrélations** : Regardez la matrice de corrélation
5. **Testez différents scénarios** : Changez les périodes et paramètres

## 🎨 Personnalisation

L'application peut être personnalisée en modifiant le fichier `portfolio_app.py` :
- Ajouter de nouvelles actions
- Modifier les couleurs des graphiques
- Ajouter de nouvelles métriques
- Créer de nouvelles stratégies

## 📞 Support

En cas de problème :
1. Vérifiez que tous les packages sont installés
2. Redémarrez l'application
3. Vérifiez les logs dans le terminal
4. Assurez-vous que les données sont disponibles

---

**🎉 Amusez-vous bien avec l'analyse de portefeuille !**
