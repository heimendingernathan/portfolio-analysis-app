#!/usr/bin/env python3
"""
Script pour déployer l'application sur Streamlit Cloud
Crée les fichiers nécessaires pour le déploiement
"""

import os
import shutil
import subprocess
import sys

def create_requirements_file():
    """Crée le fichier requirements.txt pour Streamlit Cloud"""
    requirements = [
        "streamlit>=1.28.0",
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.0",
        "cvxpy>=1.3.0",
        "matplotlib>=3.6.0",
        "plotly>=5.15.0",
        "pyarrow>=10.0.0"
    ]
    
    with open('requirements.txt', 'w') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print("✅ Fichier requirements.txt créé")

def create_streamlit_config():
    """Crée le fichier de configuration Streamlit"""
    config_content = """[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
"""
    
    os.makedirs('.streamlit', exist_ok=True)
    with open('.streamlit/config.toml', 'w') as f:
        f.write(config_content)
    
    print("✅ Fichier de configuration Streamlit créé")

def create_readme():
    """Crée un README pour le déploiement"""
    readme_content = """# Portfolio Analysis App

Application d'analyse de portefeuille avec stratégies d'optimisation.

## Fonctionnalités

- **Equal Weight** : Répartition égale
- **Minimum Variance** : Minimisation du risque (mensuel/trimestriel)
- **Portefeuille Optimal** : Maximisation du rendement (mensuel/trimestriel)
- **Benchmark SPY** : Comparaison avec l'ETF

## Stratégies disponibles

1. **Equal Weight** : Diversification simple
2. **Minimum Variance (Mensuel)** : Réduction du risque avec rééquilibrage mensuel
3. **Minimum Variance (Trimestriel)** : Réduction du risque avec rééquilibrage trimestriel
4. **Portefeuille Optimal (Mensuel)** : Maximisation du rendement avec rééquilibrage mensuel
5. **Portefeuille Optimal (Trimestriel)** : Maximisation du rendement avec rééquilibrage trimestriel
6. **Benchmark SPY** : Indice de référence

## Paramètres configurables

- **Actions** : Sélection des tickers
- **Période** : Date de début et fin
- **Fenêtre de lookback** : Nombre de jours d'historique
- **Coûts de transaction** : Frais en points de base
- **Poids maximum** : Limite de concentration par action
- **Poids minimum** : Diversification forcée

## Utilisation

1. Sélectionnez vos actions
2. Ajustez les paramètres
3. Choisissez les stratégies à comparer
4. Analysez les résultats

## Déploiement

Cette application est déployée sur Streamlit Cloud.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ README.md créé")

def create_gitignore():
    """Crée un fichier .gitignore"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Streamlit
.streamlit/

# Data files
*.parquet
*.csv
*.xlsx

# Logs
*.log

# OS
.DS_Store
Thumbs.db
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore créé")

def create_deployment_guide():
    """Crée un guide de déploiement"""
    guide_content = """# Guide de Déploiement sur Streamlit Cloud

## Étapes pour déployer votre application

### 1. Préparer le repository GitHub

1. **Créer un repository GitHub** :
   - Allez sur https://github.com
   - Cliquez sur "New repository"
   - Nommez-le "portfolio-analysis-app"
   - Cochez "Public" (nécessaire pour Streamlit Cloud gratuit)

2. **Initialiser Git localement** :
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Portfolio Analysis App"
   ```

3. **Connecter au repository GitHub** :
   ```bash
   git remote add origin https://github.com/VOTRE_USERNAME/portfolio-analysis-app.git
   git branch -M main
   git push -u origin main
   ```

### 2. Déployer sur Streamlit Cloud

1. **Aller sur Streamlit Cloud** :
   - Visitez https://share.streamlit.io
   - Connectez-vous avec votre compte GitHub

2. **Créer une nouvelle app** :
   - Cliquez sur "New app"
   - Sélectionnez votre repository : "portfolio-analysis-app"
   - Nom du fichier principal : "portfolio_app.py"
   - Cliquez sur "Deploy"

3. **Attendre le déploiement** :
   - Streamlit Cloud va installer les dépendances
   - L'application sera disponible à l'URL fournie

### 3. Configuration avancée (optionnel)

Si vous voulez personnaliser le déploiement, créez un fichier `streamlit_app.py` :

```python
# streamlit_app.py
import streamlit as st

# Votre application existante
exec(open('portfolio_app.py').read())
```

### 4. URLs de déploiement

Une fois déployé, votre application sera accessible à :
- **URL publique** : `https://VOTRE_USERNAME-portfolio-analysis-app-main-XXXXXX.streamlit.app`
- **Partage** : Cette URL peut être partagée avec n'importe qui

### 5. Mise à jour de l'application

Pour mettre à jour l'application :
1. Modifiez les fichiers localement
2. Committez les changements :
   ```bash
   git add .
   git commit -m "Update app"
   git push
   ```
3. Streamlit Cloud redéploiera automatiquement

## Avantages de Streamlit Cloud

- ✅ **Gratuit** pour les repositories publics
- ✅ **URL publique** accessible partout
- ✅ **Déploiement automatique** à chaque push
- ✅ **HTTPS sécurisé** inclus
- ✅ **Pas de configuration serveur** nécessaire
- ✅ **Mise à jour automatique** des dépendances

## Support

- Documentation Streamlit Cloud : https://docs.streamlit.io/streamlit-community-cloud
- Support communautaire : https://discuss.streamlit.io
"""
    
    with open('DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide_content)
    
    print("✅ Guide de déploiement créé")

def main():
    """Fonction principale"""
    print("🚀 Préparation du déploiement sur Streamlit Cloud")
    print("=" * 60)
    
    # Créer les fichiers nécessaires
    create_requirements_file()
    create_streamlit_config()
    create_readme()
    create_gitignore()
    create_deployment_guide()
    
    print("\n" + "=" * 60)
    print("✅ FICHIERS DE DÉPLOIEMENT CRÉÉS!")
    print("=" * 60)
    print("📁 Fichiers créés :")
    print("   - requirements.txt")
    print("   - .streamlit/config.toml")
    print("   - README.md")
    print("   - .gitignore")
    print("   - DEPLOYMENT_GUIDE.md")
    
    print("\n🎯 PROCHAINES ÉTAPES :")
    print("1. Créer un repository GitHub public")
    print("2. Initialiser Git et pousser le code")
    print("3. Déployer sur https://share.streamlit.io")
    print("4. Partager l'URL publique générée")
    
    print("\n📖 Consultez DEPLOYMENT_GUIDE.md pour les instructions détaillées")

if __name__ == "__main__":
    main()
