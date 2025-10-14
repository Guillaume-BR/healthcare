# Healthcare Analysis

Étude d'un dataset consacré aux données médicales de certains patients hospitalisés, avec analyse descriptive et outil de prédiction interactif.

## 🌐 Site Web

Visitez notre site GitHub Pages pour:
- 📊 **Statistiques Descriptives**: Analyse complète des données médicales avec visualisations
- 🔮 **Prédiction Interactive**: Outil en ligne pour estimer la durée de séjour à l'hôpital

**Lien du site**: [https://guillaume-br.github.io/healthcare/](https://guillaume-br.github.io/healthcare/)

## 📁 Structure du Projet

- `index.html` - Page principale avec statistiques descriptives
- `prediction.html` - Page de prédiction interactive
- `style.css` - Styles CSS pour l'interface
- `charts.js` - Visualisations des données
- `prediction.js` - Logique de prédiction
- `data.csv` - Données exemple

## 🎯 Fonctionnalités

### Page Statistiques
- Statistiques descriptives (âge moyen, durée de séjour, etc.)
- Graphiques interactifs
- Analyse par sévérité et type d'admission
- Principales observations

### Page Prédiction
- Formulaire interactif avec plusieurs paramètres:
  - Âge du patient
  - Sexe
  - Type d'admission (urgence/programmée)
  - Sévérité du cas
- Calcul en temps réel de la durée estimée
- Intervalle de confiance
- Résultats visuels et détaillés

## 🚀 Utilisation Locale

1. Cloner le repository:
```bash
git clone https://github.com/Guillaume-BR/healthcare.git
cd healthcare
```

2. Ouvrir `index.html` dans votre navigateur

## 📊 Données

Le projet utilise des données anonymisées de patients avec les variables suivantes:
- **age**: Âge du patient
- **sexe**: M (Masculin) ou F (Féminin)
- **type_admission**: urgence ou programme
- **severite**: faible, moyen, eleve
- **duree_sejour**: Durée en jours

## 📝 License

MIT License - voir [LICENSE](LICENSE)
