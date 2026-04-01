# 🏥 Prédiction de la Durée d'Hospitalisation

Application web interactive développée avec Streamlit permettant d'estimer la durée d'hospitalisation d'un patient à partir de données démographiques, comportementales et médicales.

---

## 🎯 Objectif

Ce projet vise à construire un outil d’aide à la décision basé sur le machine learning pour :

* Estimer la durée d'hospitalisation
* Fournir des indicateurs de santé en temps réel
* Aider à l’anticipation des besoins hospitaliers

---

## 🚀 Fonctionnalités

### 🔍 Prédiction

* Interface utilisateur intuitive
* Saisie des données patient :

  * Données démographiques (âge, sexe)
  * Habitudes de vie (tabac, alcool, activité physique)
  * Indicateurs médicaux (IMC, glucose, HbA1c)
  * Antécédents médicaux
* Prédiction instantanée de la durée d'hospitalisation

---

### 📊 Indicateurs en temps réel

* Calcul automatique de l’IMC
* Interprétation du niveau de glucose
* Feedback visuel dynamique

---

### 🤖 Modèle de Machine Learning

* Modèle entraîné sur des données réelles
* Pipeline incluant :

  * Préprocessing (`preprocessor.pkl`)
  * Modèle final (`best_model.pkl`)
* Validation croisée effectuée

---

## 🧠 Technologies utilisées

* Python
* Streamlit
* NumPy / Pandas
* Scikit-learn
* Joblib

---

## 📁 Structure du projet

```
.
├── app.py
├── model/
│   ├── best_model.pkl
│   └── preprocessor.pkl
├── data/
├── src/
└── README.md
```

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <repo_url>
cd <repo_name>
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## ▶️ Lancer l'application

```bash
streamlit run app.py
```

---

## 📈 Exemple d'utilisation

1. Renseigner les informations du patient
2. Vérifier les indicateurs calculés (IMC, glucose)
3. Cliquer sur **"Prédire la durée d'hospitalisation"**
4. Obtenir une estimation en jours

---

## ⚠️ Limites et avertissements

* Ce modèle est basé sur des données statistiques et n'obtient qu'un R2 de 56%.
* Il ne remplace pas un avis médical.
* Les prédictions peuvent comporter des erreurs.
* Utilisation à des fins éducatives et exploratoires uniquement.

---

## 🔒 Confidentialité

* Aucune donnée utilisateur n’est stockée
* Les calculs sont réalisés en temps réel

---

## 💡 Améliorations possibles

* Ajout de nouvelles variables médicales
* Amélioration des performances du modèle
* Déploiement cloud (AWS / GCP)
* Monitoring des prédictions
* Ajout d'explicabilité (SHAP)

---

## 👨‍💻 Auteur

Projet réalisé dans le cadre d’un projet personnel en data science.

---

## 📬 Contact

N'hésitez pas à me contacter pour toute question ou collaboration.

---
