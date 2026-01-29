# API FastAPI - Prédiction du Cancer du Sein

API REST pour la prédiction du cancer du sein basée sur l'apprentissage automatique.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## Description

API REST permettant de prédire si une tumeur mammaire est **bénigne (B)** ou **maligne (M)** à partir de 10 caractéristiques cellulaires. Utilise un modèle **HistGradientBoostingClassifier** avec normalisation MinMaxScaler.

**Performances** : Recall ~98% | Precision ~97 | Accuracy ~97%

---

## Fonctionnalités

- **Health checks** : Surveillance de l'état de l'API
- **ML Pipeline** : Modèle + Scaler chargés au démarrage
- **Documentation** : Swagger UI intégrée
- **Docker** : Conteneurisation

---

## Installation

### Prérequis

- Python 3.12+ (pour installation locale)
- Docker Desktop
- FastAPI

### Option 1 : Installation locale

```bash
# Installer les dépendances
pip install -r requirements-prod.txt

# Lancer l'API
uvicorn app:app --reload
```

L'API sera accessible sur [http://localhost:8000](http://localhost:8000)


### Option 2 : Avec Docker

#### Build de l'image

```bash
docker build -t my_app:v1.0.0 .
```

#### Lancer le conteneur

```bash
# Lancement simple
docker run -d -p 8000:8000 --name api_cancer my_app:v1.0.0
```

#### Gestion du conteneur

```bash
# Voir les logs
docker logs api_cancer

# Arrêter le conteneur
docker stop api_cancer

# Redémarrer
docker start api-container

# Ou en une fois
docker rm -f api_cancer
```

---


**Réponse de l'API** :

```json
{
  "prediction": "M",
  "label": "Présence de cancer (Malin)",
  "probability": 0.9845
}
```

**Codes de statut** :
- `200` : Prédiction réussie
- `422` : Données invalides
- `500` : Erreur interne
- `503` : Modèles non chargés

---

## Modèle ML

### Pipeline

```
Input (10 features) 
    ↓
MinMaxScaler (normalisation)
    ↓
HistGradientBoostingClassifier
    ↓
Output (B ou M + probabilité)
```

---

## Avertissement

**Cette API est à but éducatif uniquement.** Elle ne doit pas être utilisée pour prendre des décisions médicales réelles. 

---

## Auteur

**Madiba**

---

**API développée avec FastAPI**
