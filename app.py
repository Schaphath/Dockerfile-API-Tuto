
# Librairies
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import pickle
from pathlib import Path
from typing import Literal
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration des chemins
MODEL_PATH = Path("models/logistic_all.pkl")
SCALER_PATH = Path("models/MinMax_scaler.pkl")
MODEL_VERSION = "v1"

# Ordre des features (constante)
FEATURE_ORDER = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "concave_points_mean"
]

# Stockage global des modèles
class ModelContainer:
    model = None
    scaler = None

models = ModelContainer()

# Gestion du cycle de vie de l'application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le chargement et la libération des ressources"""
    # Startup
    logger.info("Chargement des modèles...")
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler introuvable : {SCALER_PATH}")
        
        with open(MODEL_PATH, "rb") as f:
            models.model = pickle.load(f)
        
        with open(SCALER_PATH, "rb") as f:
            models.scaler = pickle.load(f)
        
        logger.info("Modèles chargés avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du chargement : {e}")
        raise RuntimeError(f"Impossible de charger les modèles : {e}")
    
    yield
    
    # Shutdown
    logger.info("Arrêt de l'application")
    models.model = None
    models.scaler = None

# Configuration de l'application
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API de prédiction du cancer du sein utilisant Gradient Boosting",
    version="1.0.0",
    lifespan=lifespan
)

# Ajout du middleware CORS 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Schémas Pydantic
class InputVars(BaseModel):
    """Schéma de validation des données d'entrée"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "radius_mean": 0.99,
            "texture_mean": 10.38,
            "perimeter_mean": 122.8,
            "area_mean": 45.0,
            "smoothness_mean": 0.1184,
            "compactness_mean": 0.2776,
            "concavity_mean": 0.3001,
            "symmetry_mean": 0.2419,
            "fractal_dimension_mean": 0.07871,
            "concave_points_mean": 0.1471
        }
    })
    
    radius_mean: float = Field(..., gt=0, description="Rayon moyen de la tumeur")
    texture_mean: float = Field(..., gt=0, description="Texture moyenne")
    perimeter_mean: float = Field(..., gt=0, description="Périmètre moyen")
    area_mean: float = Field(..., gt=0, description="Aire moyenne")
    smoothness_mean: float = Field(..., gt=0, le=1, description="Lissage moyen")
    compactness_mean: float = Field(..., ge=0, le=1, description="Compacité moyenne")
    concavity_mean: float = Field(..., ge=0, le=1, description="Concavité moyenne")
    symmetry_mean: float = Field(..., gt=0, le=1, description="Symétrie moyenne")
    fractal_dimension_mean: float = Field(..., gt=0, le=1, description="Dimension fractale moyenne")
    concave_points_mean: float = Field(..., ge=0, le=1, description="Points concaves moyens")

class PredictionResponse(BaseModel):
    """Schéma de la réponse de prédiction"""
    prediction: Literal["M", "B"]
    label: str
    probability: float = Field(..., ge=0, le=1)
    model_version: str
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prediction": "M",
            "label": "Présence de cancer (Malin)",
            "probability": 0.8709,
            "model_version": "v1"
        }
    })

class HealthResponse(BaseModel):
    """Schéma de la réponse de santé"""
    status: str
    model_loaded: bool
    scaler_loaded: bool

# Endpoints de santé
@app.get("/", tags=["Info"])
def root():
    """Point d'entrée de l'API"""
    return {
        "message": "Breast Cancer Prediction API",
        "version": MODEL_VERSION,
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Vérifie l'état de santé de l'API"""
    return HealthResponse(
        status="ok",
        model_loaded=models.model is not None,
        scaler_loaded=models.scaler is not None
    )

@app.get("/ready", tags=["Health"])
def ready():
    """Vérifie si l'API est prête à recevoir des requêtes"""
    if models.model is None or models.scaler is None:
        logger.warning("Tentative d'accès avec modèles non chargés")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèles non chargés. Veuillez réessayer."
        )
    return {"status": "ready", "model_version": MODEL_VERSION}

# Endpoint de prédiction
@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)

def predict(data: InputVars):
    """
    Effectue une prédiction de cancer du sein.
    
    - **M** : Malin (présence de cancer)
    - **B** : Bénin (absence de cancer)
    """
    # Vérification des modèles
    if models.model is None or models.scaler is None:
        logger.error("Tentative de prédiction avec modèles non chargés")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèles non disponibles"
        )
    
    try:
        # Construction du vecteur de features dans le bon ordre
        features = np.array(
            [[getattr(data, feature) for feature in FEATURE_ORDER]]
        )
        
        logger.info(f"Prédiction demandée - Features: {features.flatten()}")
        
        # Normalisation
        features_scaled = models.scaler.transform(features)
        
        # Prédiction
        pred = models.model.predict(features_scaled)[0]
        proba = models.model.predict_proba(features_scaled)[0]
        
        # Interprétation des résultats
        if pred == "M":
            label = "Présence de cancer (Malin)"
            probability = float(proba[1])
        else:
            label = "Absence de cancer (Bénin)"
            probability = float(proba[0])
        
        logger.info(f"Prédiction: {pred} - Probabilité: {probability:.4f}")
        
        return PredictionResponse(
            prediction=pred,
            label=label,
            probability=round(probability, 4),
            model_version=MODEL_VERSION
        )
    
    except ValueError as ve:
        logger.error(f"Erreur de validation: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Données invalides : {str(ve)}"
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne lors de la prédiction"
        )