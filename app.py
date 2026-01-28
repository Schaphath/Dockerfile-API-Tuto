"""
API de prédiction du cancer du sein
Utilise un modèle de régression logistique avec MinMax scaling
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field, field_validator
import numpy as np
import pickle
from pathlib import Path
from typing import Literal
import logging

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/hist_gradient_boosting_best.pkl")
SCALER_PATH = Path("models/MinMax_scaler.pkl")

FEATURE_ORDER = [
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

# ═══════════════════════════════════════════════════════════════
# Stockage global des modèles
# ═══════════════════════════════════════════════════════════════

class ModelContainer:
    """Conteneur singleton pour les modèles ML"""
    model = None
    scaler = None

models = ModelContainer()

# ═══════════════════════════════════════════════════════════════
# Cycle de vie de l'application
# ═══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge les modèles au démarrage et les libère à l'arrêt"""
    logger.info("Démarrage de l'API")
    
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler introuvable: {SCALER_PATH}")
        
        with open(MODEL_PATH, "rb") as f:
            models.model = pickle.load(f)
        
        with open(SCALER_PATH, "rb") as f:
            models.scaler = pickle.load(f)
        
        logger.info("Modèles chargés avec succès")
        
    except Exception as e:
        logger.error(f"Erreur au chargement: {e}")
        raise RuntimeError(f"Impossible de charger les modèles: {e}")
    
    yield
    
    logger.info("Arrêt de l'API")
    models.model = None
    models.scaler = None

# ═══════════════════════════════════════════════════════════════
# Application FastAPI
# ═══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API de prédiction du cancer du sein",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ═══════════════════════════════════════════════════════════════
# Schémas Pydantic
# ═══════════════════════════════════════════════════════════════

class InputData(BaseModel):
    """Données d'entrée pour la prédiction (10 features 'worst')"""
    
    radius_worst: float = Field(..., gt=0)
    texture_worst: float = Field(..., gt=0)
    perimeter_worst: float = Field(..., gt=0)
    area_worst: float = Field(..., gt=0)
    smoothness_worst: float = Field(..., gt=0, le=1)
    compactness_worst: float = Field(..., ge=0, le=1)
    concavity_worst: float = Field(..., ge=0, le=1)
    concave_points_worst: float = Field(..., ge=0, le=1)
    symmetry_worst: float = Field(..., gt=0, le=1)
    fractal_dimension_worst: float = Field(..., gt=0, le=1)
    
    @field_validator('*')
    @classmethod
    def check_not_nan(cls, v):
        """Vérifie qu'aucune valeur n'est NaN"""
        if np.isnan(v):
            raise ValueError("Les valeurs NaN ne sont pas autorisées")
        return v


class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    
    prediction: Literal["M", "B"]
    label: str
    probability: float = Field(..., ge=0, le=1)


class HealthResponse(BaseModel):
    """Réponse du health check"""
    
    status: str
    model_loaded: bool
    scaler_loaded: bool

# ═══════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/", tags=["Info"])
def root():
    """Point d'entrée de l'API"""
    return {
        "name": "Breast Cancer Prediction API",
        "version": "1.0.0",
        "endpoints": ["/health", "/predict", "/docs"]
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Vérifie l'état de santé de l'API"""
    return HealthResponse(
        status="healthy" if (models.model and models.scaler) else "unhealthy",
        model_loaded=models.model is not None,
        scaler_loaded=models.scaler is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: InputData):
    """
    Prédit la présence de cancer du sein
    
    Returns:
        - M (Malin): Cancer détecté
        - B (Bénin): Pas de cancer détecté
    """
    
    # Vérification de la disponibilité des modèles
    if models.model is None or models.scaler is None:
        logger.error("Tentative de prédiction avec modèles non chargés")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèles non disponibles"
        )
    
    try:
        # Construction du vecteur de features dans l'ordre correct
        features = np.array([[
            getattr(data, feature) for feature in FEATURE_ORDER
        ]])
        
        logger.info(f"Prédiction - Features reçues: {features.shape}")
        
        # Normalisation
        features_scaled = models.scaler.transform(features)
        
        # Prédiction
        prediction_raw = models.model.predict(features_scaled)[0]
        probabilities = models.model.predict_proba(features_scaled)[0]
        
        # Convertir la prédiction (0/1) en label (B/M)
        # 0 = Bénin (B), 1 = Malin (M)
        prediction = "M" if prediction_raw == 1 else "B"
        
        # Formater la réponse
        if prediction == "M":
            label = "Présence de cancer (Malin)"
            probability = float(probabilities[1])
        else:
            label = "Absence de cancer (Bénin)"
            probability = float(probabilities[0])
        
        logger.info(f"Prédiction: {prediction} | Probabilité: {probability:.4f}")
        
        return PredictionResponse(
            prediction=prediction,
            label=label,
            probability=round(probability, 4)
        )
    
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Données invalides: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erreur interne du serveur"
        )