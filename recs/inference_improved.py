"""
Improved inference: predict audio features from metadata using enhanced model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

try:
    import joblib
except ImportError:
    joblib = None

try:
    import pandas as pd
except ImportError:
    pd = None

MODEL_DIR = Path(__file__).resolve().parent / "model"
FEATURE_COLS = ["tempo", "energy", "danceability", "valence", "loudness"]


def predict_audio_features_from_metadata_improved(
    metadata: dict,
    model_dir: Optional[Path | str] = None,
    training_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    """
    Predict audio features from Spotify metadata using improved model.
    
    Uses the improved feature extraction from train_metadata_model_improved.py
    """
    model_dir = Path(model_dir) if model_dir else MODEL_DIR
    
    # Import the improved feature extraction
    try:
        from recs.train_metadata_model_improved import extract_metadata_features_improved
    except ImportError:
        # Fallback to old method
        from recs.inference import predict_audio_features_from_metadata
        return predict_audio_features_from_metadata(metadata, model_dir, training_df)
    
    # Load feature names
    feature_names_path = model_dir / "metadata_feature_names.json"
    if not feature_names_path.exists():
        return None
    
    try:
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
    except Exception:
        return None
    
    # Build metadata DataFrame
    metadata_df = pd.DataFrame([metadata])
    
    # Extract features using improved method
    try:
        X_meta = extract_metadata_features_improved(
            metadata_df,
            audio_features_df=training_df,
            feature_names=feature_names,  # Pass feature names for inference mode
            top_n_genres=50  # Match training
        )
    except Exception as e:
        # Fallback to old method
        from recs.inference import predict_audio_features_from_metadata
        return predict_audio_features_from_metadata(metadata, model_dir, training_df)
    
    # Load models and predict
    predictions = {}
    
    for feature_name in FEATURE_COLS:
        model_path = model_dir / f"metadata_model_{feature_name}.joblib"
        if not model_path.exists():
            return None
        
        try:
            if joblib is not None:
                model = joblib.load(model_path)
            else:
                import pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            
            # Ensure feature vector matches (reorder columns to match feature_names)
            X_pred = X_meta[feature_names].values if len(X_meta) > 0 else X_meta.values
            
            # Convert to DataFrame with proper column names to avoid warnings
            import pandas as pd
            X_pred_df = pd.DataFrame(X_pred, columns=feature_names)
            
            pred_value = model.predict(X_pred_df)[0]
            predictions[feature_name] = float(pred_value)
        except Exception as e:
            # If prediction fails, return None
            return None
    
    return predictions
