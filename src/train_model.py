import pandas as pd
import joblib
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# --- Rutas de Archivos ---
ARTIFACTS_DIR = 'artifacts'
DATA_DIR = 'data'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def preprocess_text_for_train(text):
    """Función de preprocesamiento simplificada para aplicar en la vectorización."""
    return str(text).lower().strip() 

def run_training():
    print("--- 1. Carga de Splits de Datos ---")
    try:
        # CORRECCIÓN DE CARGA: Usar .iloc[:, 0] para tomar la primera columna (Texto o Label)
        X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv')).iloc[:, 0]
        y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).iloc[:, 0]
    except Exception as e:
        print(f"ERROR: No se pudieron cargar los archivos de split. {e}")
        return

    # 2. Vectorización TF-IDF
    print("--- 2. Vectorización y Transformación ---")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, 
                                       preprocessor=preprocess_text_for_train)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train.astype(str))

    # 3. Balanceo con SMOTE
    print("--- 3. Balanceo de Clases (SMOTE) ---")
    sm = SMOTE(random_state=42)
    X_train_smote, y_train_smote = sm.fit_resample(X_train_vec, y_train)

    # 4. Entrenamiento del Modelo Final
    print("--- 4. Entrenamiento de Regresión Logística (Final) ---")
    model_res = LogisticRegression(solver='liblinear', random_state=42)
    model_res.fit(X_train_smote, y_train_smote)
    
    # 5. Serialización (Persistencia)
    print("--- 5. Serialización de Artefactos ---")
    joblib.dump(model_res, os.path.join(ARTIFACTS_DIR, 'logistic_regression_smote_model.pkl'))
    joblib.dump(tfidf_vectorizer, os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl'))
    
    label_map = {1: 'spam', 0: 'ham'} 
    with open(os.path.join(ARTIFACTS_DIR, 'label_map.pkl'), 'wb') as f:
        pickle.dump(label_map, f)
        
    print("✅ Entrenamiento completado y artefactos guardados.")


if __name__ == "__main__":
    run_training()
