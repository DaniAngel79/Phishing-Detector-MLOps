import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import json
# --- FUNCIÓN DE INYECCIÓN CRÍTICA ---
def preprocess_text_for_train(text):
    """Función requerida por joblib para deserializar correctamente el TfidfVectorizer."""
    return str(text).lower().strip() 
# -----------------------------------
# --- Rutas de Archivos y Configuración de Seguridad ---
ARTIFACTS_DIR = 'artifacts'
DATA_DIR = 'data'
METRICS_DIR = 'metrics'
os.makedirs(METRICS_DIR, exist_ok=True)

SECURITY_THRESHOLD = 0.40 # Umbral optimizado para reducir FN

def run_evaluation():
    print("--- 1. Carga de Artefactos y Datos de Prueba ---")
    try:
        # Cargar Modelo y Vectorizador
        model = joblib.load(os.path.join(ARTIFACTS_DIR, 'logistic_regression_smote_model.pkl'))
        vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl'))
        
        # Cargar Datos de Prueba
        X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))['text']
        y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv'))['label_num'].values
    except Exception as e:
        print(f"ERROR al cargar recursos para evaluación: {e}")
        return

    # 2. Vectorización del Test Set
    X_test_vec = vectorizer.transform(X_test)

    # 3. Predicción y Aplicación del Umbral de Seguridad
    print(f"--- 2. Aplicando Umbral de Seguridad ({SECURITY_THRESHOLD}) ---")
    y_pred_proba = model.predict_proba(X_test_vec)[:, 1]
    y_pred = (y_pred_proba >= SECURITY_THRESHOLD).astype(int)

    # 4. Cálculo de Métricas
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)
    
    # 5. Auditoría Crítica de Falsos Negativos (FN)
    # FN es la celda (1, 0) de la matriz de confusión (Real=1, Predicho=0)
    fn_count = conf_matrix[1, 0] 

    # 6. Persistencia de Métricas
    print("--- 3. Persistencia de Métricas ---")
    
    # Guardar Reporte Completo (JSON)
    with open(os.path.join(METRICS_DIR, 'classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    # Guardar Conteo de FN (Métrica de Seguridad CRÍTICA)
    with open(os.path.join(METRICS_DIR, 'fn_count.txt'), 'w') as f:
        f.write(f"FN_COUNT: {fn_count}\n")
        f.write(f"RECALL_SPAM: {report['spam']['recall']:.4f}\n")
        f.write(f"PRECISION_SPAM: {report['spam']['precision']:.4f}\n")
        f.write(f"THRESHOLD_USED: {SECURITY_THRESHOLD}\n")
        
    print(f"✅ Evaluación completada. FN: {fn_count}. Métricas guardadas en '{METRICS_DIR}/'.")


if __name__ == "__main__":
    run_evaluation()
