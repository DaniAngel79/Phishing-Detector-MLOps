import joblib
import pickle
import os
import re

# --- Rutas de Archivos y Configuración de Seguridad ---
ARTIFACTS_DIR = 'artifacts'
SECURITY_THRESHOLD = 0.40 # Usar el umbral optimizado
# --- FUNCIÓN DE INYECCIÓN CRÍTICA ---
def preprocess_text_for_train(text):
    """Función requerida por joblib para deserializar correctamente el TfidfVectorizer."""
    return str(text).lower().strip() 
# -----------------------------------
def preprocess_text(text):
    """Función de preprocesamiento, debe ser idéntica a la usada en el entrenamiento."""
    text = str(text).lower().strip()
    return text

def load_model_and_vectorizer():
    """Carga los artefactos persistidos."""
    try:
        vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, 'tfidf_vectorizer.pkl'))
        model = joblib.load(os.path.join(ARTIFACTS_DIR, 'logistic_regression_smote_model.pkl'))
        with open(os.path.join(ARTIFACTS_DIR, 'label_map.pkl'), 'rb') as f:
            label_map = pickle.load(f)
        return vectorizer, model, label_map
    except Exception as e:
        print(f"ERROR: No se pudieron cargar artefactos: {e}")
        return None, None, None

def run_prediction(text_list):
    """Ejecuta la inferencia en una lista de textos."""
    vectorizer, model, label_map = load_model_and_vectorizer()
    if not model:
        return

    # Preprocesar y Vectorizar
    processed_texts = [preprocess_text(text) for text in text_list]
    X_new = vectorizer.transform(processed_texts)

    # Predecir Probabilidades
    probabilities = model.predict_proba(X_new)[:, 1]
    
    results = []
    for text, proba in zip(text_list, probabilities):
        # Aplicar el umbral de seguridad optimizado (0.40)
        prediction_num = 1 if proba >= SECURITY_THRESHOLD else 0
        prediction_label = label_map.get(prediction_num, 'ERROR')
        
        results.append({
            'Texto': text[:80] + '...',
            'Clasificación': prediction_label,
            'Probabilidad_Spam': f"{proba:.4f}"
        })
    return results

if __name__ == "__main__":
    # CASOS DE PRUEBA DE INTEGRIDAD (Sanity Check)
    test_samples = [
        "URGENTE: Su cuenta bancaria ha sido bloqueada. Haga clic aquí para verificar su identidad: http://phish.ly/login", # Phishing obvio
        "Hey, ¿quieres ir a almorzar mañana? Tengo una reunión a las 2.", # Ham (Seguro)
        "Has ganado un premio de 1000 euros. Responde 'GANADOR' para reclamar.", # Spam/Scam
    ]
    
    predictions = run_prediction(test_samples)
    if predictions:
        print("\n--- RESULTADOS DE INFERENCIA DE INTEGRIDAD ---")
        for res in predictions:
            print(f"\nTexto: {res['Texto']}")
            print(f"-> CLASIFICACIÓN ({res['Probabilidad_Spam']}): {res['Clasificación']}")

