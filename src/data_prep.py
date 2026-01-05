import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Rutas de Archivos (Relativas a la ejecución) ---
DATA_PATH = 'data/emails_auditing.csv'

# Directorios de salida para los splits
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

def run_data_prep():
    """Carga, limpia y divide los datos, guardando los splits como CSV."""
    
    print("--- 1. Carga y Limpieza de Datos ---")
    try:
        # 🛠️ SOLUCIÓN para emails_auditing.csv (Delimitador = TABULADOR)
        emails_df = pd.read_csv(DATA_PATH, sep='\t', encoding='latin1', header=None, names=['label', 'text'])
        
        # Mapeo y Limpieza
        emails_df['label_num'] = emails_df['label'].map({'spam': 1, 'ham': 0})
        emails_df.dropna(subset=['text', 'label_num'], inplace=True)
        
    except Exception as e:
        print(f"ERROR: Falló la carga o limpieza de {DATA_PATH}. {e}")
        return

    print("--- 2. División del Dataset ---")
    X = emails_df['text']
    y = emails_df['label_num']
    
    # Stratify asegura que la proporción de spam/ham sea similar en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Guardar los splits para el Pipeline
    X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), header=True, index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), header=True, index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), header=True, index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), header=True, index=False)
    
    print(f"✅ Splits de datos guardados en el directorio '{DATA_DIR}/'.")


if __name__ == "__main__":
    run_data_prep()
