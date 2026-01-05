# 🛡️ Phishing Detector MLOps

Este repositorio contiene la arquitectura, el código y los pipelines (MLOps) para un sistema de detección de ataques de Phishing basado en Machine Learning (ML). El sistema se centra en el análisis de URLs y contenido de correos electrónicos.

## 🚀 Estructura del Proyecto

El proyecto sigue una estructura de repositorio estandarizada para proyectos de Data Science/ML:

| Directorio | Descripción |
| :--- | :--- |
| `src/` | Código fuente principal del modelo, entrenamiento y funciones de preprocesamiento. |
| `data/` | Datos brutos y preprocesados (solo metadatos o muestras pequeñas). |
| `metrics/` | Resultados de evaluación del modelo (F1-Score, Recall, Curvas ROC). |
| `artifacts/` | Modelos serializados (`.pkl`, `.h5`) y *checkpoints* de entrenamiento. |
| `requirements.txt` | Dependencias de Python necesarias para replicar el entorno (pandas, scikit-learn, etc.). |
| `EvaluationsModels_Practical.ipynb` | Notebook de Colab para la experimentación y evaluación de modelos. |

## ⚙️ Requisitos del Entorno

Para replicar el entorno de entrenamiento y validación:

1.  **Clonar el repositorio:**
    ```bash
    git clone git@github.com:DaniAngel79/Phishing-Detector-MLOps.git
    cd Phishing-Detector-MLOps
    ```

2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Modelos Clave

El proyecto se enfoca en la implementación y comparación de:

* **Clasificadores Tradicionales:** Logistic Regression, Random Forest.
* **Modelos de Aprendizaje Profundo (DL):** Redes Neuronales Recurrentes (RNN/LSTM) o Convencionales (CNN) para el análisis de texto y secuencias de URLs.

## 📝 Evaluación

Los resultados detallados de la evaluación, incluyendo métricas de seguridad clave (Recall, Falsos Positivos) y la selección del umbral de decisión, se encuentran en el notebook: `EvaluationsModels_Practical.ipynb`.
