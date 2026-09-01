# Interconnect — Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10.21-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.62.0-FF4B4B)
![Airflow](https://img.shields.io/badge/Apache%20Airflow-3.3.1-017CEE)
![Tests](https://img.shields.io/badge/Tests-29%20passed-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

Proyecto de Ciencia de Datos orientado a la predicción de cancelación de
clientes (*customer churn*) para la empresa de telecomunicaciones
**Interconnect**.

El proyecto desarrolla un sistema de Machine Learning capaz de estimar la
probabilidad de cancelación de un cliente y convertir esa predicción en una
herramienta operacional mediante:

- un pipeline de Machine Learning reproducible;
- una API REST desarrollada con FastAPI;
- un dashboard interactivo desarrollado con Streamlit;
- monitoreo de data drift mediante Population Stability Index (PSI);
- automatización del monitoreo con Apache Airflow;
- pruebas automatizadas para API, dashboard y monitoring.

---

## ⭐ Project Highlights

- **7,043 clientes** analizados.
- Modelo final: **CatBoostClassifier + One-Hot Encoding**.
- **AUC-ROC CV:** 0.8506.
- **AUC-ROC Test:** 0.8440.
- **Accuracy Test:** 0.8077.
- Pipeline completo de preprocessing + modelo serializado con `joblib`.
- API REST de predicción con **FastAPI**.
- Dashboard ejecutivo y predictor interactivo con **Streamlit**.
- Validación de entradas mediante **Pydantic**.
- Interpretabilidad mediante **feature importance y SHAP**.
- Monitoreo de drift basado en **Population Stability Index (PSI)**.
- Perfil de referencia persistente para monitoreo.
- Simulación controlada de data drift.
- Automatización diaria mediante **Apache Airflow 3.3.1**.
- **29 pruebas automatizadas**.
- Separación entre entorno de Machine Learning y entorno de Airflow.
- Pipeline reproducible con **Conda y pip**.

---

## 🎯 Objetivo de negocio

Interconnect busca reducir la pérdida de clientes mediante la identificación
anticipada de usuarios con mayor riesgo de cancelar sus servicios.

La cancelación de clientes representa un problema relevante para una empresa
de telecomunicaciones porque puede:

- reducir ingresos recurrentes;
- aumentar los costos asociados con adquisición de nuevos clientes;
- disminuir el valor de vida del cliente;
- generar pérdida de oportunidades comerciales.

El objetivo del proyecto consiste en utilizar información contractual,
demográfica, de servicios y facturación para desarrollar un sistema que
permita:

- identificar clientes con mayor probabilidad de churn;
- generar un score de riesgo;
- priorizar campañas de retención;
- apoyar la segmentación de clientes;
- utilizar de manera más eficiente los recursos comerciales.

El problema se aborda como una tarea de **clasificación binaria**:

```text
Churn = 1 → el cliente canceló el servicio
Churn = 0 → el cliente permanece activo
```

La métrica principal utilizada durante el desarrollo es **AUC-ROC**,
complementada con:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- matriz de confusión.

---

## 📊 Datos

El proyecto integra diferentes fuentes de información relacionadas con los
clientes de Interconnect:

- información contractual;
- características personales;
- servicios de Internet;
- servicios telefónicos;
- cargos mensuales;
- cargos acumulados;
- métodos de pago;
- servicios adicionales contratados.

El dataset integrado contiene:

```text
7,043 clientes
```

Los datasets originales no se incluyen en este repositorio.

Para ejecutar los notebooks localmente, los archivos originales deben
colocarse dentro de:

```text
data/
└── final_provider/
```

Durante el procesamiento se generan archivos intermedios:

```text
data/
├── final_provider/          # datasets originales
├── interconnect_raw.csv     # dataset integrado
└── interconnect_clean.csv   # dataset limpio para modelado
```

La carpeta `data/` está excluida del control de versiones mediante
`.gitignore`.

---

## 🔎 Metodología

El proyecto sigue un flujo completo de Ciencia de Datos y Machine Learning:

```text
Comprensión de datos
        ↓
Integración
        ↓
Limpieza
        ↓
EDA
        ↓
Feature Engineering
        ↓
Train/Test Split
        ↓
Preprocessing
        ↓
Comparación de modelos
        ↓
Optimización
        ↓
Selección del modelo
        ↓
Evaluación final
        ↓
Interpretabilidad
        ↓
Serialización
        ↓
API REST
        ↓
Dashboard
        ↓
Drift Monitoring
        ↓
Airflow
```

---

## 1. Comprensión e integración de datos

Se analizaron las distintas fuentes de información antes de realizar su
integración.

Las principales tareas incluyeron:

- revisión inicial de las fuentes;
- análisis de llaves de identificación;
- verificación de dimensiones;
- revisión de tipos de datos;
- identificación de relaciones entre tablas;
- integración de las fuentes;
- construcción del dataset maestro.

Notebook:

[`notebooks/01_comprension_e_integracion.ipynb`](notebooks/01_comprension_e_integracion.ipynb)

---

## 2. Limpieza y análisis exploratorio

La etapa de preparación inicial incluyó:

- revisión y conversión de tipos de datos;
- tratamiento de valores ausentes;
- identificación de inconsistencias;
- construcción de la variable objetivo;
- análisis de la distribución de churn;
- análisis de características numéricas;
- análisis de características categóricas;
- evaluación de patrones relacionados con churn;
- preparación del dataset para modelado.

Notebook:

[`notebooks/02_limpieza_y_eda.ipynb`](notebooks/02_limpieza_y_eda.ipynb)

---

## 3. Preparación y modelado

La etapa de Machine Learning incluyó:

- selección de variables predictoras;
- prevención de data leakage;
- división entrenamiento/prueba;
- preprocessing;
- construcción de pipelines;
- validación cruzada estratificada;
- establecimiento de baseline;
- comparación de algoritmos;
- optimización de hiperparámetros;
- evaluación final;
- análisis de generalización;
- interpretabilidad;
- exportación del pipeline para producción.

Notebook:

[`notebooks/03_preparacion_y_modelado.ipynb`](notebooks/03_preparacion_y_modelado.ipynb)

---

## 4. Informe de solución y estrategia de negocio

Los resultados se consolidaron en un informe que integra:

- evaluación del modelo;
- interpretación;
- variables predictivas principales;
- limitaciones;
- aplicación operacional;
- selección del threshold;
- recomendaciones de negocio;
- mejoras futuras.

Notebook:

[`notebooks/04_informe_solucion.ipynb`](notebooks/04_informe_solucion.ipynb)

Informe:

[`reports/informe_solucion.md`](reports/informe_solucion.md)

---

## 🔀 División de los datos

Para mantener una evaluación metodológicamente correcta, el conjunto de prueba
permaneció separado durante el desarrollo del modelo.

```text
Dataset total:    7,043 clientes
Training set:     5,634 clientes
Test set:         1,409 clientes
```

La división se realizó utilizando estratificación para mantener una
distribución similar de churn en ambos conjuntos.

El conjunto de prueba permaneció aislado durante:

- selección de variables;
- comparación de algoritmos;
- optimización de hiperparámetros;
- selección del modelo final.

Solo después de finalizar estas decisiones se utilizó para estimar la
capacidad de generalización del modelo.

---

## 📏 Estrategia de evaluación

## Métrica principal

**AUC-ROC**

AUC-ROC permite evaluar la capacidad del modelo para distinguir entre clientes
que cancelan y clientes que permanecen utilizando las probabilidades
estimadas.

Como métricas complementarias se utilizaron:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- matriz de confusión.

La validación durante entrenamiento utilizó:

```text
Stratified 5-Fold Cross Validation
```

---

## 🤖 Modelos evaluados

Durante el proyecto se compararon:

- DummyClassifier;
- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- CatBoost.

Todos los modelos se evaluaron bajo una estrategia consistente para permitir
una comparación metodológicamente válida.

---

## 📉 Baseline

Se utilizó `DummyClassifier` como referencia mínima.

Resultados aproximados:

| Métrica | Resultado |
| --- | ---: |
| AUC-ROC | **0.5000** |
| Accuracy | **0.7346** |

Un AUC-ROC cercano a `0.50` representa una capacidad discriminativa equivalente
al azar.

---

## 📈 Regresión Logística

La regresión logística se utilizó como modelo lineal de referencia.

Su inclusión permite:

- establecer una baseline interpretable;
- medir la ganancia obtenida por modelos no lineales;
- comparar complejidad frente a rendimiento.

Las variables categóricas se transformaron mediante One-Hot Encoding y las
variables continuas mediante escalamiento.

---

## 🌲 Comparación inicial de modelos

Se evaluaron modelos con distinta capacidad para capturar relaciones no
lineales.

Los modelos basados en árboles mostraron una capacidad discriminativa superior
a las alternativas más simples.

Los mejores candidatos fueron posteriormente optimizados mediante búsqueda de
hiperparámetros y validación cruzada.

---

## ⚙️ Optimización de modelos

## Gradient Boosting

Se evaluaron diferentes configuraciones del algoritmo para estudiar su
capacidad de modelar relaciones no lineales.

## CatBoost

CatBoost presentó el mejor desempeño global y fue seleccionado para una
optimización más profunda.

Los mejores hiperparámetros obtenidos fueron:

```python
{
    "learning_rate": 0.03,
    "l2_leaf_reg": 5,
    "iterations": 200,
    "depth": 4,
}
```

---

## 🧪 CatBoost: One-Hot Encoding vs categorías nativas

Se evaluaron dos estrategias:

```text
CatBoost + One-Hot Encoding
CatBoost + categorías nativas
```

La comparación permitió validar empíricamente cuál configuración ofrecía
mejor desempeño bajo el esquema de validación utilizado.

La configuración seleccionada fue:

```text
CatBoostClassifier + One-Hot Encoding
```

---

## 🏆 Modelo final

## CatBoostClassifier + One-Hot Encoding

El modelo final forma parte de un `Pipeline` de Scikit-learn que integra
preprocessing y estimador.

Esto permite utilizar exactamente las mismas transformaciones durante:

```text
training
   ↓
validation
   ↓
testing
   ↓
API inference
```

El artefacto de producción se encuentra en:

```text
models/churn_pipeline.joblib
```

El objeto serializado corresponde al pipeline completo y no únicamente al
estimador CatBoost.

---

## 🧩 Variables utilizadas por el modelo

El pipeline final utiliza **23 variables predictoras**.

## Variables continuas

```text
MonthlyCharges
TotalCharges
```

Se procesan mediante `StandardScaler`.

## Variables binarias/discretas

```text
SeniorCitizen
HasInternet
HasPhone
InternetAddOnCount
StreamingCount
HasTechProtection
AutomaticPayment
```

## Variables categóricas

```text
Type
PaperlessBilling
PaymentMethod
gender
Partner
Dependents
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
MultipleLines
```

Las variables categóricas utilizan:

```python
OneHotEncoder(handle_unknown="ignore")
```

---

## 🛡️ Prevención de data leakage

Las siguientes variables no forman parte del conjunto de predictores:

```text
Churn
customerID
EndDate
EndDateParsed
EffectiveEndDate
BeginDate
HistoricalTenure
```

`HistoricalTenure` no se utiliza como predictor oficial porque su construcción
puede depender de información disponible después de la cancelación.

En un escenario real, cualquier variable de antigüedad debe calcularse
utilizando únicamente información disponible al momento de la predicción.

---

## 📈 Resultados finales

Resultados oficiales del modelo:

| Métrica | Resultado |
| --- | ---: |
| CV AUC-ROC | **0.8506** |
| Test AUC-ROC | **0.8440** |
| Test Accuracy | **0.8077** |
| Precision — Churn | **0.67** |
| Recall — Churn | **0.53** |
| F1-score — Churn | **0.60** |

Valores completos de AUC-ROC:

```text
CV AUC-ROC:    0.850601
Test AUC-ROC:  0.843972
```

Threshold oficial:

```text
0.50
```

---

## 📊 Generalización

La diferencia entre validación cruzada y test es pequeña:

```text
CV AUC-ROC      ≈ 0.8506
Test AUC-ROC    ≈ 0.8440
```

Esto indica que el rendimiento observado durante entrenamiento se mantiene de
forma razonablemente consistente sobre datos no utilizados durante la
optimización.

---

## 🎯 Rendimiento sobre la clase Churn

Sobre los clientes que realmente cancelaron:

```text
Precision: 0.67
Recall:    0.53
F1-score:  0.60
```

El recall muestra que el modelo detecta aproximadamente la mitad de los
clientes que efectivamente cancelan utilizando el threshold oficial de `0.50`.

La probabilidad generada por el modelo también puede utilizarse directamente
como **score de riesgo** para priorizar campañas sin modificar el modelo
entrenado.

---

## 🧮 Matriz de confusión

Resultados sobre el conjunto de prueba:

| | Predicción No Churn | Predicción Churn |
| --- | ---: | ---: |
| **Real No Churn** | 938 | 97 |
| **Real Churn** | 174 | 200 |

Equivalente a:

```text
TN = 938
FP = 97
FN = 174
TP = 200
```

El conjunto de test contiene:

```text
1,409 clientes
1,035 No Churn
374 Churn
```

---

## 🔍 Interpretabilidad del modelo

La interpretación del modelo se realizó mediante:

- feature importance;
- análisis SHAP.

Estas técnicas ayudan a comprender qué variables tienen mayor peso predictivo
en las decisiones del modelo.

---

## 📊 Principales variables predictivas

Entre las variables con mayor importancia se encuentran:

| Variable | Importancia aproximada |
| --- | ---: |
| Type | 28.12 |
| TotalCharges | 21.52 |
| InternetService | 14.05 |
| MonthlyCharges | 6.86 |
| OnlineSecurity | 4.49 |
| TechSupport | 4.28 |
| PaymentMethod | 4.24 |

Estas importancias indican **relevancia predictiva**, no causalidad.

---

## ⚠️ Importancia predictiva ≠ causalidad

Una variable con importancia elevada no implica necesariamente que modificar
esa variable produzca directamente una reducción del churn.

El modelo identifica asociaciones presentes en los datos históricos.

Las decisiones comerciales deben complementarse con:

- conocimiento del negocio;
- experimentación;
- pruebas A/B;
- análisis causal cuando sea necesario.

---

## 💼 Aplicación de negocio

El modelo permite transformar cada cliente en una probabilidad:

```text
P(Churn = 1)
```

Esta probabilidad puede utilizarse como score para priorizar acciones de
retención.

Ejemplo:

```text
Cliente
   ↓
Datos contractuales y de servicios
   ↓
Pipeline ML
   ↓
Probabilidad de churn
   ↓
Nivel de riesgo
   ↓
Acción comercial
```

---

## 🎯 Segmentación operacional de riesgo

La aplicación utiliza tres bandas descriptivas:

```text
LOW     < 0.30
MEDIUM  0.30 – 0.59
HIGH    ≥ 0.60
```

Estas bandas sirven únicamente para facilitar la interpretación en la
interfaz.

**No sustituyen el threshold oficial de clasificación del modelo**, que
permanece en:

```text
0.50
```

---

## 💡 Recomendaciones de negocio

El score de churn puede utilizarse para:

- priorizar clientes con mayor riesgo;
- adaptar campañas de retención;
- ofrecer incentivos diferenciados;
- revisar contratos con mayor exposición al churn;
- analizar servicios asociados con abandono;
- optimizar recursos del equipo comercial.

La estrategia ideal debe considerar además:

```text
probabilidad de churn
+
valor económico del cliente
+
costo de intervención
+
beneficio esperado
```

---

## ⚖️ Selección del threshold

El modelo oficial utiliza:

```text
threshold = 0.50
```

Sin embargo, el threshold puede ajustarse según el objetivo de negocio.

Reducirlo puede aumentar el recall y permitir detectar más clientes con riesgo,
pero también puede incrementar falsos positivos.

Por esta razón, cualquier cambio futuro debería considerar costos comerciales
y no únicamente métricas estadísticas.

---

## 🚀 Arquitectura del proyecto

El proyecto separa entrenamiento, inferencia, visualización, monitoreo y
orquestación.

```text
                      ┌──────────────────────────┐
                      │    Datos de clientes     │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │ Limpieza + EDA + Feature │
                      │ Engineering              │
                      │ notebooks/               │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │ Pipeline de ML           │
                      │ CatBoost + OHE           │
                      └────────────┬─────────────┘
                                   │
                                   ▼
                      ┌──────────────────────────┐
                      │ churn_pipeline.joblib    │
                      │ models/                  │
                      └────────────┬─────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
                  ▼                                 ▼
       ┌──────────────────────┐          ┌──────────────────────┐
       │ FastAPI              │          │ Streamlit            │
       │ api/                 │◄─────────│ dashboard/           │
       │ REST Prediction API  │          │ Executive Dashboard  │
       └──────────────────────┘          └──────────────────────┘


       ┌────────────────────────────────────────────────────────┐
       │                 MODEL MONITORING                       │
       │                                                        │
       │ reference_profile.json                                 │
       │          ↓                                             │
       │ current_batch.csv                                      │
       │          ↓                                             │
       │ monitoring/drift.py                                    │
       │          ↓                                             │
       │ PSI drift report                                       │
       └─────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
                      ┌──────────────────────────┐
                      │ Apache Airflow           │
                      │ Daily orchestration      │
                      └──────────────────────────┘
```

---

## 🌐 REST API — FastAPI

La inferencia del modelo se expone mediante una API REST implementada con
FastAPI.

Archivo principal:

```text
api/main.py
```

Esquemas y validación:

```text
api/schemas.py
```

Endpoints disponibles:

| Método | Endpoint | Descripción |
| --- | --- | --- |
| GET | `/` | Estado general del servicio |
| GET | `/health` | Health check del modelo |
| GET | `/model-info` | Información del modelo |
| POST | `/predict` | Predicción de churn |

---

## Validación de entradas

La API utiliza Pydantic para verificar:

- tipos de datos;
- cargos no negativos;
- variables binarias;
- consistencia de telefonía;
- consistencia de servicios de Internet;
- presencia de todas las variables requeridas.

Ejemplo:

```text
HasPhone = 0
→ MultipleLines debe ser "No phone service"
```

y:

```text
HasInternet = 0
→ InternetService debe ser "No internet service"
```

junto con los servicios adicionales correspondientes.

---

## Ejemplo de respuesta

La API devuelve información con una estructura conceptual similar a:

```json
{
  "churn_probability": 0.26085351452633293,
  "churn_prediction": 0,
  "risk_level": "LOW"
}
```

---

## 📊 Dashboard — Streamlit

El proyecto incluye un dashboard desarrollado con Streamlit.

Archivo:

```text
dashboard/app.py
```

El dashboard consume la **API FastAPI** en lugar de cargar directamente el
artefacto `joblib`.

Esto mantiene separación entre:

```text
modelo
API
interfaz
```

El dashboard incluye cuatro áreas principales:

1. **Executive Overview**
2. **Model Performance**
3. **Model Monitoring**
4. **Customer Risk Prediction**

---

## Executive Overview

Presenta las métricas principales:

```text
CV AUC-ROC       0.8506
Test AUC-ROC     0.8440
Accuracy         0.8077
Recall Churn     0.53
```

---

## Model Performance

Incluye:

- matriz de confusión;
- importancia de variables;
- métricas del modelo.

---

## Customer Risk Prediction

El usuario puede introducir las características de un cliente y obtener:

```text
probabilidad de churn
predicción binaria
nivel de riesgo
```

Las variables derivadas se calculan automáticamente desde la interfaz para
mantener consistencia con el pipeline.

---

## 📡 Monitoreo de Data Drift

El proyecto implementa un sistema de monitoreo basado en:

**Population Stability Index — PSI**

Archivo principal:

```text
monitoring/drift.py
```

El objetivo es comparar la distribución utilizada como referencia durante
entrenamiento con lotes posteriores de datos.

---

## Umbrales PSI

Los estados definidos son:

| PSI | Estado |
| --- | --- |
| `< 0.10` | Stable |
| `0.10 – < 0.25` | Moderate |
| `>= 0.25` | Significant |

---

## 🧬 Reference Profile

El perfil de referencia se construye utilizando exclusivamente:

```text
X_train
```

y se almacena en:

```text
monitoring/reference_profile.json
```

Para variables numéricas almacena información como:

- media;
- desviación estándar;
- mínimo;
- máximo;
- límites de bins;
- proporciones.

Para variables categóricas almacena la distribución relativa de cada
categoría.

---

## 🔬 Variables monitorizadas

El sistema monitorea:

## Numéricas

```text
MonthlyCharges
TotalCharges
```

## Categóricas

```text
Type
PaymentMethod
InternetService
PaperlessBilling
MultipleLines
```

---

## 🧪 Validación del sistema de drift

Se implementaron dos escenarios.

## Validation batch

El conjunto de test se utiliza únicamente como un lote independiente de
demostración para validar el funcionamiento del monitor.

Resultados:

| Variable | PSI | Estado |
| --- | ---: | --- |
| TotalCharges | 0.016154 | stable |
| MonthlyCharges | 0.009980 | stable |
| Type | 0.005251 | stable |
| PaymentMethod | 0.003691 | stable |
| InternetService | 0.001412 | stable |
| MultipleLines | 0.001216 | stable |
| PaperlessBilling | 0.000187 | stable |

Todos los valores permanecen por debajo de:

```text
PSI = 0.10
```

Este escenario demuestra que el sistema puede evaluar un lote independiente,
pero **no representa monitoreo real de producción**.

---

## Synthetic drift simulation

Para comprobar que el detector realmente responde ante cambios
distribucionales se creó una simulación controlada.

Se modificaron:

```text
MonthlyCharges
Type
```

Resultados principales:

| Variable | PSI | Estado |
| --- | ---: | --- |
| Type | 5.779197 | significant |
| MonthlyCharges | 5.727768 | significant |
| TotalCharges | 0.016154 | stable |
| PaymentMethod | 0.003691 | stable |
| InternetService | 0.001412 | stable |
| MultipleLines | 0.001216 | stable |
| PaperlessBilling | 0.000187 | stable |

El experimento demuestra que el sistema:

```text
mantiene stable variables sin cambios
+
detecta significant drift cuando la distribución cambia
```

---

## ⚙️ Pipeline operativo de monitoring

El script operacional se encuentra en:

```text
airflow/scripts/run_monitoring.py
```

Su flujo es:

```text
reference_profile.json
        ↓
current_batch.csv
        ↓
generate_drift_report_from_profile()
        ↓
drift_report_latest.csv
```

Los archivos generados en tiempo de ejecución:

```text
monitoring/current_batch.csv
monitoring/drift_report_latest.csv
```

están excluidos del control de versiones.

---

## 🛫 Automatización con Apache Airflow

El monitoreo se automatiza mediante:

```text
Apache Airflow 3.3.1
```

El DAG se encuentra en:

```text
airflow/dags/interconnect_monitoring_dag.py
```

DAG ID:

```text
interconnect_churn_monitoring
```

Frecuencia:

```text
@daily
```

---

## Flujo del DAG

```text
interconnect_churn_monitoring
          │
          ▼
validate_monitoring_script
          │
          ▼
run_drift_monitoring
          │
          ▼
run_monitoring.py
          │
          ▼
drift_report_latest.csv
```

Las tareas realizan:

### `validate_monitoring_script`

Comprueba que el script operacional existe antes de iniciar el monitoring.

### `run_drift_monitoring`

Ejecuta el pipeline, registra stdout/stderr y falla explícitamente si el
proceso termina con un código distinto de cero.

---

## Validación end-to-end

El DAG fue probado mediante:

```bash
airflow dags test \
  interconnect_churn_monitoring \
  2026-08-31
```

Resultado:

```text
validate_monitoring_script → success
run_drift_monitoring       → success
DagRun                      → success
```

Esto valida la ejecución completa:

```text
Airflow
   ↓
script operacional
   ↓
drift engine
   ↓
reference profile
   ↓
current batch
   ↓
drift report
```

---

## 🧪 Tests automatizados

El proyecto incluye pruebas automatizadas con `pytest`.

Actualmente:

```text
29 passed
```

La suite cubre tres componentes principales.

## API

Archivo:

```text
tests/test_api.py
```

Incluye pruebas sobre:

- root endpoint;
- health check;
- información del modelo;
- predicción válida;
- probabilidad de referencia;
- entradas inconsistentes;
- cargos negativos;
- campos faltantes;
- validación de `SeniorCitizen`;
- combinaciones inválidas de Internet;
- estructura de respuesta;
- tipos de datos de salida.

## Dashboard

Archivo:

```text
tests/test_dashboard.py
```

Valida la construcción de variables derivadas como:

- disponibilidad de Internet;
- telefonía;
- servicios adicionales;
- pagos automáticos;
- protección tecnológica.

## Drift monitoring

Archivo:

```text
tests/test_drift.py
```

Valida:

- PSI numérico;
- PSI categórico;
- clasificación de drift;
- creación del reference profile;
- persistencia JSON;
- generación del reporte desde el perfil;
- detección de cambios distribucionales.

---

## 📁 Estructura del repositorio

```text
interconnect-churn-prediction/
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── airflow/
│   ├── dags/
│   │   └── interconnect_monitoring_dag.py
│   └── scripts/
│       └── run_monitoring.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   └── churn_pipeline.joblib
│
├── monitoring/
│   ├── __init__.py
│   ├── drift.py
│   ├── reference_profile.json
│   ├── drift_report_validation.csv
│   └── drift_report_synthetic.csv
│
├── notebooks/
│   ├── 01_comprension_e_integracion.ipynb
│   ├── 02_limpieza_y_eda.ipynb
│   ├── 03_preparacion_y_modelado.ipynb
│   └── 04_informe_solucion.ipynb
│
├── reports/
│   └── informe_solucion.md
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_dashboard.py
│   └── test_drift.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── requirements-airflow.txt
├── LICENSE
└── README.md
```

Los archivos de runtime no se versionan:

```text
data/
airflow/runtime/
monitoring/current_batch.csv
monitoring/drift_report_latest.csv
```

---

## 🛠️ Tecnologías utilizadas

## Data Science

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Matplotlib
- Seaborn
- SHAP
- JupyterLab

## Deployment

- FastAPI
- Uvicorn
- Pydantic

## Dashboard

- Streamlit
- Requests

## Monitoring

- Pandas
- NumPy
- PSI

## Orquestación

- Apache Airflow

## Calidad

- Pytest
- Git
- GitHub

---

## 📦 Versiones principales

Entorno principal:

| Tecnología | Versión |
| --- | --- |
| Python | 3.10.21 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.7.2 |
| Matplotlib | 3.10.9 |
| Seaborn | 0.13.2 |
| CatBoost | 1.2.10 |
| Joblib | 1.5.3 |
| FastAPI | 0.141.1 |
| Uvicorn | 0.52.4 |
| Pydantic | 2.13.5 |
| Starlette | 1.6.0 |
| Requests | 2.34.2 |
| HTTPX | 0.28.1 |
| Streamlit | 1.62.0 |
| Pytest | 9.1.1 |
| JupyterLab | 4.6.3 |

Airflow:

| Tecnología | Versión |
| --- | --- |
| Python | 3.10 |
| Apache Airflow | 3.3.1 |
| NumPy | 2.2.6 |
| Pandas | 2.3.3 |

---

## 🧪 Instalación y entorno de ejecución

El proyecto utiliza dos entornos separados:

```text
interconnect-churn
interconnect-airflow
```

Esta separación evita introducir las numerosas dependencias de Airflow dentro
del entorno principal de Machine Learning.

---

## Opción 1 — Conda

Crear el entorno principal:

```bash
conda env create -f environment.yml
```

Activarlo:

```bash
conda activate interconnect-churn
```

---

## Opción 2 — pip

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en macOS/Linux:

```bash
source .venv/bin/activate
```

Instalar:

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución del proyecto

## 1. API

Activar:

```bash
conda activate interconnect-churn
```

Ejecutar:

```bash
uvicorn api.main:app --reload
```

La API estará disponible en:

```text
http://127.0.0.1:8000
```

Documentación Swagger:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

## 2. Dashboard

Con la API ejecutándose:

```bash
streamlit run dashboard/app.py
```

Streamlit mostrará la URL local en Terminal.

---

## 3. Tests

Desde la raíz del proyecto:

```bash
pytest -v
```

Resultado esperado:

```text
29 passed
```

---

## 4. Monitoring manual

El lote actual debe existir en:

```text
monitoring/current_batch.csv
```

Después:

```bash
python airflow/scripts/run_monitoring.py
```

El script genera:

```text
monitoring/drift_report_latest.csv
```

---

## 🛫 Configuración de Airflow

Airflow utiliza un entorno dedicado.

Crear el entorno:

```bash
conda create -n interconnect-airflow python=3.10 -y
```

Activarlo:

```bash
conda activate interconnect-airflow
```

---

## Instalar Apache Airflow

Definir versión:

```bash
AIRFLOW_VERSION=3.3.1
```

Obtener versión de Python:

```bash
PYTHON_VERSION="$(
python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"
```

Construir la URL oficial de constraints:

```bash
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
```

Instalar:

```bash
pip install \
  "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "${CONSTRAINT_URL}"
```

Instalar las dependencias utilizadas por el script de monitoring cuando sea
necesario:

```bash
pip install \
  numpy==2.2.6 \
  pandas==2.3.3
```

Comprobar:

```bash
airflow version
pip check
```

---

## Configurar runtime de Airflow

Desde la raíz del proyecto:

```bash
export AIRFLOW_HOME="$PWD/airflow/runtime"
```

Definir la carpeta de DAGs:

```bash
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/airflow/dags"
```

Inicializar/migrar la metadata database:

```bash
airflow db migrate
```

Comprobar:

```bash
airflow db check
```

---

## Validar DAG

Parseo local:

```bash
airflow dags list --local
```

Validar errores:

```bash
airflow dags list-import-errors --local
```

El DAG esperado es:

```text
interconnect_churn_monitoring
```

---

## Probar DAG end-to-end

```bash
airflow dags test \
  interconnect_churn_monitoring \
  2026-08-31
```

Resultado esperado:

```text
Dag run in success state
```

---

## 📄 Informe de solución

El análisis de negocio completo se encuentra en:

```text
reports/informe_solucion.md
```

El informe integra:

- objetivo de negocio;
- metodología;
- modelo seleccionado;
- evaluación;
- interpretabilidad;
- estrategia de retención;
- threshold;
- limitaciones;
- recomendaciones;
- futuras mejoras.

---

## ⚠️ Limitaciones

## Rendimiento predictivo

Un AUC-ROC de aproximadamente `0.844` representa una buena capacidad
discriminativa, pero el modelo no identifica perfectamente todos los casos.

---

## Recall

El recall de churn es aproximadamente:

```text
0.53
```

Por lo tanto, con threshold `0.50`, una parte de los clientes que cancelan no
es identificada correctamente.

---

## Información disponible

El modelo está limitado por las variables disponibles en el dataset.

Variables potencialmente útiles en un entorno real podrían incluir:

- interacciones con soporte;
- reclamos;
- consumo histórico;
- cambios de plan;
- comportamiento de pagos en el tiempo;
- satisfacción del cliente;
- uso de servicios;
- eventos previos a cancelación.

---

## ⏳ Limitación temporal: HistoricalTenure

Durante el análisis se estudió información temporal relacionada con la
duración histórica del cliente.

Sin embargo, `HistoricalTenure` no forma parte del modelo oficial porque su
construcción puede depender de información futura para clientes cancelados.

En producción, la antigüedad debe calcularse únicamente utilizando:

```text
fecha de predicción - BeginDate
```

sin utilizar información posterior al momento de scoring.

---

## 📡 Limitaciones del monitoring actual

El proyecto implementa la infraestructura necesaria para detectar data drift,
pero actualmente utiliza datasets históricos como demostración.

El escenario:

```text
Validation batch
```

utiliza `X_test`.

Esto **no debe interpretarse como evidencia de ausencia de drift en
producción**.

En un sistema real, `current_batch.csv` debería sustituirse por datos
recibidos durante operación.

---

## 🚀 Futuras mejoras

El proyecto puede extenderse mediante:

- captura automática de datos de inferencia;
- monitoreo continuo de predicciones;
- model performance monitoring cuando exista ground truth;
- alertas ante PSI significativo;
- integración con Slack, correo o sistemas de incidentes;
- almacenamiento histórico de drift reports;
- dashboards temporales de drift;
- retraining periódico;
- model registry;
- experiment tracking;
- contenedores Docker;
- CI/CD;
- despliegue cloud;
- autenticación de la API;
- rate limiting;
- persistencia de predicciones;
- explainability individual por cliente.

---

## 🕒 Snapshots temporales

Un sistema productivo debería generar snapshots de clientes en distintos
momentos:

```text
cliente t0
cliente t1
cliente t2
...
```

Esto permitiría construir modelos que representen mejor la evolución previa al
churn.

---

## 🧪 Validación futura

Una evolución natural sería evaluar el sistema mediante validación temporal:

```text
train → periodo histórico
test  → periodo posterior
```

Esto proporcionaría una aproximación todavía más realista al uso futuro del
modelo.

---

## 🧪 Experimentos adicionales

Posibles extensiones del modelado:

- XGBoost;
- LightGBM;
- calibración de probabilidades;
- optimización del threshold basada en costo;
- cost-sensitive learning;
- undersampling/oversampling;
- feature engineering temporal;
- ensembles;
- análisis causal;
- survival analysis.

Estas alternativas deben compararse utilizando el mismo procedimiento de
validación para mantener consistencia metodológica.

---

## 📌 Resultado principal

El proyecto demuestra un flujo completo de Machine Learning que va más allá
del entrenamiento de un modelo:

```text
Datos
  ↓
EDA
  ↓
Feature Engineering
  ↓
Machine Learning
  ↓
Evaluación
  ↓
Interpretabilidad
  ↓
Modelo serializado
  ↓
FastAPI
  ↓
Streamlit
  ↓
Drift Monitoring
  ↓
Apache Airflow
```

El modelo final alcanza:

```text
CV AUC-ROC:    0.8506
Test AUC-ROC:  0.8440
Accuracy:      0.8077
```

y queda integrado dentro de una arquitectura que permite realizar
predicciones, visualizar resultados, validar entradas, monitorear cambios
distribucionales y automatizar el proceso de monitoring.

---

## 👤 Autor

**Alan Calderón Jiménez**

Proyecto desarrollado como parte de formación profesional en Ciencia de Datos
y Machine Learning.

---

## 📚 Proyecto

**Interconnect — Customer Churn Prediction**

Proyecto end-to-end de Ciencia de Datos, Machine Learning, deployment y
monitoring aplicado a predicción de churn en telecomunicaciones.
