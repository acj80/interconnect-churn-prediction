# Interconnect — Customer Churn Prediction

Proyecto de Ciencia de Datos orientado a la predicción de cancelación de clientes (*customer churn*) para la empresa de telecomunicaciones **Interconnect**.

El proyecto desarrolla un modelo de Machine Learning capaz de estimar la probabilidad de cancelación de cada cliente con el objetivo de apoyar al área de Marketing en la priorización de campañas y estrategias preventivas de retención.

---

## 🎯 Objetivo de negocio

Interconnect busca reducir la pérdida de clientes mediante la identificación anticipada de usuarios con mayor riesgo de cancelar sus servicios.

La cancelación de clientes representa un problema relevante para una empresa de telecomunicaciones, ya que puede afectar los ingresos recurrentes e incrementar la necesidad de adquirir nuevos clientes.

El objetivo del proyecto es utilizar información contractual, demográfica, de servicios y facturación para desarrollar un modelo que permita:

* identificar clientes con mayor probabilidad de churn;
* generar un score de riesgo para cada cliente;
* priorizar acciones de retención;
* apoyar la segmentación de clientes;
* utilizar de forma más eficiente los recursos destinados a campañas comerciales.

El problema se aborda como una tarea de **clasificación binaria**:

* `Churn = 1`: el cliente canceló el servicio.
* `Churn = 0`: el cliente permanece activo.

La métrica principal utilizada durante el desarrollo es **AUC-ROC**, complementada con Accuracy, Precision, Recall y F1-score.

---

## 📊 Datos

El proyecto integra diferentes fuentes de información relacionadas con los clientes de Interconnect:

* información contractual;
* características personales;
* servicios de Internet;
* servicios telefónicos;
* cargos mensuales;
* cargos acumulados;
* métodos de pago;
* servicios adicionales contratados.

Los datasets originales no se incluyen en este repositorio.

Para ejecutar los notebooks localmente, los archivos deben colocarse en la estructura correspondiente dentro de:

```text
data/
└── final_provider/
```

La carpeta `data/` está excluida del control de versiones mediante `.gitignore`.

---

## 🔎 Metodología

El proyecto se desarrolló siguiendo un flujo completo de Ciencia de Datos.

### 1. Comprensión e integración de datos

Se analizaron las diferentes fuentes de información, sus estructuras, tipos de variables y relaciones entre tablas antes de realizar su integración.

Notebook:

`notebooks/01_comprension_e_integracion.ipynb`

### 2. Limpieza y análisis exploratorio

Se realizaron tareas de:

* revisión y conversión de tipos de datos;
* tratamiento de valores ausentes;
* identificación de inconsistencias;
* construcción de la variable objetivo;
* análisis de la distribución de churn;
* análisis exploratorio de características numéricas y categóricas;
* evaluación de patrones asociados con la cancelación.

Notebook:

`notebooks/02_limpieza_y_eda.ipynb`

### 3. Preparación y modelado

La etapa de Machine Learning incluyó:

* selección de variables predictoras;
* separación de entrenamiento y prueba;
* transformación de variables numéricas y categóricas;
* construcción de pipelines;
* validación cruzada estratificada de cinco folds;
* establecimiento de modelos baseline;
* comparación de distintos algoritmos;
* optimización de hiperparámetros;
* evaluación de CatBoost con One-Hot Encoding y categorías nativas;
* selección del modelo final;
* evaluación sobre el conjunto de prueba;
* análisis de importancia de variables;
* interpretación mediante valores SHAP.

Notebook:

`notebooks/03_preparacion_y_modelado.ipynb`

### 4. Estrategia de negocio

Los resultados predictivos fueron traducidos a posibles estrategias de retención y utilización operacional del modelo.

Notebook:

`notebooks/04_estrategia_de_negocio.ipynb`

---

## 🤖 Modelos evaluados

Durante el proyecto se compararon diferentes algoritmos utilizando el mismo esquema de validación cruzada y las mismas métricas.

Entre los modelos evaluados se encuentran:

* DummyClassifier;
* Logistic Regression;
* Random Forest;
* Gradient Boosting;
* CatBoost.

El modelo Dummy estableció una línea base de:

| Métrica  | Resultado aproximado |
| -------- | -------------------: |
| AUC-ROC  |               0.5000 |
| Accuracy |               0.7346 |

La Regresión Logística mejoró considerablemente este resultado:

| Métrica  | Resultado CV |
| -------- | -----------: |
| AUC-ROC  |       0.8399 |
| Accuracy |       0.8007 |

Posteriormente, Gradient Boosting y CatBoost mostraron los mejores resultados entre los modelos no lineales y fueron seleccionados como principales candidatos para optimización.

---

## ⚙️ Optimización y selección del modelo

Los modelos con mejor desempeño fueron optimizados utilizando búsqueda de hiperparámetros y validación cruzada estratificada.

También se realizó un experimento específico para comparar dos formas de utilizar CatBoost:

1. variables categóricas transformadas mediante **One-Hot Encoding**;
2. manejo **nativo de variables categóricas**.

En los modelos baseline, el manejo nativo de categorías mejoró el desempeño de CatBoost.

Sin embargo, después de la optimización ambas alternativas presentaron resultados muy similares:

```text
CatBoost OHE Tuned     → AUC-ROC CV ≈ 0.8506
CatBoost Native Tuned  → AUC-ROC CV ≈ 0.8502
```

La configuración con **One-Hot Encoding** obtuvo el mejor AUC-ROC promedio durante la validación cruzada y fue seleccionada antes de evaluar el conjunto de prueba.

---

## 🏆 Modelo final

El modelo seleccionado fue:

**CatBoostClassifier + One-Hot Encoding**

El conjunto de prueba permaneció aislado durante:

* selección de variables;
* comparación de algoritmos;
* selección de hiperparámetros;
* optimización de modelos.

Únicamente después de seleccionar el modelo final se utilizó el conjunto de prueba para estimar su capacidad de generalización.

---

## 📈 Resultados finales

El modelo final obtuvo los siguientes resultados:

| Métrica                      |  Resultado |
| ---------------------------- | ---------: |
| AUC-ROC — Validación cruzada | **0.8506** |
| AUC-ROC — Test               | **0.8440** |
| Accuracy — Test              | **0.8077** |
| Precision — Churn            |   **0.67** |
| Recall — Churn               |   **0.53** |
| F1-score — Churn             |   **0.60** |

Los valores calculados directamente durante la evaluación fueron:

```text
CV AUC-ROC:     0.850601
Test AUC-ROC:   0.843972
Test Accuracy:  0.807665
```

La diferencia entre el AUC-ROC obtenido mediante validación cruzada y el conjunto de prueba fue aproximadamente:

```text
-0.0066
```

Esta reducción relativamente pequeña sugiere que el modelo mantiene un comportamiento razonablemente consistente sobre datos no utilizados durante su desarrollo.

---

## 🎯 Comportamiento sobre la clase Churn

En el conjunto de prueba existían:

```text
374 clientes que realmente cancelaron
```

Utilizando el threshold estándar de `0.5`, el modelo identificó correctamente:

```text
200 clientes churn
```

mientras que:

```text
174 clientes churn
```

fueron clasificados como clientes que permanecerían.

Esto se refleja en un Recall para la clase Churn de aproximadamente:

```text
0.53
```

Por lo tanto, aunque el modelo presenta una capacidad discriminativa útil, una implementación comercial debería analizar cuidadosamente el umbral de decisión.

El threshold de `0.5` no necesariamente representa la opción económicamente óptima para una estrategia de retención.

---

## 🔍 Interpretabilidad del modelo

La importancia interna de CatBoost y el análisis mediante valores SHAP mostraron resultados consistentes.

Las variables con mayor relevancia predictiva fueron:

1. `Type`
2. `TotalCharges`
3. `InternetService`
4. `PaymentMethod`
5. `MonthlyCharges`
6. `TechSupport`
7. `OnlineSecurity`

El **tipo de contrato** presentó la mayor relevancia predictiva, seguido de los cargos acumulados y el tipo de servicio de Internet.

Estos resultados son consistentes con patrones identificados durante el análisis exploratorio.

Sin embargo, las importancias representan **asociaciones predictivas y no relaciones causales**.

Por lo tanto, no puede concluirse que modificar una característica individual produzca directamente una reducción del churn.

---

## 💼 Aplicación de negocio

La principal utilidad del modelo no consiste únicamente en generar una clasificación:

```text
Churn / No Churn
```

sino en producir una:

```text
probabilidad estimada de churn
```

para cada cliente.

Esta probabilidad puede utilizarse como un **score de riesgo**.

Un posible flujo operativo sería:

```text
Clientes activos
       ↓
Preparación de variables
       ↓
Modelo de churn
       ↓
Probabilidad de cancelación
       ↓
Ranking de riesgo
       ↓
Segmentación
       ↓
Campaña de retención
       ↓
Medición de resultados
```

De esta manera, Marketing podría concentrar sus recursos en los clientes con mayor riesgo estimado en lugar de aplicar campañas generales a toda la cartera.

---

## 💡 Recomendaciones de negocio

Los resultados sugieren analizar especialmente clientes asociados con características como:

* contratos de menor duración;
* determinados métodos de pago;
* cargos mensuales elevados;
* determinados servicios de Internet;
* ausencia de soporte técnico;
* ausencia de seguridad en línea.

La probabilidad generada por el modelo debería utilizarse como indicador principal de riesgo y estas características como información adicional para comprender y segmentar a los clientes.

Entre las posibles estrategias a evaluar se encuentran:

* incentivos para migrar de contratos mensuales hacia contratos de mayor duración;
* revisión de planes para clientes con cargos mensuales elevados;
* promociones relacionadas con soporte técnico y seguridad en línea;
* incentivos para determinados métodos de pago;
* campañas específicas para clientes con scores elevados de churn.

Estas acciones deben considerarse **hipótesis comerciales**, no consecuencias causales demostradas por el modelo.

Su efectividad debería evaluarse mediante experimentos controlados, como pruebas A/B.

---

## ⚖️ Selección del threshold

El umbral de clasificación utilizado durante la evaluación fue:

```text
0.5
```

Sin embargo, desde una perspectiva comercial, el threshold óptimo debería determinarse utilizando información económica como:

* costo de una acción de retención;
* valor esperado de conservar un cliente;
* costo de adquisición de nuevos clientes;
* pérdida económica asociada al churn;
* presupuesto disponible para campañas.

Reducir el threshold podría incrementar el Recall y detectar más clientes que posteriormente cancelarían, pero también produciría más falsos positivos y aumentaría el costo de las campañas.

Por esta razón, una futura implementación debería optimizar el punto de operación del modelo utilizando criterios de negocio y no exclusivamente métricas estadísticas.

---

## ⚠️ Limitaciones

El proyecto presenta varias limitaciones.

El modelo final obtiene un AUC-ROC cercano a `0.844`, por lo que todavía existe margen para mejorar su capacidad predictiva.

Además, utilizando el threshold estándar, el Recall de churn es aproximadamente `0.53`, lo que implica que una proporción relevante de las cancelaciones reales no es identificada.

El dataset tampoco dispone de determinadas variables que podrían aportar información adicional, como:

* satisfacción del cliente;
* número de reclamaciones;
* interrupciones del servicio;
* calidad de conexión;
* llamadas al centro de atención;
* historial de promociones;
* modificaciones recientes del plan;
* comportamiento de uso;
* interacciones recientes con soporte.

Otra limitación importante está relacionada con la medición de la antigüedad del cliente.

La variable `HistoricalTenure` explorada durante el análisis no fue incluida en el modelo principal debido a que, para clientes que cancelaron, su cálculo depende de la fecha real de cancelación. Utilizarla directamente introduciría **fuga de información temporal**.

---

## 🚀 Futuras mejoras

Entre las posibles extensiones del proyecto se encuentran:

* incorporación de nuevas fuentes de datos;
* construcción de variables relacionadas con comportamiento reciente;
* creación de snapshots históricos de clientes;
* cálculo temporalmente correcto de la antigüedad;
* análisis de calibración de probabilidades;
* optimización del threshold según costos de negocio;
* evaluación de modelos adicionales de gradient boosting;
* monitoreo de drift;
* seguimiento del rendimiento del modelo;
* reentrenamiento periódico;
* experimentos controlados para evaluar estrategias de retención.

Los experimentos futuros deberán realizarse sin reutilizar el conjunto de prueba empleado para la evaluación oficial del proyecto.

---

## 📁 Estructura del repositorio

```text
interconnect-churn-prediction/
│
├── README.md
├── environment.yml
├── .gitignore
│
├── notebooks/
│   ├── 01_comprension_e_integracion.ipynb
│   ├── 02_limpieza_y_eda.ipynb
│   ├── 03_preparacion_y_modelado.ipynb
│   └── 04_estrategia_de_negocio.ipynb
│
├── reports/
│   └── informe_solucion.md
│
└── src/
```

Los datasets utilizados durante el desarrollo permanecen en el entorno local y están excluidos del repositorio mediante `.gitignore`.

---

## 🛠️ Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* Matplotlib
* Seaborn
* JupyterLab
* Git
* GitHub

---

## 🧪 Entorno de ejecución

El proyecto incluye un archivo:

```text
environment.yml
```

con las principales dependencias necesarias para crear un entorno de ejecución compatible.

Ejemplo:

```bash
conda env create -f environment.yml
conda activate interconnect-churn
```

---

## 📄 Informe de solución

El informe ejecutivo y técnico del proyecto se encuentra en:

```text
reports/informe_solucion.md
```

El documento resume:

1. objetivo y problema de negocio;
2. datos y metodología;
3. modelo seleccionado;
4. resultados finales;
5. principales hallazgos;
6. limitaciones;
7. conclusiones;
8. recomendaciones.

---

## 🧪 Experimentos adicionales

Los experimentos desarrollados después de cerrar el resultado oficial se mantendrán separados del pipeline principal.

En particular, futuras pruebas relacionadas con variables de antigüedad como `ApproxTenure` y `HistoricalTenure` tendrán carácter experimental y **no modificarán las métricas oficiales reportadas en este proyecto**.

---

## 👤 Alan Calderón | Data Scientist

Proyecto desarrollado como parte de la formación profesional en Ciencia de Datos de **TripleTen**.
