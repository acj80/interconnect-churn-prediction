# Interconnect — Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10.21-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7.2-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

Proyecto de Ciencia de Datos orientado a la predicción de cancelación de clientes (*customer churn*) para la empresa de telecomunicaciones **Interconnect**.

El proyecto desarrolla un modelo de Machine Learning capaz de estimar la probabilidad de cancelación de cada cliente con el objetivo de apoyar al área de Marketing en la priorización de campañas y estrategias preventivas de retención.

---

## ⭐ Project Highlights

- **7,043 clientes** analizados.
- Modelo final: **CatBoostClassifier + One-Hot Encoding**.
- **AUC-ROC CV:** 0.8506.
- **AUC-ROC Test:** 0.8440.
- **Accuracy Test:** 0.8077.
- Interpretabilidad mediante **feature importance y SHAP**.
- Enfoque de negocio basado en **probabilidad de churn como score de riesgo**.
- Evaluación explícita del **threshold de decisión**.
- Pipeline reproducible con **Conda y pip**.

## 🎯 Objetivo de negocio

Interconnect busca reducir la pérdida de clientes mediante la identificación anticipada de usuarios con mayor riesgo de cancelar sus servicios.

La cancelación de clientes representa un problema relevante para una empresa de telecomunicaciones, ya que puede afectar los ingresos recurrentes e incrementar la necesidad de adquirir nuevos clientes.

El objetivo del proyecto es utilizar información contractual, demográfica, de servicios y facturación para desarrollar un modelo que permita:

- identificar clientes con mayor probabilidad de churn;
- generar un score de riesgo para cada cliente;
- priorizar acciones de retención;
- apoyar la segmentación de clientes;
- utilizar de forma más eficiente los recursos destinados a campañas comerciales.

El problema se aborda como una tarea de **clasificación binaria**:

- `Churn = 1`: el cliente canceló el servicio.
- `Churn = 0`: el cliente permanece activo.

La métrica principal utilizada durante el desarrollo es **AUC-ROC**, complementada con:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- matriz de confusión.

---

## 📊 Datos

El proyecto integra diferentes fuentes de información relacionadas con los clientes de Interconnect:

- información contractual;
- características personales;
- servicios de Internet;
- servicios telefónicos;
- cargos mensuales;
- cargos acumulados;
- métodos de pago;
- servicios adicionales contratados.

El dataset integrado contiene información correspondiente a **7,043 clientes**.

Los datasets originales no se incluyen en este repositorio.

Para ejecutar los notebooks localmente, los archivos originales deben colocarse dentro de:

```text
data/
└── final_provider/
```

Durante el procesamiento se generan archivos intermedios:

```text
data/
├── final_provider/          # datasets originales
├── interconnect_raw.csv     # dataset integrado
└── interconnect_clean.csv   # dataset limpio utilizado para modelado
```

La carpeta completa `data/` está excluida del control de versiones mediante `.gitignore`.

---

## 🔎 Metodología

El proyecto se desarrolló siguiendo un flujo completo de Ciencia de Datos, desde la integración de las fuentes originales hasta la construcción, selección, evaluación e interpretación del modelo final.

### 1. Comprensión e integración de datos

Se analizaron las diferentes fuentes de información, sus estructuras, tipos de variables y relaciones entre tablas antes de realizar su integración.

Las principales tareas incluyeron:

- revisión inicial de las fuentes;
- análisis de llaves de identificación;
- verificación de dimensiones;
- revisión de tipos de datos;
- identificación de relaciones entre tablas;
- integración de las fuentes disponibles;
- construcción de un dataset maestro.

Notebook:

[`01_comprension_e_integracion.ipynb`](notebooks/01_comprension_e_integracion.ipynb)

---

### 2. Limpieza y análisis exploratorio

Se realizaron tareas de:

- revisión y conversión de tipos de datos;
- tratamiento de valores ausentes;
- identificación de inconsistencias;
- construcción de la variable objetivo;
- análisis de la distribución de churn;
- análisis exploratorio de características numéricas;
- análisis exploratorio de características categóricas;
- evaluación de patrones asociados con la cancelación;
- preparación de un dataset limpio para modelado.

Notebook:

[`02_limpieza_y_eda.ipynb`](notebooks/02_limpieza_y_eda.ipynb)

---

### 3. Preparación y modelado

La etapa de Machine Learning incluyó:

- selección de variables predictoras;
- separación de entrenamiento y prueba;
- transformación de variables numéricas y categóricas;
- construcción de pipelines;
- validación cruzada estratificada de cinco folds;
- establecimiento de modelos baseline;
- comparación de distintos algoritmos;
- optimización de hiperparámetros;
- evaluación de CatBoost con One-Hot Encoding;
- evaluación de CatBoost con manejo nativo de categorías;
- selección del modelo final;
- evaluación sobre el conjunto de prueba;
- análisis de importancia de variables;
- interpretación mediante valores SHAP.

Notebook:

[`03_preparacion_y_modelado.ipynb`](notebooks/03_preparacion_y_modelado.ipynb)

---

### 4. Informe de solución y estrategia de negocio

Los resultados del modelado se consolidaron en un informe final que integra:

- evaluación del modelo seleccionado;
- interpretación de resultados;
- principales variables predictivas;
- limitaciones;
- aplicación operacional;
- selección del threshold;
- recomendaciones de negocio;
- posibles mejoras futuras.

Notebook:

[`04_informe_solucion.ipynb`](notebooks/04_informe_solucion.ipynb)

Informe:

[`reports/informe_solucion.md`](reports/informe_solucion.md)

---

## 🔀 División de los datos

Para mantener una evaluación metodológicamente correcta, el conjunto de prueba permaneció separado durante el desarrollo del modelo.

La división utilizada fue:

```text
Dataset total:     7,043 clientes
Training set:      5,634 clientes
Test set:          1,409 clientes
```

La división se realizó utilizando estratificación para conservar aproximadamente la misma proporción de clientes churn y no churn en ambos conjuntos.

El conjunto de prueba permaneció aislado durante:

- análisis para selección del modelo;
- selección de variables;
- comparación de algoritmos;
- optimización de hiperparámetros;
- selección de la configuración final.

Únicamente después de cerrar estas decisiones se utilizó el conjunto de prueba para estimar la capacidad de generalización del modelo.

---

## 📏 Estrategia de evaluación

La métrica principal utilizada fue:

**AUC-ROC**

Esta métrica permite evaluar la capacidad del modelo para distinguir entre clientes que cancelan y clientes que permanecen utilizando las probabilidades estimadas por el modelo.

Como métricas complementarias se utilizaron:

- Accuracy;
- Precision;
- Recall;
- F1-score;
- matriz de confusión.

La validación de modelos se realizó mediante:

**Stratified 5-Fold Cross Validation**

Esto permitió comparar los algoritmos bajo una metodología consistente y redujo la dependencia de una única división de los datos.

---

## 🤖 Modelos evaluados

Durante el proyecto se compararon diferentes algoritmos utilizando el mismo esquema de validación cruzada y las mismas métricas.

Los principales modelos evaluados fueron:

- DummyClassifier;
- Logistic Regression;
- Random Forest;
- Gradient Boosting;
- CatBoost.

---

## 📉 Baseline

Se utilizó `DummyClassifier` como referencia mínima para verificar que los modelos desarrollados realmente aprendieran patrones relevantes.

El modelo Dummy obtuvo aproximadamente:

| Métrica | Resultado |
| --- | ---: |
| AUC-ROC | **0.5000** |
| Accuracy | **0.7346** |

El AUC-ROC de `0.50` representa un comportamiento equivalente al azar en términos de capacidad discriminativa.

---

## 📈 Regresión Logística

La Regresión Logística se utilizó como baseline interpretable.

Resultados aproximados mediante validación cruzada:

| Métrica | Resultado CV |
| --- | ---: |
| AUC-ROC | **0.8399** |
| Accuracy | **0.8007** |

La mejora frente al Dummy confirmó que las variables utilizadas contienen información predictiva relevante relacionada con el churn.

---

## 🌲 Comparación inicial de modelos

Durante la primera comparación, los principales resultados de AUC-ROC fueron aproximadamente:

| Modelo | AUC-ROC CV |
| --- | ---: |
| Gradient Boosting | **0.8477** |
| CatBoost | **0.8448** |
| Logistic Regression | **0.8399** |
| Random Forest | **0.8182** |
| DummyClassifier | **0.5000** |

Gradient Boosting y CatBoost mostraron los mejores resultados entre los modelos no lineales.

Por esta razón fueron seleccionados como principales candidatos para optimización de hiperparámetros.

Logistic Regression se conservó como baseline interpretable.

---

## ⚙️ Optimización de modelos

Los modelos con mejor desempeño fueron optimizados utilizando búsqueda de hiperparámetros y validación cruzada estratificada.

### Gradient Boosting

Después del tuning:

```text
AUC-ROC CV ≈ 0.8505
```

### CatBoost

Después del tuning:

```text
AUC-ROC CV ≈ 0.8506
```

La diferencia entre ambos modelos fue pequeña, por lo que se realizaron experimentos adicionales sobre la forma de manejar las variables categóricas en CatBoost.

---

## 🧪 CatBoost: One-Hot Encoding vs categorías nativas

Se compararon dos estrategias para utilizar CatBoost:

1. variables categóricas transformadas mediante **One-Hot Encoding**;
2. variables categóricas tratadas de forma **nativa por CatBoost**.

Los resultados obtenidos fueron:

| Estrategia | AUC-ROC CV |
| --- | ---: |
| CatBoost OHE Tuned | **0.850601** |
| CatBoost Native Tuned | **0.850246** |
| CatBoost Native Baseline | **0.847987** |
| CatBoost OHE Baseline | **0.844751** |

En los modelos baseline, el manejo nativo de categorías mejoró el desempeño de CatBoost.

Sin embargo, después de la optimización ambas estrategias presentaron resultados muy similares.

La configuración:

**CatBoost + One-Hot Encoding**

obtuvo el mayor AUC-ROC promedio durante validación cruzada y fue seleccionada **antes de evaluar el conjunto de prueba**.

---

## 🏆 Modelo final

El modelo seleccionado fue:

### CatBoostClassifier + One-Hot Encoding

La selección se realizó utilizando exclusivamente los resultados obtenidos sobre el conjunto de entrenamiento mediante validación cruzada.

El conjunto de prueba no participó en la decisión del modelo final.

---

## 📈 Resultados finales

El modelo final obtuvo los siguientes resultados:

| Métrica | Resultado |
| --- | ---: |
| AUC-ROC — Validación cruzada | **0.8506** |
| AUC-ROC — Test | **0.8440** |
| Accuracy — Test | **0.8077** |
| Precision — Churn | **0.67** |
| Recall — Churn | **0.53** |
| F1-score — Churn | **0.60** |

Los valores calculados directamente durante la evaluación fueron:

```text
CV AUC-ROC:     0.850601
Test AUC-ROC:   0.843972
Test Accuracy:  0.807665
```

---

## 📊 Generalización

La diferencia entre el AUC-ROC obtenido mediante validación cruzada y el conjunto de prueba fue aproximadamente:

```text
Test AUC - CV AUC ≈ -0.00663
```

Esta reducción relativamente pequeña sugiere que el modelo mantiene un comportamiento razonablemente consistente sobre datos no utilizados durante su desarrollo.

No se observa una caída pronunciada entre validación cruzada y test.

---

## 🎯 Rendimiento sobre la clase Churn

El reporte de clasificación sobre el conjunto de prueba fue aproximadamente:

| Clase | Precision | Recall | F1-score | Casos |
| --- | ---: | ---: | ---: | ---: |
| No Churn | 0.84 | 0.91 | 0.87 | 1035 |
| Churn | 0.67 | 0.53 | 0.60 | 374 |

Accuracy global:

```text
0.8077
```

---

## 🧮 Matriz de confusión

Utilizando el threshold estándar de `0.5`, los resultados fueron:

```text
True Negatives:   938
False Positives:   97
False Negatives:  174
True Positives:   200
```

En el conjunto de prueba existían:

```text
374 clientes que realmente cancelaron
```

El modelo identificó correctamente:

```text
200 clientes churn
```

mientras que:

```text
174 clientes churn
```

fueron clasificados como clientes que permanecerían.

Esto se refleja en un Recall para la clase Churn cercano a:

```text
0.53
```

Aunque el modelo presenta una capacidad discriminativa útil, una implementación comercial debería analizar cuidadosamente el umbral de decisión.

El threshold de `0.5` no necesariamente representa la opción económicamente óptima para una estrategia de retención.

---

## 🔍 Interpretabilidad del modelo

Para comprender el comportamiento del modelo se utilizaron dos enfoques:

1. importancia interna de características de CatBoost;
2. valores SHAP.

Debido a que el pipeline final utiliza One-Hot Encoding, las variables categóricas se transforman en múltiples columnas.

Para facilitar la interpretación desde una perspectiva de negocio, las importancias fueron agrupadas nuevamente a nivel de variable original.

---

## 📊 Principales variables predictivas

Las variables con mayor relevancia predictiva fueron:

1. `Type`
2. `TotalCharges`
3. `InternetService`
4. `MonthlyCharges`
5. `OnlineSecurity`
6. `TechSupport`
7. `PaymentMethod`

La importancia agrupada obtenida mediante CatBoost fue aproximadamente:

| Variable | Importancia |
| --- | ---: |
| Type | **28.12** |
| TotalCharges | **21.52** |
| InternetService | **14.05** |
| MonthlyCharges | **6.86** |
| OnlineSecurity | **4.49** |
| TechSupport | **4.28** |
| PaymentMethod | **4.24** |

El **tipo de contrato** presentó la mayor relevancia predictiva, seguido de los cargos acumulados y el tipo de servicio de Internet.

Estos resultados son consistentes con diferentes patrones identificados durante el análisis exploratorio.

---

## ⚠️ Importancia predictiva ≠ causalidad

Las importancias representan asociaciones predictivas dentro del modelo.

No deben interpretarse automáticamente como relaciones causales.

Por ejemplo:

si `PaymentMethod` aparece como una variable importante, no puede concluirse que cambiar el método de pago provoque directamente una reducción del churn.

Únicamente puede afirmarse que el modelo encontró esa variable útil para distinguir clientes con diferentes patrones de cancelación.

Por lo tanto:

```text
Importancia predictiva ≠ causalidad
```

Las estrategias comerciales derivadas del análisis deben validarse mediante experimentos controlados.

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

En lugar de aplicar una campaña general a toda la base de clientes, la empresa podría utilizar este score para ordenar a los clientes según su riesgo estimado.

---

## 🔄 Flujo operativo propuesto

Una posible integración del modelo dentro del proceso comercial sería:

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
       ↓
Retroalimentación
```

De esta manera, Marketing podría concentrar recursos en los clientes con mayor riesgo estimado.

---

## 🎯 Segmentación de clientes

El score de churn podría utilizarse para construir segmentos como:

```text
Riesgo alto
Riesgo medio
Riesgo bajo
```

La definición exacta de los segmentos debería depender de:

- presupuesto disponible;
- tamaño de la campaña;
- costo de la acción de retención;
- valor esperado de cada cliente;
- capacidad operativa del equipo comercial.

La probabilidad estimada por el modelo debería utilizarse como indicador principal de riesgo.

Las variables predictivas deberían emplearse como información complementaria para interpretar y segmentar los perfiles.

---

## 💡 Recomendaciones de negocio

Los resultados sugieren analizar especialmente clientes asociados con características como:

- contratos de menor duración;
- contratos mes a mes;
- determinados métodos de pago;
- cargos mensuales elevados;
- determinados tipos de servicio de Internet;
- ausencia de soporte técnico;
- ausencia de seguridad en línea.

Entre las posibles estrategias a evaluar se encuentran:

- incentivos para migrar de contratos mensuales hacia contratos de mayor duración;
- revisión de planes para clientes con cargos mensuales elevados;
- promociones relacionadas con soporte técnico;
- promociones relacionadas con seguridad en línea;
- incentivos para determinados métodos de pago;
- campañas específicas para clientes con scores elevados de churn;
- estrategias diferenciadas por nivel de riesgo.

Estas acciones deben considerarse **hipótesis comerciales**, no consecuencias causales demostradas por el modelo.

Su efectividad debería evaluarse mediante experimentos controlados, como pruebas A/B.

---

## ⚖️ Selección del threshold

El umbral utilizado durante la evaluación oficial fue:

```text
0.5
```

Este valor funciona como una referencia estándar para convertir probabilidades en clases.

Sin embargo, desde una perspectiva comercial, el threshold óptimo debería determinarse utilizando información económica.

Entre los factores relevantes se encuentran:

- costo de una acción de retención;
- valor esperado de conservar un cliente;
- costo de adquisición de nuevos clientes;
- pérdida económica asociada al churn;
- presupuesto disponible para campañas;
- cantidad máxima de clientes que pueden ser contactados.

Reducir el threshold puede incrementar el Recall y detectar más clientes que posteriormente cancelarían.

Sin embargo, también incrementaría los falsos positivos y el costo de las campañas.

Por esta razón, una futura implementación debería optimizar el punto de operación del modelo utilizando criterios de negocio y no exclusivamente métricas estadísticas.

---

## ⚠️ Limitaciones

El proyecto presenta varias limitaciones.

### Rendimiento predictivo

El modelo final obtiene un AUC-ROC cercano a:

```text
0.844
```

por lo que todavía existe margen para mejorar su capacidad predictiva.

### Recall

Con threshold `0.5`, el Recall de churn es aproximadamente:

```text
0.53
```

Esto significa que una proporción relevante de las cancelaciones reales no es identificada.

### Información disponible

El dataset está compuesto principalmente por información:

- contractual;
- demográfica;
- servicios contratados;
- facturación;
- métodos de pago.

No dispone de algunas variables que podrían aportar información adicional, como:

- satisfacción del cliente;
- número de reclamaciones;
- interrupciones del servicio;
- calidad de conexión;
- llamadas al centro de atención;
- historial de promociones;
- modificaciones recientes del plan;
- comportamiento de uso;
- consumo reciente;
- interacciones recientes con soporte.

---

## ⏳ Limitación temporal: HistoricalTenure

Durante el análisis se exploró una variable denominada:

```text
HistoricalTenure
```

Esta variable representa la duración histórica del contrato.

Sin embargo, no fue incluida en el modelo principal.

Para clientes que cancelaron, su cálculo depende de la fecha real de cancelación.

En un escenario de predicción real, esa fecha todavía no sería conocida.

Utilizarla directamente como predictor introduciría:

**fuga de información temporal (*temporal data leakage*)**.

Por esta razón, `HistoricalTenure` fue excluida del pipeline oficial.

En producción, una variable de antigüedad debería calcularse utilizando:

```text
fecha de predicción - BeginDate
```

y nunca una fecha futura de cancelación.

---

## 🚀 Futuras mejoras

Entre las posibles extensiones del proyecto se encuentran:

- incorporación de nuevas fuentes de datos;
- construcción de variables relacionadas con comportamiento reciente;
- creación de snapshots históricos de clientes;
- cálculo temporalmente correcto de la antigüedad;
- análisis de calibración de probabilidades;
- optimización del threshold según costos de negocio;
- evaluación de modelos adicionales de gradient boosting;
- nuevas estrategias de feature engineering;
- monitoreo de drift;
- seguimiento del rendimiento del modelo;
- reentrenamiento periódico;
- experimentos controlados para evaluar estrategias de retención.

---

## 🕒 Snapshots temporales

Una mejora especialmente importante sería construir snapshots históricos.

Por ejemplo:

```text
Cliente A — Enero
Cliente A — Febrero
Cliente A — Marzo
Cliente A — Abril
```

Cada observación representaría únicamente la información conocida hasta ese momento.

Este enfoque permitiría:

- calcular correctamente la antigüedad;
- crear variables de comportamiento reciente;
- evitar leakage temporal;
- construir escenarios más similares a una implementación real.

---

## 🧪 Validación futura

El conjunto de prueba utilizado para la evaluación oficial no debería reutilizarse para seleccionar nuevas mejoras.

Los futuros experimentos deberían evaluarse utilizando:

- nuevos conjuntos temporales;
- nuevas particiones de validación;
- datos futuros;
- esquemas de backtesting temporal.

De esta manera se preserva la independencia del conjunto de prueba oficial.

---

## 🧪 Experimentos adicionales

Los experimentos desarrollados después de cerrar el resultado oficial se mantendrán separados del pipeline principal.

En particular, futuras pruebas relacionadas con variables de antigüedad como:

```text
ApproxTenure
HistoricalTenure
```

tendrán carácter experimental.

Estos experimentos:

- no modificarán las métricas oficiales;
- no reemplazarán el modelo validado;
- se documentarán de forma separada;
- utilizarán nuevas estrategias de validación cuando corresponda.

El resultado oficial del proyecto permanece:

```text
CatBoostClassifier + One-Hot Encoding

CV AUC-ROC:    0.850601
Test AUC-ROC:  0.843972
Test Accuracy: 0.807665
```

---

## 📁 Estructura del repositorio

```text
interconnect-churn-prediction/
│
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 01_comprension_e_integracion.ipynb
│   ├── 02_limpieza_y_eda.ipynb
│   ├── 03_preparacion_y_modelado.ipynb
│   └── 04_informe_solucion.ipynb
│
└── reports/
    └── informe_solucion.md
```

Los datasets utilizados durante el desarrollo permanecen en el entorno local y están excluidos del repositorio mediante `.gitignore`.

---

## 🛠️ Tecnologías utilizadas

El proyecto utiliza principalmente:

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- Matplotlib
- Seaborn
- JupyterLab
- Git
- GitHub

---

## 📦 Versiones principales

El entorno utilizado para reproducir el proyecto contiene las siguientes versiones principales:

| Tecnología | Versión |
| --- | ---: |
| Python | **3.10.21** |
| NumPy | **2.2.6** |
| Pandas | **2.3.3** |
| Scikit-learn | **1.7.2** |
| Matplotlib | **3.10.9** |
| Seaborn | **0.13.2** |
| CatBoost | **1.2.10** |
| JupyterLab | **4.6.3** |

---

## 🧪 Instalación y entorno de ejecución

El proyecto puede reproducirse utilizando **Conda** o un entorno virtual de Python con **pip**.

### Opción 1 — Conda

Clonar el repositorio:

```bash
git clone https://github.com/acj80/interconnect-churn-prediction.git
```

Entrar al proyecto:

```bash
cd interconnect-churn-prediction
```

Crear el entorno:

```bash
conda env create -f environment.yml
```

Activarlo:

```bash
conda activate interconnect-churn
```

El entorno fue validado con:

```text
Python 3.10.21
```

---

### Opción 2 — pip

Clonar el repositorio:

```bash
git clone https://github.com/acj80/interconnect-churn-prediction.git
```

Entrar al proyecto:

```bash
cd interconnect-churn-prediction
```

Crear un entorno virtual:

```bash
python -m venv .venv
```

Activarlo en macOS/Linux:

```bash
source .venv/bin/activate
```

Activarlo en Windows:

```bash
.venv\Scripts\activate
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Las dependencias principales definidas para el proyecto son:

```text
numpy==2.2.6
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.9
seaborn==0.13.2
catboost==1.2.10
jupyterlab==4.6.3
```

---

## ▶️ Ejecución del proyecto

Antes de ejecutar los notebooks, los datasets originales deben colocarse dentro de:

```text
data/final_provider/
```

Los notebooks deben ejecutarse en el siguiente orden:

1. [`01_comprension_e_integracion.ipynb`](notebooks/01_comprension_e_integracion.ipynb)
2. [`02_limpieza_y_eda.ipynb`](notebooks/02_limpieza_y_eda.ipynb)
3. [`03_preparacion_y_modelado.ipynb`](notebooks/03_preparacion_y_modelado.ipynb)
4. [`04_informe_solucion.ipynb`](notebooks/04_informe_solucion.ipynb)

El flujo general es:

```text
Datos originales
       ↓
01 — Comprensión e integración
       ↓
interconnect_raw.csv
       ↓
02 — Limpieza y EDA
       ↓
interconnect_clean.csv
       ↓
03 — Preparación y modelado
       ↓
Modelo final + evaluación
       ↓
04 — Informe de solución
       ↓
Interpretación y estrategia de negocio
```

---

## 📄 Informe de solución

El informe ejecutivo y técnico del proyecto se encuentra en:

[`reports/informe_solucion.md`](reports/informe_solucion.md)

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

## 📌 Resultado principal

El proyecto demuestra que la información contractual, de servicios y facturación permite construir un modelo con capacidad útil para priorizar clientes según su riesgo de cancelación.

El modelo final:

```text
CatBoostClassifier + One-Hot Encoding
```

alcanzó:

```text
AUC-ROC CV:     0.850601
AUC-ROC Test:   0.843972
Accuracy Test:  0.807665
```

El valor principal para una futura implementación comercial no sería únicamente la clasificación binaria, sino la **probabilidad estimada de churn**, utilizada como score para priorizar acciones de retención.

La selección final del threshold debería realizarse utilizando costos y beneficios reales del negocio.

---

## 👤 Autor

**Alan Calderón**

Data Scientist

Proyecto desarrollado como parte de la formación profesional en Ciencia de Datos de **TripleTen**.

---

## 📚 Proyecto

**Customer Churn Prediction — Interconnect**

Machine Learning · Classification · Customer Churn · CatBoost · Scikit-learn · SHAP · Data Science
