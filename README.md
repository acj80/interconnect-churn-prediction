# Interconnect — Customer Churn Prediction

Proyecto de Ciencia de Datos orientado a la predicción de cancelación de clientes (*customer churn*) para la empresa de telecomunicaciones **Interconnect**.

El objetivo del proyecto es desarrollar un modelo de Machine Learning capaz de identificar clientes con mayor probabilidad de cancelar su servicio, con el propósito de apoyar estrategias de retención y permitir que la empresa actúe de forma preventiva.

## Objetivo de negocio

Interconnect busca reducir la pérdida de clientes mediante la identificación anticipada de usuarios con riesgo de cancelación.

El proyecto aborda este problema mediante:

* integración de información procedente de diferentes servicios;
* limpieza y transformación de datos;
* análisis exploratorio de datos (EDA);
* creación de la variable objetivo;
* preparación de características;
* entrenamiento y comparación de modelos de clasificación;
* optimización del modelo seleccionado;
* evaluación mediante métricas de clasificación;
* interpretación de resultados;
* generación de recomendaciones orientadas al negocio.

## Datos

El proyecto utiliza información relacionada con:

* contratos de los clientes;
* características personales;
* servicios de Internet;
* servicios telefónicos;
* cargos mensuales y acumulados;
* fechas de inicio y finalización del contrato.

Los datasets originales no se incluyen en este repositorio.

Para ejecutar el proyecto localmente, los archivos deben colocarse dentro de la estructura correspondiente en:

```text
data/
└── final_provider/
```

## Metodología

El proyecto fue desarrollado siguiendo un flujo de trabajo de Ciencia de Datos dividido en cuatro etapas principales.

### 1. Comprensión e integración de datos

Exploración inicial de las distintas fuentes de información, análisis de estructura, identificación de relaciones entre tablas e integración de los datasets.

Notebook:

`notebooks/01_comprension_e_integracion.ipynb`

### 2. Limpieza y análisis exploratorio

Tratamiento de tipos de datos, valores ausentes e inconsistencias, creación de variables necesarias y análisis exploratorio de las principales características relacionadas con la cancelación de clientes.

Notebook:

`notebooks/02_limpieza_y_eda.ipynb`

### 3. Preparación y modelado

Preparación de características, separación de conjuntos de entrenamiento y prueba, entrenamiento de diferentes modelos, comparación mediante validación cruzada y optimización del modelo seleccionado.

Notebook:

`notebooks/03_preparacion_y_modelado.ipynb`

### 4. Estrategia de negocio

Interpretación de los resultados del modelo y análisis de su posible utilización para apoyar estrategias de retención de clientes.

Notebook:

`notebooks/04_estrategia_de_negocio.ipynb`

## Variable objetivo

La variable objetivo representa si un cliente canceló o no su contrato con Interconnect.

El problema se aborda como una tarea de **clasificación binaria**.

## Modelos evaluados

Durante el proyecto se probaron y compararon diferentes algoritmos de clasificación.

El proceso incluyó:

* modelos de referencia;
* modelos basados en árboles;
* técnicas de boosting;
* optimización de hiperparámetros;
* validación cruzada.

## Modelo seleccionado

El modelo final seleccionado fue:

**CatBoostClassifier**

La selección se realizó considerando principalmente su desempeño en **AUC-ROC**, complementado con otras métricas de clasificación.

## Resultados finales

Los resultados oficiales del modelo final se incorporarán aquí a partir del informe final del proyecto.

| Métrica  |               Resultado |
| -------- | ----------------------: |
| AUC-ROC  | Pendiente de incorporar |
| Accuracy | Pendiente de incorporar |

## Principales hallazgos

El análisis permitió identificar patrones asociados con la probabilidad de cancelación de los clientes.

Los hallazgos definitivos y su interpretación de negocio se documentarán en el informe de solución del proyecto.

## Estructura del repositorio

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
│
└── src/
```

La carpeta `data/` se mantiene únicamente en el entorno local y está excluida del repositorio mediante `.gitignore`.

## Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* CatBoost
* JupyterLab
* Git
* GitHub

## Entorno de ejecución

El repositorio incluye el archivo:

```text
environment.yml
```

que contiene las principales dependencias necesarias para reproducir el entorno de trabajo.

## Informe de solución

El informe ejecutivo del proyecto se incorporará en:

```text
reports/informe_solucion.md
```

El documento incluirá:

1. Objetivo y problema de negocio
2. Datos y metodología
3. Modelo seleccionado
4. Resultados finales
5. Principales hallazgos
6. Limitaciones
7. Conclusiones
8. Recomendaciones

## Experimentos adicionales

Los experimentos posteriores al resultado oficial del proyecto se documentarán por separado para evitar modificar o confundir las métricas finales previamente validadas.

Estos análisis tendrán carácter exploratorio y no sustituirán el modelo oficial presentado en el proyecto.

## Autor

Proyecto desarrollado como parte de la formación profesional en Ciencia de Datos de TripleTen.
