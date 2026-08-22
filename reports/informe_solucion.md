# Informe de solución — Predicción de churn en Interconnect

## 1. Objetivo y problema de negocio

Interconnect busca reducir la pérdida de clientes mediante la identificación anticipada de usuarios con mayor riesgo de cancelar sus servicios.

El objetivo del proyecto fue desarrollar un modelo de Machine Learning capaz de estimar la probabilidad de cancelación de cada cliente y proporcionar al área de Marketing una herramienta para priorizar acciones preventivas de retención, como promociones, ofertas personalizadas o modificaciones de plan.

El problema se abordó como una tarea de clasificación binaria:

* `Churn = 1`: cliente que canceló el servicio.
* `Churn = 0`: cliente que permanece activo.

La métrica principal empleada para comparar los modelos fue **AUC-ROC**, complementada con **Accuracy**, **Precision**, **Recall** y **F1-score**.

Desde la perspectiva de negocio, el propósito del modelo no es únicamente asignar una etiqueta de `Churn` o `No Churn`, sino generar una probabilidad estimada que pueda utilizarse como un **score de riesgo** para ordenar a los clientes y concentrar los recursos de retención en aquellos con mayor probabilidad de cancelación.

---

## 2. Datos y metodología

El análisis se realizó sobre un conjunto integrado de **7,043 clientes**, con información contractual, demográfica, de facturación y de servicios contratados. La variable objetivo `Churn` identifica si el cliente permanecía activo o había cancelado su relación con Interconnect.
El desarrollo del proyecto siguió un flujo de trabajo compuesto por las siguientes etapas:

1. integración y revisión de las distintas fuentes de datos;
2. limpieza y tratamiento de inconsistencias;
3. análisis exploratorio;
4. ingeniería y selección de características;
5. separación de los conjuntos de entrenamiento y prueba;
6. construcción de pipelines de preprocesamiento;
7. entrenamiento de modelos baseline;
8. comparación mediante validación cruzada;
9. optimización de hiperparámetros;
10. selección del modelo final;
11. evaluación sobre el conjunto de prueba;
12. análisis de importancia de variables e interpretación del modelo.

La división final produjo **5,634 observaciones para entrenamiento y 1,409 para prueba**, manteniendo separado el conjunto de test durante el proceso de selección y optimización.

Se evaluaron distintos algoritmos, entre ellos:

* DummyClassifier;
* Logistic Regression;
* Random Forest;
* Gradient Boosting;
* CatBoost.

El modelo Dummy estableció una referencia de AUC-ROC de `0.50`, mientras que los modelos supervisados mejoraron claramente esa capacidad discriminativa.

Dentro de los modelos con mejor desempeño, Gradient Boosting alcanzó un AUC-ROC aproximado de `0.850466` después de optimización, mientras que CatBoost alcanzó `0.850601`.

También se compararon dos estrategias de CatBoost:

* transformación de variables categóricas mediante One-Hot Encoding;
* tratamiento nativo de variables categóricas.

Después de la optimización, CatBoost con One-Hot Encoding obtuvo un AUC-ROC de `0.850601`, ligeramente superior al `0.850246` obtenido por la versión optimizada con categorías nativas.

---

## 3. Modelo seleccionado

El modelo final seleccionado fue:

**CatBoostClassifier optimizado con One-Hot Encoding.**

La elección se realizó utilizando el desempeño obtenido mediante validación cruzada y antes de consultar el resultado del conjunto de prueba.

El mejor CatBoost alcanzó un **AUC-ROC promedio de validación cruzada de 0.850601**.

La comparación final entre configuraciones confirmó que:

| Modelo                   |   AUC-ROC CV |
| ------------------------ | -----------: |
| CatBoost OHE Tuned       | **0.850601** |
| CatBoost Native Tuned    |     0.850246 |
| CatBoost Native Baseline |     0.847987 |
| CatBoost OHE Baseline    |     0.844751 |

Aunque la diferencia entre los dos CatBoost optimizados es pequeña, el modelo con One-Hot Encoding presentó el mayor AUC-ROC promedio y por ello se mantuvo como modelo oficial del proyecto.

---

## 4. Resultados finales

El modelo final fue evaluado una sola vez sobre el conjunto de prueba y obtuvo:

| Métrica           |    Resultado |
| ----------------- | -----------: |
| AUC-ROC CV        | **0.850601** |
| AUC-ROC Test      | **0.843972** |
| Accuracy Test     | **0.807665** |
| Precision — Churn |     **0.67** |
| Recall — Churn    |     **0.53** |
| F1-score — Churn  |     **0.60** |

Los valores de AUC-ROC y Accuracy corresponden directamente a la evaluación final del modelo seleccionado.

El AUC-ROC disminuyó aproximadamente `0.00663` entre validación cruzada y test:

```text
0.850601 → 0.843972
```

Esta diferencia relativamente pequeña indica que el modelo mantiene un comportamiento razonablemente consistente sobre datos no utilizados durante el desarrollo.

El reporte de clasificación mostró:

| Clase    | Precision | Recall | F1-score | Casos |
| -------- | --------: | -----: | -------: | ----: |
| No Churn |      0.84 |   0.91 |     0.87 | 1,035 |
| Churn    |      0.67 |   0.53 |     0.60 |   374 |

Con el threshold estándar, de los **374 clientes que realmente cancelaron**, el modelo identificó correctamente **200**, mientras que **174** fueron clasificados como `No Churn`.

Esto convierte a los falsos negativos en uno de los principales puntos de atención para una futura utilización comercial.

---

## 5. Principales hallazgos

Los análisis de importancia de variables y SHAP identificaron un conjunto consistente de características con alta relevancia predictiva. Las principales fueron:

* `Type`;
* `TotalCharges`;
* `InternetService`;
* `PaymentMethod`;
* `MonthlyCharges`;
* `TechSupport`;
* `OnlineSecurity`.

Al agrupar la importancia de CatBoost por variable original, las primeras posiciones fueron:

| Variable        | Importancia |
| --------------- | ----------: |
| Type            |       28.12 |
| TotalCharges    |       21.52 |
| InternetService |       14.05 |
| MonthlyCharges  |        6.86 |
| OnlineSecurity  |        4.49 |
| TechSupport     |        4.28 |
| PaymentMethod   |        4.24 |

El **tipo de contrato** fue la característica de mayor relevancia predictiva, seguida por los cargos acumulados y el tipo de servicio de Internet.

Los resultados sugieren prestar especial atención a clientes asociados con:

* contratos de menor duración;
* determinados métodos de pago;
* cargos mensuales elevados;
* determinados tipos de servicio de Internet;
* ausencia de soporte técnico;
* ausencia de seguridad en línea.

Sin embargo, estas variables no deben utilizarse de forma aislada para decidir una intervención. La recomendación es utilizar la **probabilidad predicha de churn como indicador principal de riesgo**, y utilizar las características individuales para interpretar y segmentar ese riesgo.

Asimismo, las asociaciones detectadas por el modelo son predictivas y **no implican causalidad**.

---

## 6. Limitaciones

El modelo presenta varias limitaciones que deben considerarse antes de una implementación productiva.

La primera es su capacidad predictiva. Aunque un AUC-ROC cercano a `0.844` representa una capacidad discriminativa útil, todavía existe margen de mejora.

La segunda limitación está relacionada con la detección de la clase churn. Con el threshold estándar, el recall fue aproximadamente `0.53`, por lo que una proporción considerable de clientes que realmente cancelaron no fue identificada.

Además, el conjunto de datos contiene principalmente información contractual, demográfica y relacionada con servicios. No incluye otras variables que podrían mejorar la predicción, como:

* satisfacción del cliente;
* reclamaciones;
* fallas o interrupciones del servicio;
* calidad de conexión;
* llamadas al centro de atención;
* historial de promociones;
* modificaciones recientes del plan;
* comportamiento de uso;
* interacciones recientes con soporte.

Otra limitación importante corresponde a la antigüedad del cliente. La variable `HistoricalTenure`, calculada durante el análisis, no se utilizó en el modelo principal porque para los clientes que cancelaron depende de la fecha real de cancelación. Incluirla en el modelo oficial introduciría **fuga de información temporal**.

En un escenario productivo, cualquier variable de antigüedad debe calcularse exclusivamente utilizando la información disponible hasta la fecha de predicción.

Finalmente, el modelo identifica asociaciones y patrones predictivos, pero no demuestra que una característica determinada sea la causa de la cancelación.

---

## 7. Conclusiones

El proyecto permitió construir un sistema predictivo capaz de estimar el riesgo de cancelación de clientes de Interconnect.

Después de integrar y preparar los datos, desarrollar características, comparar diferentes algoritmos y optimizar los modelos de mejor desempeño, se seleccionó un **CatBoostClassifier con One-Hot Encoding** como solución final.

El modelo alcanzó un AUC-ROC de aproximadamente `0.844` y una Accuracy cercana a `0.808` en datos no utilizados durante el desarrollo.

Los resultados indican que variables relacionadas con el tipo de contrato, cargos acumulados, servicio de Internet, método de pago, cargos mensuales y determinados servicios adicionales contienen información relevante para diferenciar clientes según su riesgo de cancelación.

Desde una perspectiva empresarial, la principal utilidad del modelo es la generación de un **score de riesgo de churn** que permita ordenar y segmentar a los clientes, facilitando que Marketing priorice acciones de retención sobre los clientes con mayor riesgo estimado.

Sin embargo, la clasificación binaria producida con un threshold de `0.5` no debería considerarse una decisión comercial definitiva. La implementación real debe considerar costos, capacidad operativa y objetivos de negocio antes de determinar el punto de operación del modelo.

---

## 8. Recomendaciones

Se recomienda utilizar el modelo como un **sistema de priorización de clientes**, generando periódicamente la probabilidad de churn para cada cliente activo y ordenando la cartera de acuerdo con ese riesgo.

A partir de estos scores pueden evaluarse estrategias como:

* ofertas para migrar de contratos mensuales hacia contratos de mayor duración;
* revisión de planes para clientes con cargos mensuales elevados;
* promociones o pruebas relacionadas con soporte técnico y seguridad en línea;
* incentivos asociados con determinados métodos de pago;
* campañas diferenciadas para segmentos con scores elevados de churn.

Estas estrategias deben considerarse inicialmente como **hipótesis comerciales**. Para determinar si realmente reducen el churn, deberían validarse mediante experimentos controlados, como pruebas A/B.

También se recomienda analizar el threshold de decisión. El valor `0.5` utilizado durante la evaluación funciona como referencia técnica, pero no necesariamente maximiza el beneficio para la empresa.

La selección del threshold debería considerar:

* costo de una acción de retención;
* valor económico esperado de conservar un cliente;
* costo de adquisición de un nuevo cliente;
* pérdida asociada al churn;
* presupuesto disponible para campañas.

En futuras versiones del sistema sería recomendable incorporar nuevas fuentes de información, construir variables de comportamiento reciente, crear snapshots históricos de clientes, evaluar calibración de probabilidades, monitorear drift y reentrenar periódicamente el modelo.

Para una implementación operativa, el flujo propuesto sería:

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

Las futuras mejoras y experimentos deberán evaluarse utilizando nuevos conjuntos de validación temporal o datos futuros, evitando reutilizar el conjunto de prueba empleado para establecer los resultados oficiales de este proyecto.
