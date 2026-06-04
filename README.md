# Laboratorio1_ia2
Laboratorio1_ia2
# Laboratorio 01 (01/2026) - Clasificacion con Perceptron Multicapa (MLP)

**Estudiante:** Claudia Pereira Cuba  
**Materia:** SIS325  
**Grupo:** G6 - TECHMOCK-AI  
**Archivo del laboratorio:** `lab1_mlp_fashion_mnist_claudia_pereira.ipynb`

---

## Descripcion

Este laboratorio implementa un modelo de clasificacion basado en el Perceptron Multicapa (MLP) usando PyTorch. Se trabaja con el dataset Fashion-MNIST, que contiene 70,000 imagenes de prendas de ropa clasificadas en 10 categorias. El objetivo es mostrar el flujo completo de un proyecto de aprendizaje automatico: desde el analisis exploratorio de los datos hasta la evaluacion final del modelo con graficas, matriz de confusion e inferencia sobre imagenes nuevas.

Se aplican tecnicas de optimizacion, regularizacion y buenas practicas de entrenamiento tal como lo indica el enunciado del laboratorio.

---

## Dataset: Fashion-MNIST

Fashion-MNIST fue creado por Zalando Research como una alternativa mas desafiante al clasico dataset MNIST de digitos. Contiene imagenes en escala de grises de 28x28 pixeles distribuidas en 10 categorias de ropa y accesorios.

| Clase | Categoria   |
|-------|-------------|
| 0     | Camiseta    |
| 1     | Pantalon    |
| 2     | Sueter      |
| 3     | Vestido     |
| 4     | Abrigo      |
| 5     | Sandalia    |
| 6     | Camisa      |
| 7     | Zapatilla   |
| 8     | Bolso       |
| 9     | Botin       |

- Total de imagenes: 70,000
- Entrenamiento: 50,000
- Validacion: 10,000
- Prueba: 10,000
- Caracteristicas por imagen: 784 pixeles (28x28 aplanados)
- Dataset perfectamente balanceado: 6,000 imagenes por clase en entrenamiento

---

## Arquitectura del modelo

```
Entrada:   784 neuronas  (pixeles de la imagen aplanada)
Capa 1:    512 neuronas  + BatchNorm1d + ReLU + Dropout(0.4)
Capa 2:    256 neuronas  + BatchNorm1d + ReLU + Dropout(0.4)
Salida:     10 neuronas  (una por categoria)
```

Total de parametros entrenables: aproximadamente 536,586

---

## Tecnicas aplicadas

**Optimizacion**
- Optimizador Adam con tasa de aprendizaje inicial de 0.001
- Scheduler ReduceLROnPlateau: reduce el learning rate a la mitad si la perdida de validacion no mejora en 7 epocas consecutivas
- Mini-Batch Gradient Descent con batch size de 256

**Regularizacion**
- Dropout con probabilidad 0.4 en cada capa oculta
- Batch Normalization despues de cada capa oculta
- Weight Decay (L2) de 1e-4 en el optimizador Adam

**Buenas practicas**
- Inicializacion de pesos con el metodo de Kaiming (recomendado para ReLU)
- Normalizacion de pixeles al rango [0, 1] dividiendo entre 255
- Guardado del mejor modelo segun la perdida de validacion (Early Stopping manual)
- Separacion estricta entre conjuntos de entrenamiento, validacion y prueba
- Uso de semillas para reproducibilidad de resultados

---

## Contenido del notebook

El notebook esta dividido en 13 secciones:

1. Importacion de librerias y configuracion del dispositivo (CPU/GPU)
2. Carga del dataset Fashion-MNIST
3. Analisis exploratorio: visualizacion de imagenes, distribucion de clases, imagen promedio por categoria, histograma de pixeles
4. Preparacion de los datos: aplanado, normalizacion, split y DataLoaders
5. Definicion de la arquitectura MLP
6. Configuracion del criterio, optimizador y scheduler
7. Ciclo de entrenamiento con registro de historial
8. Curvas de aprendizaje (loss y accuracy por epoca)
9. Evaluacion final sobre el conjunto de prueba con reporte de clasificacion
10. Matriz de confusion: absoluta y normalizada
11. Metricas por clase: precision, recall, F1-score y exactitud por categoria
12. Inferencia sobre imagenes individuales con porcentaje de confianza
13. Analisis de errores: ejemplos de imagenes mal clasificadas y resumen final

---

## Requisitos

```
torch
tensorflow  (solo para cargar el dataset)
scikit-learn
matplotlib
seaborn
numpy
pandas
```

Todas las librerias vienen preinstaladas en Google Colab. El notebook incluye una celda de instalacion con pip al inicio por si se ejecuta en otro entorno.

---

## Como ejecutar

1. Abrir el archivo `lab1_mlp_fashion_mnist_claudia_pereira.ipynb` en Google Colab
2. Ir a **Entorno de ejecucion > Ejecutar todo**
3. El entrenamiento toma aproximadamente 3 a 5 minutos en CPU de Colab gratuito y menos de 1 minuto si hay GPU disponible

---

## Resultados obtenidos

| Metrica              | Valor        |
|----------------------|--------------|
| Exactitud en prueba  | ~88%         |
| Epocas de entreno    | 30           |
| Batch size           | 256          |
| Mejor loss val.      | ~0.32        |
| Parametros totales   | ~536,586     |

Las categorias con mejor desempeno son Pantalon, Sandalia y Bolso. Las categorias con mayor confusion entre si son Camiseta, Camisa y Sueter, lo cual es esperable dado que visualmente son similares.

---

## Estructura del repositorio

```

+-- lab1_mlp_fashion_mnist_claudia_pereira.ipynb   # Notebook principal

```

---



