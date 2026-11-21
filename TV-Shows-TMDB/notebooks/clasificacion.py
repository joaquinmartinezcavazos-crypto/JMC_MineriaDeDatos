#semana 10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

def preparar_datos(df):
    datos = df[["vote_count", "vote_average", "popularity"]].dropna()

    datos = datos[(datos["vote_count"] >= 0) & (datos["vote_average"] >= 0) & (datos["popularity"] >= 0)].copy()

    if len(datos) < 10:
        print("\nNo hay suficientes datos para entrenar el modelo.")
        return None

    percentiles = datos["popularity"].quantile([0.33, 0.66])
    limite_bajo = percentiles.loc[0.33]
    limite_medio = percentiles.loc[0.66]

    datos["nivel_popularidad"] = pd.cut(
        datos["popularity"],
        bins=[-np.inf, limite_bajo, limite_medio, np.inf],
        labels=["Baja", "Media", "Alta"],
        include_lowest=True)

    X = datos[["vote_count", "vote_average"]]
    y = datos["nivel_popularidad"]

    print("\n--- DISTRIBUCION DE CLASES ---")
    print(y.value_counts())

    return X, y

def entrenar_modelo(X, y, k=5):
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    escalador = StandardScaler()

    X_entrenamiento = escalador.fit_transform(X_entrenamiento)
    X_prueba = escalador.transform(X_prueba)

    modelo = KNeighborsClassifier(n_neighbors=k)

    modelo.fit(X_entrenamiento, y_entrenamiento)

    predicciones = modelo.predict(X_prueba)

    return y_prueba, predicciones

def evaluar_modelo(y_prueba, predicciones):
    etiquetas = ["Baja", "Media", "Alta"]
    exactitud = accuracy_score(y_prueba, predicciones)
    matriz = confusion_matrix(y_prueba, predicciones, labels=etiquetas)
    reporte = classification_report(y_prueba, predicciones, labels=etiquetas, target_names=etiquetas, output_dict=True, zero_division=0)

    print("\n--- EVALUACION DEL MODELO KNN ---")
    print(f"Exactitud (Accuracy): {exactitud:.4f}")

    print("\n--- REPORTE DE CLASIFICACION ---")
    print(reporte)

    print("\n--- MATRIZ DE CONFUSION ---")
    print(matriz)

    carpeta = Path(__file__).resolve().parent.parent / "resultados" / "modelos" / "clasificacion"
    carpeta.mkdir(parents=True, exist_ok=True)

    df_reporte = pd.DataFrame(reporte).transpose()
    df_reporte.to_csv(carpeta / "reporte_clasificacion.csv", encoding="utf-8-sig")

    df_matriz = pd.DataFrame(matriz, index=etiquetas, columns=etiquetas)
    df_matriz.to_csv(carpeta / "matriz_confusion.csv", encoding="utf-8-sig")

    print("\nResultados guardados correctamente:")
    print(f"- {carpeta / 'reporte_clasificacion.csv'}")
    print(f"- {carpeta / 'matriz_confusion.csv'}")

def graficar_matriz_confusion(y_prueba, predicciones):
    etiquetas = ["Baja", "Media", "Alta"]

    matriz = confusion_matrix(y_prueba, predicciones, labels=etiquetas)
    visualizacion = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=etiquetas)
    visualizacion.plot(cmap="Blues", values_format="d")

    plt.title("Matriz de confusion - Modelo KNN")
    plt.xlabel("Categoria predicha")
    plt.ylabel("Categoria real")

    carpeta = Path(__file__).resolve().parent.parent / "resultados" / "modelos" / "clasificacion"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / "matriz_confusion.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print(f"\nGrafica guardada correctamente en: {ruta}")

    plt.show()
    plt.close()

def graficar_comparacion(y_prueba, predicciones):
    etiquetas = ["Baja", "Media", "Alta"]

    reales = pd.Series(y_prueba).value_counts().reindex(etiquetas, fill_value=0)
    predichos = pd.Series(predicciones).value_counts().reindex(etiquetas, fill_value=0)

    posiciones = np.arange(len(etiquetas))
    ancho = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(posiciones - ancho / 2, reales, ancho, label="Categorias reales")
    plt.bar(posiciones + ancho / 2, predichos, ancho, label="Categorias predichas")

    plt.title("Comparacion de categorias reales y predichas")
    plt.xlabel("Nivel de popularidad")
    plt.ylabel("Cantidad de series")
    plt.xticks(posiciones, etiquetas)
    plt.legend()
    plt.grid(axis="y")

    carpeta = Path(__file__).resolve().parent.parent / "resultados" / "modelos" / "clasificacion"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / "comparacion_categorias.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print(f"\nGrafica guardada correctamente en: {ruta}")

    plt.show()
    plt.close()

def menu_clasificacion(df):
    datos_preparados = preparar_datos(df)

    if datos_preparados is None:
        return
    X, y = datos_preparados

    while True:
        print("\n======================================")
        print("       CLASIFICACION DE DATOS KNN")
        print("======================================")
        print("1. Entrenar modelo KNN")
        print("2. Evaluar modelo")
        print("3. Graficar matriz de confusion")
        print("4. Graficar comparacion de categorias")
        print("5. Ejecutar analisis completo")
        print("0. Regresar al menu principal")
        print("======================================")

        opcion = input("Selecciona una opcion: ")

        if opcion == "1":
            y_prueba, predicciones = entrenar_modelo(X, y)
            print("\nModelo KNN entrenado correctamente.")

        elif opcion == "2":
            y_prueba, predicciones = entrenar_modelo(X, y)
            evaluar_modelo(y_prueba, predicciones)

        elif opcion == "3":
            y_prueba, predicciones = entrenar_modelo(X, y)
            graficar_matriz_confusion(y_prueba, predicciones)

        elif opcion == "4":
            y_prueba, predicciones = entrenar_modelo(X, y)
            graficar_comparacion(y_prueba, predicciones)

        elif opcion == "5":
            y_prueba, predicciones = entrenar_modelo(X, y)
            evaluar_modelo(y_prueba, predicciones)
            graficar_matriz_confusion(y_prueba, predicciones)
            graficar_comparacion(y_prueba, predicciones)
            print("\nAnalisis de clasificacion finalizado.")

        elif opcion == "0":
            print("\nRegresando al menu principal...")
            break
        else:
            print("\nOpcion no valida. Intenta nuevamente.")

