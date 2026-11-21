#semana 9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def preparar_datos(df):
    datos = df[["vote_count", "popularity"]].dropna()
    datos = datos[(datos["vote_count"] >= 0) & (datos["popularity"] >= 0)]

    if (len(datos) < 2 or datos["vote_count"].nunique() < 2 or datos["popularity"].nunique() < 2):
        print("\nNo hay suficientes datos diferentes para generar el modelo.")
        return None

    X = datos[["vote_count"]]
    y = datos["popularity"]

    return X, y

def calcular_correlacion(X, y):
    correlacion, valor_p = stats.pearsonr(X["vote_count"], y)

    print("\n--- CORRELACION DE PEARSON ---")
    print(f"Coeficiente de correlacion: {correlacion:.4f}")
    print(f"Valor p: {valor_p:.6f}")

    if valor_p < 0.05:
        print("Existe una correlacion estadisticamente significativa.")
    else:
        print("No existe una correlacion estadisticamente significativa.")

    if correlacion > 0:
        print("La relacion entre las variables es positiva.")
    elif correlacion < 0:
        print("La relacion entre las variables es negativa.")
    else:
        print("No existe una relacion lineal.")

    return correlacion, valor_p

def generar_modelo(X, y):
    modelo = LinearRegression()
    modelo.fit(X, y)

    predicciones = modelo.predict(X)

    r2 = r2_score(y, predicciones)
    error = np.sqrt(mean_squared_error(y, predicciones))

    print("\n--- MODELO DE REGRESION LINEAL ---")
    print(f"Intercepto: {modelo.intercept_:.4f}")
    print(f"Pendiente: {modelo.coef_[0]:.4f}")
    print(f"Coeficiente R2: {r2:.4f}")
    print(f"Error RMSE: {error:.4f}")

    return modelo, predicciones

def graficar_regresion(X, y, modelo):
    datos_ordenados = X.sort_values("vote_count")
    predicciones_ordenadas = modelo.predict(datos_ordenados)

    plt.figure(figsize=(10, 6))
    plt.scatter(X["vote_count"], y, alpha=0.5, label="Datos reales")
    plt.plot(datos_ordenados["vote_count"], predicciones_ordenadas, label="Regresion lineal")

    plt.title("Regresion lineal: Votos vs Popularidad")
    plt.xlabel("Cantidad de votos")
    plt.ylabel("Popularidad")
    plt.legend()
    plt.grid(True)

    carpeta = Path(__file__).resolve().parent.parent / "resultados" / "modelos" / "modelos_lineales"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / "regresion_lineal.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print(f"\nGrafica guardada correctamente en: {ruta}")
    
    plt.show()
    plt.close()

def graficar_residuos(X, y, predicciones):
    residuos = y - predicciones

    plt.figure(figsize=(10, 6))
    plt.scatter(predicciones, residuos, alpha=0.5)
    plt.axhline(y=0, linestyle="--")

    plt.title("Grafica de residuos")
    plt.xlabel("Valores predichos")
    plt.ylabel("Residuos")
    plt.grid(True)

    carpeta = Path(__file__).resolve().parent.parent / "resultados" / "modelos" / "modelos_lineales"
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta = carpeta / "residuos.png"
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    print(f"\nGrafica guardada correctamente en: {ruta}")

    plt.show()
    plt.close()

def menu_modelos_lineales(df):

    datos_preparados = preparar_datos(df)

    if datos_preparados is None:
        return

    X, y = datos_preparados

    while True:
        print("\n======================================")
        print("       MODELOS LINEALES")
        print("======================================")
        print("1. Calcular correlacion de Pearson")
        print("2. Generar modelo de regresion lineal")
        print("3. Graficar regresion lineal")
        print("4. Graficar residuos")
        print("5. Ejecutar analisis completo")
        print("0. Regresar al menu principal")
        print("======================================")

        opcion = input("Selecciona una opcion: ")

        if opcion == "1":
            calcular_correlacion(X, y)

        elif opcion == "2":
            generar_modelo(X, y)

        elif opcion == "3":
            modelo, predicciones = generar_modelo(X, y)
            graficar_regresion(X, y, modelo)

        elif opcion == "4":
            modelo, predicciones = generar_modelo(X, y)
            graficar_residuos(X, y, predicciones)

        elif opcion == "5":
            calcular_correlacion(X, y)
            modelo, predicciones = generar_modelo(X, y)

            graficar_regresion(X, y, modelo)
            graficar_residuos(X, y, predicciones)
            print("\nAnalisis completo finalizado.")

        elif opcion == "0":
            print("\nRegresando al menu principal...")
            break

        else:
            print("\nOpcion no valida. Intenta nuevamente.")