#semana 13
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def obtener_carpeta_resultados():
    base_dir = Path(__file__).resolve().parent.parent
    carpeta = base_dir / "resultados" / "predicciones" / "pronostico"
    carpeta.mkdir(parents=True, exist_ok=True)

    return carpeta

def preparar_datos(df):
    datos = df[["first_air_date", "popularity"]].copy()
    datos["first_air_date"] = pd.to_datetime(datos["first_air_date"], errors="coerce")

    datos = datos.dropna(subset=["first_air_date", "popularity"])
    datos = datos[datos["popularity"] >= 0].copy()
    datos["year"] = datos["first_air_date"].dt.year

    if len(datos) < 10:
        print("\nNo hay suficientes datos para realizar el pronostico.")
        return None

    return datos


def agrupar_por_anio(datos):
    historico = datos.groupby("year")["popularity"].mean().reset_index()
    historico = historico.sort_values("year").reset_index(drop=True)
    historico.rename(columns={"popularity": "popularidad_promedio"}, inplace=True)
    return historico

def dividir_datos(historico):
    if len(historico) < 5:
        print("\nNo hay suficientes años para dividir los datos.")
        return None

    cantidad_prueba = max(1, int(len(historico) * 0.2))
    entrenamiento = historico.iloc[:-cantidad_prueba].copy()
    prueba = historico.iloc[-cantidad_prueba:].copy()

    X_entrenamiento = entrenamiento[["year"]]
    y_entrenamiento = entrenamiento["popularidad_promedio"]
    X_prueba = prueba[["year"]]
    y_prueba = prueba["popularidad_promedio"]

    print("\n===== DIVISION DE DATOS =====")
    print(f"Anios de entrenamiento: {len(entrenamiento)}")
    print(f"Anios de prueba: {len(prueba)}")
    print(f"Periodo de entrenamiento: {entrenamiento['year'].min()} - {entrenamiento['year'].max()}")
    print(f"Periodo de prueba: {prueba['year'].min()} - {prueba['year'].max()}")

    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba

def entrenar_modelo(X_entrenamiento, y_entrenamiento):
    modelo = LinearRegression()
    modelo.fit(X_entrenamiento, y_entrenamiento)

    print("\n===== ENTRENAMIENTO DEL MODELO =====")
    print("Modelo de regresion lineal entrenado correctamente.")
    print(f"Pendiente: {modelo.coef_[0]:.4f}")
    print(f"Intercepto: {modelo.intercept_:.4f}")

    return modelo

def evaluar_modelo(modelo, X_prueba, y_prueba):
    predicciones = modelo.predict(X_prueba)
    mae = mean_absolute_error(y_prueba, predicciones)
    rmse = np.sqrt(mean_squared_error(y_prueba, predicciones))

    if len(y_prueba) >= 2:
        r2 = r2_score(y_prueba, predicciones)
    else:
        r2 = np.nan

    print("\n===== EVALUACION DEL MODELO =====")
    print(f"Error absoluto medio (MAE): {mae:.4f}")
    print(f"Raiz del error cuadratico medio (RMSE): {rmse:.4f}")

    if np.isnan(r2):
        print("Coeficiente de determinacion (R2): No disponible")
    else:
        print(f"Coeficiente de determinacion (R2): {r2:.4f}")

    evaluacion = pd.DataFrame({"Metrica": ["MAE", "RMSE", "R2"], "Valor": [mae, rmse, r2]})
    return evaluacion

def predecir_futuro(modelo, historico, cantidad_anios=3):
    ultimo_anio = int(historico["year"].max())
    anios_futuros = pd.DataFrame({"year": range(ultimo_anio + 1, ultimo_anio + cantidad_anios + 1)})

    predicciones = modelo.predict(anios_futuros)

    resultados = pd.DataFrame({"year": anios_futuros["year"], "popularidad_predicha": predicciones})
    resultados["popularidad_predicha"] = resultados["popularidad_predicha"].clip(lower=0)

    print("\n===== PRONOSTICO DE POPULARIDAD =====")
    print(resultados.to_string(index=False))
    return resultados

def entrenar_modelo_completo(historico):
    X = historico[["year"]]
    y = historico["popularidad_promedio"]

    modelo = LinearRegression()
    modelo.fit(X, y)

    print("\nModelo final entrenado con todos los datos historicos.")
    return modelo

def graficar_pronostico(historico, prueba, predicciones_futuras):
    carpeta = obtener_carpeta_resultados()

    plt.figure(figsize=(12, 6))
    plt.plot(historico["year"], historico["popularidad_promedio"], marker="o", label="Datos historicos")
    plt.scatter(prueba["year"], prueba["popularidad_promedio"], color="orange", s=80, label="Datos reales de prueba")
    plt.plot(predicciones_futuras["year"],predicciones_futuras["popularidad_predicha"], marker="X", linestyle="--", color="red", label="Predicciones futuras")

    plt.title("Pronostico de popularidad de series de television")
    plt.xlabel("Anio")
    plt.ylabel("Popularidad promedio")
    plt.legend()
    plt.grid(True)

    ruta = carpeta / "grafica_pronostico.png"
    plt.savefig(ruta)

    print(f"\nGrafica del pronostico guardada en: {ruta.resolve()}")

    plt.show()
    plt.close()

def guardar_datos_historicos(historico):
    carpeta = obtener_carpeta_resultados()
    ruta = carpeta / "datos_historicos.csv"
    historico.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"\nDatos historicos guardados en: {ruta.resolve()}")

def guardar_evaluacion(evaluacion):
    carpeta = obtener_carpeta_resultados()
    ruta = carpeta / "evaluacion_modelo.csv"
    evaluacion.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"\nEvaluacion guardada en: {ruta.resolve()}")

def guardar_predicciones(predicciones_futuras):
    carpeta = obtener_carpeta_resultados()
    ruta = carpeta / "predicciones_futuras.csv"

    predicciones_futuras.to_csv(ruta, index=False, encoding="utf-8-sig")
    print(f"\nPredicciones futuras guardadas en: {ruta.resolve()}")

def menu_pronostico(df):
    while True:
        print("\n========== PRONOSTICO (FORECASTING) ==========")
        print("1. Ejecutar pronostico")
        print("2. Salir")

        opcion = input("\nSelecciona una opcion: ")

        if opcion == "1":
            datos = preparar_datos(df)
            if datos is None:
                continue

            historico = agrupar_por_anio(datos)
            resultado_division = dividir_datos(historico)
            if resultado_division is None:
                continue

            X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = resultado_division
            modelo = entrenar_modelo(X_entrenamiento, y_entrenamiento)
            evaluacion = evaluar_modelo(modelo, X_prueba, y_prueba)
            modelo_final = entrenar_modelo_completo(historico)
            predicciones_futuras = predecir_futuro(modelo_final, historico)
            prueba = historico.iloc[-len(X_prueba):].copy()

            graficar_pronostico(historico, prueba, predicciones_futuras)
            guardar_datos_historicos(historico)
            guardar_evaluacion(evaluacion)
            guardar_predicciones(predicciones_futuras)
            print("\nPronostico completado correctamente.")

        elif opcion == "2":
            break
        else:
            print("\nOpcion no valida. Intenta nuevamente.")