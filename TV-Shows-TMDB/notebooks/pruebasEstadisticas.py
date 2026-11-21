#semana 7
#¿El promedio de calificaciones (vote_average) cambia significativamente 
#dependiendo del idioma original (original_language) de las series?

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.stats.multicomp import pairwise_tukeyhsd

def obtener_ruta_resultados():
    ruta = (Path(__file__).resolve().parent.parent/ "resultados"/ "estadistica"/ "pruebas_estadisticas")
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta

def guardar_csv(df, nombre):
    ruta = obtener_ruta_resultados()
    archivo = ruta / f"{nombre}.csv"
    df.to_csv(archivo, index=False, encoding="utf-8-sig")

    print(f"\nArchivo guardado: {archivo.name}")

def preparar_grupos(df):
    datos = df[["original_language", "vote_average"]].dropna()

    idiomas_disponibles = sorted(datos["original_language"].unique())

    print("\n===== IDIOMAS DISPONIBLES =====")
    for i, idioma in enumerate(idiomas_disponibles, start=1):
        cantidad = (datos["original_language"] == idioma).sum()
        print(f"{i}. {idioma} ({cantidad} series)")

    while True:
        seleccion = input("\nSelecciona los idiomas a comparar (separados por coma): ").strip()

        if not seleccion:
            print("Error: No ingresaste ninguna seleccion.")
            continue
        try:
            indices = [int(x.strip()) for x in seleccion.split(",")]

        except ValueError:
            print("Error: Debes ingresar solamente numeros.")
            continue

        if any(i < 1 or i > len(idiomas_disponibles) for i in indices):
            print("Error: Uno o mas numeros no son validos.")
            continue

        if len(indices) != len(set(indices)):
            print("Error: No puedes seleccionar idiomas repetidos.")
            continue

        if len(indices) < 2:
            print("Error: Debes seleccionar al menos 2 idiomas.")
            continue

        idiomas = [idiomas_disponibles[i - 1] for i in indices]
        grupos = []

        for idioma in idiomas: 
            grupo = datos.loc[datos["original_language"] == idioma, "vote_average"].values
            grupos.append(grupo)

        if any(len(grupo) < 2 for grupo in grupos):
            print("Error: Cada idioma debe tener al menos 2 registros.")
            continue

        print("\nIdiomas seleccionados correctamente:")
        print(", ".join(idiomas))

        return idiomas, grupos

def prueba_anova(grupos):
    resultado = stats.f_oneway(*grupos)

    estadistico = resultado.statistic
    valor_p = resultado.pvalue

    print("\n===================================")
    print("          PRUEBA ANOVA")
    print("===================================")

    print(f"Estadistico F: {estadistico:.4f}")
    print(f"Valor p: {valor_p:.6f}")

    if valor_p < 0.05:
        conclusion = "Diferencias significativas"
        print("\nSe rechaza la hipotesis nula.")
    else:
        conclusion = "No se encontraron diferencias significativas"
        print("\nNo se rechaza la hipotesis nula.")

    resultados = pd.DataFrame([{
        "Prueba": "ANOVA",
        "Estadistico F": estadistico,
        "Valor p": valor_p,
        "Nivel significancia": 0.05,
        "Conclusion": conclusion
    }])

    guardar_csv(resultados, "resultados_anova")

    return estadistico, valor_p

def prueba_kruskal(grupos):
    resultado = stats.kruskal(*grupos)

    estadistico = resultado.statistic
    valor_p = resultado.pvalue

    print("\n===================================")
    print("       PRUEBA KRUSKAL-WALLIS")
    print("===================================")

    print(f"Estadistico H: {estadistico:.4f}")
    print(f"Valor p: {valor_p:.6f}")

    if valor_p < 0.05:
        conclusion = "Diferencias significativas"
        print("\nSe rechaza la hipotesis nula.")
    else:
        conclusion = "No se encontraron diferencias significativas"
        print("\nNo se rechaza la hipotesis nula.")

    resultados = pd.DataFrame([{
        "Prueba": "Kruskal-Wallis",
        "Estadistico H": estadistico,
        "Valor p": valor_p,
        "Nivel significancia": 0.05,
        "Conclusion": conclusion
    }])

    guardar_csv(resultados, "resultados_kruskal")

    return estadistico, valor_p

def resumen_grupos(idiomas, grupos):
    resultados = []

    print("\n===================================")
    print("       RESUMEN DE GRUPOS")
    print("===================================")

    for idioma, valores in zip(idiomas, grupos):
        media = np.mean(valores)
        mediana = np.median(valores)
        desviacion = np.std(valores, ddof=1)
        cantidad = len(valores)

        resultados.append({
            "Idioma": idioma,
            "Cantidad": cantidad,
            "Media": media,
            "Mediana": mediana,
            "Desviacion estandar": desviacion,
            "Minimo": np.min(valores),
            "Maximo": np.max(valores)
        })

        print(f"\nIdioma: {idioma}")
        print(f"Cantidad: {cantidad}")
        print(f"Media: {media:.2f}")
        print(f"Mediana: {mediana:.2f}")
        print(f"Desviacion estandar: {desviacion:.2f}")

    tabla = pd.DataFrame(resultados)

    guardar_csv(tabla, "resumen_grupos")

    return pd.DataFrame(resultados)

def prueba_t_student(idiomas, grupos):
    if len(idiomas) != 2:
        print("\nError: Para la prueba T debes seleccionar exactamente 2 idiomas.")
        return

    grupo1 = grupos[0]
    grupo2 = grupos[1]

    idioma1 = idiomas[0]
    idioma2 = idiomas[1]

    resultado = stats.ttest_ind(grupo1, grupo2, equal_var=False)

    estadistico = resultado.statistic
    valor_p = resultado.pvalue

    print("\n===== PRUEBA T DE STUDENT =====")
    print(f"Comparacion: {idioma1} vs {idioma2}")
    print("=================================")
 
    print(f"Media {idioma1}: {np.mean(grupo1):.2f}")
    print(f"Media {idioma2}: {np.mean(grupo2):.2f}")
    print(f"Estadistico t: {estadistico:.4f}")
    print(f"Valor p: {valor_p:.6f}")

    if valor_p < 0.05:
        conclusion = "Diferencias significativas"
        print("\nSe rechaza la hipotesis nula, pues existe una diferencia significativa entre los grupos.")
    else:
        conclusion = "No se encontraron diferencias significativas"
        print("\nNo se rechaza la hipotesis nula, pues no existe evidencia suficiente de diferencias.")

    resultados = pd.DataFrame([{
        "Idioma 1": idioma1,
        "Idioma 2": idioma2,
        "Media 1": np.mean(grupo1),
        "Media 2": np.mean(grupo2),
        "Estadistico t": estadistico,
        "Valor p": valor_p,
        "Conclusion": conclusion
    }])

    guardar_csv(resultados, "resultados_t_student")


def prueba_posthoc(idiomas, grupos):
    valores = []
    etiquetas = []

    for idioma, grupo in zip(idiomas, grupos):
        valores.extend(grupo)
        etiquetas.extend([idioma] * len(grupo))

    resultado = pairwise_tukeyhsd(endog=valores, groups=etiquetas, alpha=0.05)

    print("\n===================================")
    print("       PRUEBA POST HOC TUKEY")
    print("===================================")

    print(resultado)

    tabla = pd.DataFrame(resultado._results_table.data[1:], columns=resultado._results_table.data[0])

    guardar_csv(tabla, "resultados_posthoc")

    return tabla

def grafica_boxplot(df, idiomas):
    datos = df[df["original_language"].isin(idiomas)][["original_language", "vote_average"]].dropna()

    plt.figure(figsize=(10, 6))

    sns.boxplot(data=datos, x="original_language", y="vote_average")

    plt.title("Distribucion de calificaciones por idioma")
    plt.xlabel("Idioma original")
    plt.ylabel("Calificacion promedio")

    plt.tight_layout()

    ruta = obtener_ruta_resultados() / "graficas"
    ruta.mkdir(parents=True, exist_ok=True)

    plt.savefig(ruta / "boxplot_calificaciones_idiomas.png", dpi=300, bbox_inches="tight")

    print("\nGrafica guardada correctamente.")

    plt.show()

def grafica_medias(idiomas, grupos):

    medias = [np.mean(grupo) for grupo in grupos]

    plt.figure(figsize=(10, 6))

    sns.barplot(x=list(idiomas), y=medias)

    plt.title("Calificacion promedio por idioma")
    plt.xlabel("Idioma original")
    plt.ylabel("Calificacion promedio")

    plt.tight_layout()

    ruta = obtener_ruta_resultados() / "graficas"
    ruta.mkdir(parents=True, exist_ok=True)

    plt.savefig(ruta / "medias_calificaciones_idiomas.png", dpi=300, bbox_inches="tight")

    print("\nGrafica guardada correctamente.")

    plt.show()

def menu_pruebas_estadisticas(df):
    while True:
        print("\n===================================")
        print("       PRUEBAS ESTADISTICAS")
        print("===================================")
        print("1. Resumen de grupos")
        print("2. Prueba ANOVA")
        print("3. Prueba Kruskal-Wallis")
        print("4. Prueba T Student")
        print("5. Pruebas post hoc Tukey")
        print("6. Grafica boxplot")
        print("7. Grafica de medias")
        print("8. Ejecutar todos los analisis")
        print("0. Regresar al menu principal")
        print("===================================")

        opcion = input("Selecciona una opcion: ")

        if opcion == "0":
            break

        idiomas, grupos = preparar_grupos(df)

        if opcion == "1":
            resumen_grupos(idiomas, grupos)

        elif opcion == "2":
            prueba_anova(grupos)

        elif opcion == "3":
            prueba_kruskal(grupos)

        elif opcion == "4":
            prueba_t_student(idiomas, grupos)

        elif opcion == "5":
            prueba_posthoc(idiomas, grupos)

        elif opcion == "6":
            grafica_boxplot(df, idiomas)

        elif opcion == "7":
            grafica_medias(idiomas, grupos)

        elif opcion == "8":
            resumen_grupos(idiomas, grupos)
            prueba_anova(grupos)
            prueba_kruskal(grupos)
            prueba_posthoc(idiomas, grupos)
            grafica_boxplot(df, idiomas)
            grafica_medias(idiomas, grupos)

        else:
            print("\nOpcion no valida...")