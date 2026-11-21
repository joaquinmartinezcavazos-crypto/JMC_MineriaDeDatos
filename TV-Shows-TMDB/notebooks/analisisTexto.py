#semana 14
import pandas as pd
import matplotlib.pyplot as plt
import re

from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def obtener_carpeta_resultados():
    base_dir = Path(__file__).resolve().parent.parent
    carpeta = base_dir / "resultados" / "modelos" / "analisis_texto"
    carpeta.mkdir(parents=True, exist_ok=True)

    return carpeta

def preparar_textos(df):
    textos = df["overview"].dropna().astype(str)
    if textos.empty:
            print("\nNo hay descripciones disponibles en el dataset.")
            return []
    
    textos_limpios = []
    palabras_vacias = set(ENGLISH_STOP_WORDS)
    palabras_vacias.update({"series", "show", "story", "life", "one", "two", "new", "young"})

    for texto in textos:
        texto = texto.lower()
        texto = re.sub(r"http\S+|www\S+", "", texto)
        texto = re.sub(r"[^a-z\s]", " ", texto)

        palabras = texto.split()
        palabras = [palabra for palabra in palabras if palabra not in palabras_vacias and len(palabra) > 2]

        textos_limpios.append(" ".join(palabras))

    print("\n===== PREPARACION DE TEXTOS =====")
    print(f"Descripciones originales: {len(textos)}")
    print(f"Descripciones procesadas: {len(textos_limpios)}")

    return textos_limpios

def analizar_frecuencias(textos_limpios, cantidad=20):
    texto_completo = " ".join(textos_limpios)

    frecuencias = Counter(texto_completo.split())
    if not frecuencias:
        print("\nNo se encontraron palabras para analizar.")
        return pd.DataFrame(columns=["Palabra", "Frecuencia"])

    palabras_frecuentes = frecuencias.most_common(cantidad)

    tabla = pd.DataFrame(palabras_frecuentes, columns=["Palabra", "Frecuencia"])

    print("\n===== PALABRAS MAS FRECUENTES =====")
    print(tabla.to_string(index=False))

    ruta = obtener_carpeta_resultados() / "palabras_frecuentes.csv"
    tabla.to_csv(ruta, index=False, encoding="utf-8-sig")

    print(f"\nResultados guardados en: {ruta}")

    return tabla

def generar_nube_palabras(textos_limpios):
    texto_completo = " ".join(textos_limpios)
    if not texto_completo.strip():
        print("\nNo hay palabras suficientes para generar la nube.")
        return

    nube = WordCloud(width=1200, height=600, background_color="white", max_words=100, collocations=False).generate(texto_completo)

    plt.figure(figsize=(14, 7))
    plt.imshow(nube, interpolation="bilinear")
    plt.axis("off")
    plt.title("Nube de palabras de las series")

    ruta = obtener_carpeta_resultados() / "nube_palabras.png"

    plt.savefig(ruta, bbox_inches="tight")

    print(f"\nNube de palabras guardada en: {ruta}")

    plt.show()
    plt.close()

def analizar_longitud_textos(df):
    textos = df["overview"].dropna().astype(str)

    longitudes = textos.str.split().str.len()

    estadisticas = pd.DataFrame({
        "Estadistica": [
            "Cantidad de descripciones",
            "Promedio de palabras",
            "Mediana de palabras",
            "Desviacion estandar",
            "Cantidad minima de palabras",
            "Cantidad maxima de palabras"
        ],
        "Valor": [
            len(longitudes),
            round(longitudes.mean(), 2),
            round(longitudes.median(), 2),
            round(longitudes.std(), 2),
            longitudes.min(),
            longitudes.max()
        ]
    })

    print("\n===== ANALISIS DE LONGITUD DE DESCRIPCIONES =====")
    print(estadisticas.to_string(index=False))

    ruta_csv = obtener_carpeta_resultados() / "longitud_descripciones.csv"
    estadisticas.to_csv(ruta_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.hist(longitudes, bins=20, edgecolor="black")

    plt.title("Distribucion de longitud de las descripciones")
    plt.xlabel("Cantidad de palabras")
    plt.ylabel("Cantidad de series")
    plt.grid(axis="y", alpha=0.3)

    ruta_grafica = obtener_carpeta_resultados() / "distribucion_longitud.png"
    plt.savefig(ruta_grafica, bbox_inches="tight")

    print(f"\nEstadisticas guardadas en: {ruta_csv}")
    print(f"Grafica guardada en: {ruta_grafica}")

    plt.show()
    plt.close()

def menu_analisis_texto(df):
    while True:
        print("\n===================================")
        print("         ANALISIS DE TEXTO")
        print("===================================")
        print("1. Ejecutar analisis completo")
        print("0. Regresar al menu principal")
        print("===================================")

        opcion = input("Selecciona una opcion: ")

        if opcion == "1":
            textos_limpios = preparar_textos(df)
            if not any(textos_limpios):
                print("\nNo hay texto suficiente para realizar el analisis.")
                continue
            analizar_frecuencias(textos_limpios)
            generar_nube_palabras(textos_limpios)
            analizar_longitud_textos(df)
            print("\nAnalisis de texto finalizado.")

        elif opcion == "0":
            break
        else:
            print("\nOpcion no valida.")