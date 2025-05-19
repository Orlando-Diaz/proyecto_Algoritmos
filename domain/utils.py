import os
import bibtexparser
import seaborn as sns
from bibtexparser.bwriter import BibTexWriter
from fuzzywuzzy import fuzz
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt




def leer_bibtex(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return bibtexparser.load(file).entries
    except FileNotFoundError:
        print(f"Archivo no encontrado: {file_path}")
        return []
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return []

def normalize_data(entries):
    """Normaliza los datos de los artículos"""
    return [{
        'title': e.get('title', '').strip().lower(),
        'author': e.get('author', '').strip().lower(),
        'year': e.get('year', '').strip(),
        'doi': e.get('doi', '').strip(),
        'journal': e.get('journal', '').strip().lower(),
        'publisher': e.get('publisher', '').strip().lower(),
        'abstract': e.get('abstract', '').strip().lower(),  # Normalizando 'abstract'
        'raw_data': e
    } for e in entries]


def save_bibtex(filename, articles):
    """Guarda artículos en archivo BibTeX, filtrando None"""
    db = bibtexparser.bibdatabase.BibDatabase()
    # Filtrar artículos que no son None y tienen raw_data
    db.entries = [a['raw_data'] for a in articles if a is not None and 'raw_data' in a]
    
    writer = BibTexWriter()
    writer.order_entries_by = None
    writer.indent = '    '
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(writer.write(db))

def obtener_anio_valido(x):
    """Obtiene año como entero o devuelve valor por defecto"""
    if isinstance(x, dict):  # Si es un diccionario de artículo
        year_str = x.get('year', '')
    else:  # Si es directamente el valor year
        year_str = x
        
    if year_str and str(year_str).strip().isdigit():
        return int(str(year_str).strip())
    return 9999  # Valor por defecto para años inválidos

def buscar_duplicados(articles):
    """Identifica artículos duplicados usando fuzzy matching"""
    unicos = []
    duplicados = []
    vistos = {}
    
    for article in articles:
        title = article.get('title', '').strip().lower()
        if not title:
            print("Artículo sin título:", article)
        author = article.get('author', '').strip().lower()
        key = (title, author)

        # Verificar si ya existe un artículo similar
        duplicado = False
        for visto_key in vistos:
            if fuzz.ratio(article['title'], visto_key[0]) > 90:
                duplicados.append(article)
                duplicado = True
                break
        
        if not duplicado:
            vistos[key] = article
            unicos.append(article)
    
    return unicos, duplicados


def graficar_tiempos(mediciones, num_articles):
    """Genera gráfico de comparación de tiempos"""
    metodos = list(mediciones.keys())
    tiempos = list(mediciones.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(metodos, tiempos)
    plt.xlabel('Método de Ordenamiento')
    plt.ylabel('Tiempo (s)')
    plt.title(f'Comparación de Tiempos ({num_articles} artículos)')
    plt.xticks(rotation=45, ha='right')
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.6f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    

def extraer_abstracts_bibtex(ruta):
    with open(ruta, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    abstracts = []
    etiquetas = []

    for entry in bib_database.entries:
        if 'abstract' in entry:
            abstracts.append(entry['abstract'])

            # Intenta usar keywords, si no hay, intenta con el título, o marca como Desconocido
            keyword = entry.get('keywords', '').strip()
            if keyword:
                etiquetas.append(keyword)
            else:
                etiquetas.append(entry.get('title', 'Desconocido')[:30])  # Usa título recortado

    return abstracts, etiquetas

"""""
# Graficar dendrograma
def graficar_dendrograma_rq5(dist_matrix, labels, metodo='ward', titulo='Dendrograma'):
    condensed = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed, method=metodo)
    plt.figure(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
    plt.title(titulo)
    plt.tight_layout()
    plt.show()
"""


def graficar_similitud(sim_matrix, etiquetas, titulo="Similitud de Abstracts - SBERT"):
    import numpy as np
    import matplotlib.pyplot as plt

    # Asegurarse de que sea una matriz de numpy
    sim_matrix = np.array(sim_matrix)

    # Normalizar si tiene valores negativos o si proviene de distancias
    if np.min(sim_matrix) < 0 or np.max(sim_matrix) > 1:
        max_val = np.max(sim_matrix)
        sim_matrix = 1 - sim_matrix / max_val
        sim_matrix = np.clip(sim_matrix, 0, 1)

    # Eliminar la diagonal (similitud consigo mismo)
    np.fill_diagonal(sim_matrix, 0)

    # Calcular promedio de similitud con los demás
    promedio = sim_matrix.sum(axis=1) / (sim_matrix.shape[1] - 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(etiquetas, promedio)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=90)
    ax.set_title(titulo)
    plt.tight_layout()

    return fig





def graficar_heatmap_similitud(dist_matrix):
    max_dist = np.max(dist_matrix)
    simil_matrix = 100 * (1 - dist_matrix / max_dist)
    etiquetas = [f"A{i+1}" for i in range(len(dist_matrix))]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(simil_matrix, xticklabels=etiquetas, yticklabels=etiquetas,
                cmap="viridis", annot=True, fmt=".1f", ax=ax)
    ax.set_title("Similitud entre Abstracts (%) - Basado en WMD")
    ax.set_xlabel("Abstract")
    ax.set_ylabel("Abstract")
    plt.tight_layout()

    return fig  # ✅


"""""
def graficar_dendrograma_rq5(dist_matrix, labels, metodo='ward', titulo='Dendrograma'):
    condensed = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed, method=metodo)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title(titulo)
    plt.tight_layout()
    
    return fig  # ✅
"""

def graficar_dendrograma_rq5(dist_matrix, labels, metodo='ward', titulo='Dendrograma'):
    condensed = squareform(dist_matrix, checks=False)
    linkage_matrix = linkage(condensed, method=metodo)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, ax=ax)
    ax.set_title(titulo)
    plt.tight_layout()
    
    return fig

