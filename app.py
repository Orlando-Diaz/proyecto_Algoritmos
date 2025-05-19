import numpy as np
import streamlit as st
import os
import shutil
import time
import copy
import matplotlib.pyplot as plt
from domain.agrupamiento.clustering_algoritmos import CompleteLinkageClustering, SingleLinkageClustering
from domain.agrupamiento.dendograma import ClusteringJerarquico
from domain.agrupamiento.similitud import calcular_matriz_similitud
from domain.metodos_ordenamiento import *
from domain.metodos_ordenamiento.binary_insertion_sort import BinaryInsertionSort
from domain.requerimientos import requerimiento2, requerimiento3, requerimiento5
from domain.utils import graficar_dendrograma_rq5, graficar_heatmap_similitud, graficar_similitud, leer_bibtex, normalize_data, save_bibtex, buscar_duplicados
from main import ejecutar_req_2_y_3

# Configuración inicial
st.set_page_config(page_title="Procesamiento de Artículos", layout="wide")
st.title("📚 Sistema de Procesamiento de Artículos")

# Definir carpeta temporal (antes para que esté disponible siempre)
temp_folder = "archivos_temp"

# --- Carga de archivos ---
uploaded_files = st.file_uploader("Selecciona archivos BibTeX", type=["bib"], accept_multiple_files=True)

if uploaded_files:
    # Crear carpeta temporal si no existe
    os.makedirs(temp_folder, exist_ok=True)
    
    # Guardar archivos subidos
    file_paths = []
    for file in uploaded_files:
        temp_path = os.path.join(temp_folder, file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(temp_path)

    # --- Menú de opciones ---
    opcion = st.radio(
        "Selecciona el requerimiento a ejecutar:",
        options=[
            "Requerimientos 1 al 3 (ordenamiento y análisis)",
            "Solo Reqs 2 y 3 (sin ordenar)",
            "Requerimiento 4 (agrupamiento jerárquico)",
            "Requerimiento 5 (similitud con SBERT)"
        ],
        index=0  # Opción por defecto
    )

    # --- Requerimientos 1 al 3 ---
    if opcion == "Requerimientos 1 al 3 (ordenamiento y análisis)":
        # num_articulos = st.slider("Cantidad máxima de artículos a analizar", 5, 5000, 10)
        
        if st.button("Ejecutar Requerimientos 1 al 3"):
            try:
                # Inicializar progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Paso 1: Leer y normalizar artículos
                status_text.text("Leyendo archivos BibTeX...")
                all_articles = []
                for path in file_paths:
                    entries = leer_bibtex(path)
                    st.write(f"Archivo {path} tiene {len(entries)} artículos")
                    all_articles.extend(normalize_data(entries))
                progress_bar.progress(25)
                
                st.write(f"Artículos totales cargados: {len(all_articles)}")
                
                # Paso 2: Buscar duplicados
                status_text.text("Buscando artículos duplicados...")
                articulos_unicos, duplicados = buscar_duplicados(all_articles)
                st.write(f"Artículos únicos después de eliminar duplicados: {len(articulos_unicos)}")
                
                # Quitar límite para analizar todos los artículos
                # articulos_unicos = articulos_unicos[:num_articulos]
                
                progress_bar.progress(50)
                
                # Mostrar resumen inicial
                st.subheader("📊 Resumen inicial")
                col1, col2 = st.columns(2)
                col1.metric("Artículos cargados", len(all_articles))
                col2.metric("Artículos únicos a analizar", len(articulos_unicos))
                
                # Paso 3: Ordenamiento
                status_text.text("Ejecutando algoritmos de ordenamiento...")
                metodos = {
                    'TimSort': TimSort(),
                    'QuickSort': QuickSort(),
                    'SelectionSort': SelectionSort(),
                    'HeapSort': HeapSort(),
                    'CombSort': CombSort(),
                    'TreeSort': TreeSort(),
                    'BitonicSort': BitonicSort(),
                    'GnomeSort': GnomeSort(),
                    'RadixSort': RadixSort(),
                    'PigeonholeSort': PigeonholeSort(),
                    'BucketSort': BucketSort(),
                    'BinaryInsertionSort': BinaryInsertionSort()
                }
                
                resultados = []
                for nombre, instancia in metodos.items():
                    inicio = time.time()
                    instancia.ordenar(copy.deepcopy(articulos_unicos))
                    tiempo = time.time() - inicio
                    resultados.append((nombre, tiempo))
                    st.write(f"✅ {nombre}: {tiempo:.4f} segundos")
                
                progress_bar.progress(75)
                
                # Gráfico de tiempos
                st.subheader("⏱ Tiempos de ordenamiento")
                nombres = [r[0] for r in resultados]
                tiempos = [r[1] for r in resultados]
                
                fig, ax = plt.subplots()
                ax.bar(nombres, tiempos)
                ax.set_ylabel('Segundos')
                ax.set_title('Comparación de algoritmos')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                plt.close(fig)
                
                # Paso 4: Guardar resultados
                status_text.text("Guardando resultados...")
                final_ordenados = TimSort().ordenar(articulos_unicos)
                save_bibtex("articulos_unificados.bib", final_ordenados)
                
                # Paso 5: Requerimiento 2
                st.subheader("📈 Requerimiento 2 - Estadísticas")

                try:
                    r2 = requerimiento2(final_ordenados)
                    r2.procesar_estadisticas()
                    
                    with st.expander("Ver estadísticas completas"):
                        st.write(r2.mostrar_resultados())
                except Exception as e:
                    st.error(f"Error en Requerimiento 2: {str(e)}")
                    st.exception(e)

                # Paso 6: Requerimiento 3
                st.subheader("📊 Requerimiento 3 - Análisis de frecuencias")

                try:
                    os.makedirs("imagenes/salidas", exist_ok=True)  # asegurar carpeta
                    
                    r3 = requerimiento3(ruta_bibtex="articulos_unificados.bib", carpeta_salida="imagenes/salidas")
                    r3.analizar_frecuencias()
                    r3.generar_nube_palabras()  # 🔥 AÑADIDO
                    # r3.generar_grafico_coocurrencia()  # opcional
                    # r3.guardar_csv_frecuencias()       # opcional

                    wordcloud_path = os.path.join("imagenes", "salidas", "nube_general.png")  

                    if os.path.exists(wordcloud_path):
                        st.image(wordcloud_path)
                    else:
                        st.warning("No se generó la nube de palabras")
                        st.write("Ruta esperada:", wordcloud_path)
                except Exception as e:
                    st.error(f"Error en Requerimiento 3: {str(e)}")
                    st.exception(e)

                progress_bar.progress(100)
                status_text.text("Proceso completado!")
                st.success("✅ Todos los requerimientos procesados correctamente")


            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)  # Muestra el traceback completo

    # --- Requerimientos 2 y 3 ---
    elif opcion == "Solo Reqs 2 y 3 (sin ordenar)":
        st.subheader("📑 Ejecutar Requerimientos 2 y 3 combinados")

        if st.button("🚀 Ejecutar análisis combinado (Requerimientos 2 y 3)"):
            try:
                ruta_archivo_unificado = "articulos_unificados.bib"
                if not os.path.exists(ruta_archivo_unificado):
                    st.error(f"❌ El archivo {ruta_archivo_unificado} no existe.")
                else:
                    tablas_frecuencia = ejecutar_req_2_y_3(ruta_archivo_unificado)

                    # Mostrar nube general
                    wordcloud_path = os.path.join("imagenes", "salidas", "nube_general.png")
                    if os.path.exists(wordcloud_path):
                        st.image(wordcloud_path, caption="Nube de palabras general")

                    # Mostrar gráfico de co-ocurrencias
                    cooc_path = os.path.join("imagenes", "salidas", "co_word_network.png")
                    if os.path.exists(cooc_path):
                        st.image(cooc_path, caption="Red de co-ocurrencia de variables")

                    # Mostrar tablas si están disponibles
                    if tablas_frecuencia:
                        for categoria, df in tablas_frecuencia.items():
                            st.markdown(f"### 📌 Frecuencia: {categoria}")
                            st.dataframe(df)


                    st.success("✅ Requerimientos 2 y 3 ejecutados correctamente")

            except Exception as e:
                st.error("❌ Error al ejecutar los requerimientos 2 y 3.")
                st.exception(e)



    # --- Requerimiento 4 ---
    elif opcion == "Requerimiento 4 (agrupamiento jerárquico)":
        st.subheader("🔍 Agrupamiento Jerárquico de Abstracts")

        if st.button("Ejecutar Agrupamiento"):
            try:
                ruta_bibtex = "articulos_unificados.bib"  # Ajusta si necesitas ruta dinámica

                if not os.path.exists(ruta_bibtex):
                    st.error("❌ El archivo .bib no existe.")
                else:
                    entradas = leer_bibtex(ruta_bibtex)

                    if not entradas:
                        st.warning("❌ No se encontraron artículos en el archivo.")
                    else:
                        st.info(f"✅ {len(entradas)} artículos encontrados.")

                        datos_normalizados = normalize_data(entradas)
                        abstracts = [d['abstract'] for d in datos_normalizados if d['abstract']]

                        if not abstracts:
                            st.warning("❌ No se encontraron abstracts válidos.")
                        else:
                            matriz_similitud = calcular_matriz_similitud(abstracts)

                            # SINGLE linkage
                            st.write("🔗 Clustering con SINGLE linkage")
                            clustering_single = SingleLinkageClustering()
                            linkage_single = clustering_single.fit(matriz_similitud)

                            clustering_s = ClusteringJerarquico(matriz_similitud)
                            ruta_s = "imagenes/salidas/dendrograma_single.png"
                            fig_s = clustering_s.graficar_dendrograma(linkage_single, "Dendrograma - Single Linkage", ruta_s)

                            # COMPLETE linkage
                            st.write("🔗 Clustering con COMPLETE linkage")
                            clustering_complete = CompleteLinkageClustering()
                            linkage_complete = clustering_complete.fit(matriz_similitud)

                            clustering_c = ClusteringJerarquico(matriz_similitud)
                            ruta_c = "imagenes/salidas/dendrograma_complete.png"
                            fig_c = clustering_c.graficar_dendrograma(linkage_complete, "Dendrograma - Complete Linkage", ruta_c)

                            # Mostrar resultados
                            if os.path.exists(ruta_s):
                                st.image(ruta_s, caption="Dendrograma - Single Linkage")
                            else:
                                st.warning("⚠️ No se encontró la imagen de SINGLE linkage.")

                            if os.path.exists(ruta_c):
                                st.image(ruta_c, caption="Dendrograma - Complete Linkage")
                            else:
                                st.warning("⚠️ No se encontró la imagen de COMPLETE linkage.")

                            st.success("✅ Agrupamiento completado exitosamente!")

            except Exception as e:
                st.error(f"❌ Error durante el agrupamiento: {str(e)}")
                st.exception(e)



    # --- Requerimiento 5 ---
    elif opcion == "Requerimiento 5 (similitud con SBERT)":
            st.subheader("🧠 Análisis de Similitud de Abstracts con SBERT")

            if st.button("Ejecutar Requerimiento 5"):
                try:
                    import numpy as np
                    from scipy.spatial.distance import squareform

                    ruta_bibtex = "articulos_unificados.bib"  # Ajusta si necesitas ruta dinámica

                    if not os.path.exists(ruta_bibtex):
                        st.error("❌ El archivo .bib no existe.")
                    else:
                        entries = leer_bibtex(ruta_bibtex)
                        data = normalize_data(entries)
                        data = [d for d in data if d['abstract']]
                        data = data[:50]  # Limitar a 50 abstracts

                        if len(data) < 2:
                            st.warning("❌ No hay suficientes abstracts válidos para comparar.")
                        else:
                            abstracts = [item['abstract'] for item in data]
                            etiquetas = [f"Abstract {i+1}" for i in range(len(abstracts))]

                            st.info("✅ Calculando similitud con SBERT...")

                            # Calcular matriz de similitud
                            sim_matrix_sbert = requerimiento5.calcular_similitud_sbert(abstracts)

                            # Mostrar heatmap
                            fig_heatmap = graficar_heatmap_similitud(sim_matrix_sbert)
                            st.pyplot(fig_heatmap)

                            # Mostrar gráfico de similitud
                            fig_similitud = graficar_similitud(sim_matrix_sbert, etiquetas, "Gráfico de Similitud - SBERT")
                            st.pyplot(fig_similitud)

                            # Convertir matriz de similitud a distancia
                            sim_matrix_sbert = np.array(sim_matrix_sbert)
                            dist_matrix = 1 - sim_matrix_sbert
                            np.fill_diagonal(dist_matrix, 0)
                            dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Asegurar simetría

                            # Verificar que es válida para squareform
                            try:
                                squareform(dist_matrix)  # Solo para validar
                                fig_dendro_sbert = graficar_dendrograma_rq5(dist_matrix, etiquetas, titulo="Dendrograma - Similitud SBERT")
                                st.pyplot(fig_dendro_sbert)
                            except ValueError as ve:
                                st.error(f"❌ Error generando el dendrograma: {ve}")

                            st.success("✅ Análisis de similitud completado exitosamente!")

                except Exception as e:
                    st.error(f"❌ Error durante el análisis de similitud: {str(e)}")
                    st.exception(e)



else:
    st.warning("Por favor, sube al menos un archivo BibTeX para continuar.")

# Limpieza opcional
if st.sidebar.button("🧹 Limpiar archivos temporales"):
    shutil.rmtree(temp_folder, ignore_errors=True)
    st.sidebar.success("Archivos temporales eliminados.")
