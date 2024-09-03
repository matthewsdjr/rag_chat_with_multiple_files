---

# 📚 Chatbot con Múltiples Archivos (Streamlit) 🤖

Bienvenido a **Chatbot con Múltiples Archivos**, una aplicación interactiva construida con Streamlit que permite a los usuarios subir archivos PDF y chatear con el modelo Llama 3.1 70b, utilizando los servidores de Groq para evitar la necesidad de hardware de alta potencia.

## 📋 Descripción del Proyecto

Este proyecto utiliza un modelo de embeddings avanzado, **FlagEmbedding**, que ocupa la segunda posición en el Leaderboard de MTEB. Al ser un modelo open-source, proporciona una excelente precisión y eficiencia en la generación de embeddings de texto. Los embeddings generados son almacenados y gestionados por **FAISS**, un robusto vector store desarrollado por Meta, que facilita la búsqueda y recuperación de información relevante a partir de grandes conjuntos de datos.

## Chatbot Demo
https://github.com/user-attachments/assets/25b94657-a64f-48b8-8f19-39156e7c820d

![Vídeo sin título](https://github.com/user-attachments/assets/58e5eaaf-b062-47dd-8bc9-b14812ded417)

### Características Principales

- **Soporte de Múltiples Archivos**: Permite la carga de múltiples archivos PDF para una experiencia de chat fluida y continua.
- **Modelo de Lenguaje Avanzado**: Utiliza el modelo Llama 3.1 70b, conocido por su capacidad para comprender y generar texto de alta calidad.
- **Servidores Groq**: Aprovecha los servidores de Groq para ejecutar el modelo de lenguaje de manera eficiente sin requerir hardware especializado.
- **Embeddings de Alto Rendimiento**: Implementación del modelo FlagEmbedding para la representación vectorial de documentos.
- **Vector Store Eficiente**: Integración con FAISS para un almacenamiento y recuperación rápida de información.

## 🚀 Comenzando

Estas instrucciones te guiarán para obtener una copia del proyecto en funcionamiento en tu máquina local para propósitos de desarrollo y pruebas.

### Pre-requisitos

Asegúrate de tener las siguientes herramientas instaladas:

- **Python** (versión 3.7 o superior)
- **pip** (gestor de paquetes de Python)

### Instalación

1. **Clona el repositorio:**

   ```bash
   git clone https://github.com/matthewsdjr/Chatbot-with-multiple-files.git
   cd tu-repositorio
   ```

2. **Crea y activa un entorno virtual:**

   ```bash
   python -m venv env
   source env/bin/activate   # En Windows usa `env\Scripts\activate`
   ```

3. **Instala las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

### Uso

1. **Inicia la aplicación Streamlit:**

   ```bash
   streamlit run app.py
   ```

2. **Carga tus archivos PDF** y comienza a chatear con el modelo directamente desde la interfaz.

## 🛠️ Construido con

- **[Streamlit](https://streamlit.io/)** - Framework para construir aplicaciones web de datos en Python.
- **[FlagEmbedding](https://huggingface.co/BAAI/bge-m3)** - Modelo de embeddings de alta precisión.
- **[FAISS](https://github.com/facebookresearch/faiss)** - Librería para la búsqueda de similitudes vectoriales.
- **[Groq](https://groq.com/)** - Servidores de alto rendimiento para ejecutar modelos de IA.

## 🤝 Contribuciones

Las contribuciones son bienvenidas.

## 📧 Contacto

Si tienes alguna pregunta o sugerencia, no dudes en abrir un issue o contactarme directamente a través de [matthews.djr@gmail.com](mailto:matthews.djr@gmail.com).

---
