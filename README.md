#Nombre del modelo: deepseek-ai/deepseek-llm-7b-base
#Repositorio HF: https://huggingface.co/deepseek-ai/deepseek-llm-7b-base

# Inferencia con DeepSeek LLM 7B

Este repositorio contiene un script en Python para cargar e inferir usando el modelo de lenguaje [`deepseek-ai/deepseek-llm-7b-base`](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base) a través de la librería `transformers` de Hugging Face. Este modelo es parte de la serie LLMs de DeepSeek AI, entrenado sobre un gran corpus multilingüe.

## 📁 Archivos

- `11_2_deepseek_7b.py`: Script principal para cargar el modelo y realizar inferencia.
- `11_2_deepseek_7b.ipynb`: Versión en Jupyter Notebook (opcional, útil para exploración interactiva).

## 🚀 Requisitos

Para ejecutar el script, se recomienda un entorno con GPU (CUDA) y los siguientes paquetes instalados:

```bash
pip install torch transformers accelerate
