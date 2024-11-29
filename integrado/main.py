import os

from fastapi import FastAPI, UploadFile, File
from io import BytesIO
from PIL import Image
from get_brand_from_image import get_brand  # Importando a função get_brand do arquivo externo
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from fastapi import FastAPI, UploadFile, File
import google.generativeai as genai
import json
import numpy as np
import io
import openai
import os
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lista de domínios permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos, como GET, POST, PUT, etc.
    allow_headers=["*"],  # Permitir todos os cabeçalhos
)


# Função para consultar a API do ChatGPT
def ask_ai_about_brand(brand_name):
    genai.configure(api_key="AIzaSyDp6-CUAD5OLLCw3joQnNiq6xKRkAJdAP4")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        f"O que você acha da marca {brand_name} em termos de sustentabilidade e responsabilidade corporativa? De acordo com a PETA, há informações sobre a marca?")

    return response.text


@app.get("/")
async def root():
    return {"message": "Hello World"}

# Endpoint para upload de imagem e retorno da marca detectada
@app.post("/detect-brand")
async def detect_brand(file: UploadFile = File(...)):
    # Abrir o arquivo de imagem
    image = Image.open(BytesIO(await file.read()))

    # Salvar a imagem em um arquivo temporário
    temp_path = "temp_image.jpg"
    image.save(temp_path)

    # Usar a função get_brand para detectar a marca
    brand = get_brand(temp_path)

    # Remover o arquivo temporário após o processamento
    os.remove(temp_path)

    # Perguntar a LLM(germini) sobre a marca prevista
    ai_response = ask_ai_about_brand(brand)

    # Retornar a predição e a resposta do ChatGPT
    return {
        "predicted_label": brand,
        "ai_response": ai_response
    }

    # Retornar o nome da marca
    return {"brand": brand}

# Para rodar a aplicação, use o comando:
# uvicorn main:app --reload

@app.get("/hi")
async def hi():
    return {"hi"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
