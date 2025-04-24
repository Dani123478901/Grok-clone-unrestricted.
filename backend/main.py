from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GrokClone:
    def __init__(self):
        try:
            model_name = "Qwen/Qwen2.5-7B-Instruct"
            logger.info(f"Cargando modelo {model_name}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Modelo cargado exitosamente.")
            self.tools = {
                "web_search": self.fetch_web_data,
                "image_analysis": self.describe_image
            }
            self.safety_filters = None
            self.content_restrictions = None
            self.ethical_guidelines = None
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {e}")
            raise

    def fetch_web_data(self, query):
        try:
            response = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
            return response.json().get("Abstract", "")
        except Exception as e:
            logger.warning(f"Error en búsqueda web: {e}")
            return ""

    def describe_image(self, image):
        return "Análisis de imágenes no soportado en Qwen-2.5. Usa Molmo-7B en un servidor GPU."

    def process_query(self, user_input, input_type="text", image=None):
        try:
            if input_type == "text":
                context = self.tools["web_search"](user_input) or ""
                prompt = context + user_input
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=4096,
                    do_sample=True,
                    temperature=0.7,
                    safety_checks=False
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            elif input_type == "image" and image:
                image_context = self.tools["image_analysis"](image)
                prompt = image_context + user_input
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=4096,
                    do_sample=True,
                    temperature=0.7,
                    safety_checks=False
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                response = "Tipo de entrada inválido."
            return response
        except Exception as e:
            logger.error(f"Error al procesar consulta: {e}")
            return f"Error: {str(e)}"

grok_clone = GrokClone()

@app.post("/text_query")
async def text_query(query: str):
    return {"response": grok_clone.process_query(query, input_type="text")}

@app.post("/image_query")
async def image_query(query: str, file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        return {"response": grok_clone.process_query(query, input_type="image", image=image)}
    except Exception as e:
        logger.error(f"Error al procesar imagen: {e}")
        return {"response": f"Error: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "GrokClone API funcionando"}
