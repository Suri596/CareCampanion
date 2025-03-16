from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# Load Model
MODEL_NAME = "HuggingFaceH4/zephyr-7b-alpha"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")

class QueryRequest(BaseModel):
    question: str

@app.post("/generate")
async def generate_response(request: QueryRequest):
    prompt = request.question
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = model.generate(**inputs, max_length=200)
    response_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"response": response_text}

