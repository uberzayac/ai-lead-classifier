from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

good_examples = [
    "Хочу заказать 1000 визиток, интересует стоимость и сроки",
    "Нужен расчет тиража 500 буклетов",
    "Пришлите коммерческое предложение на печать листовок",
    "ДТФ печать",
    "Печать на футболках"
]

bad_examples = [
    "Предлагаем SEO продвижение",
    "Купите базу клиентов",
    "Заказ плёнки",
    ""
    "Рассылка по бизнесу"
]

good_embeddings = model.encode(good_examples)
bad_embeddings = model.encode(bad_examples)

class EmailRequest(BaseModel):
    text: str

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/classify")
def classify(req: EmailRequest):
    email_embedding = model.encode([req.text])[0]

    good_score = np.mean([cosine_similarity(email_embedding, g) for g in good_embeddings])
    bad_score = np.mean([cosine_similarity(email_embedding, b) for b in bad_embeddings])

    if good_score > bad_score:
        quality = "high"
        score = float(good_score)
    else:
        quality = "low"
        score = float(bad_score)

    return {
        "quality": quality,
        "score": round(score, 3),
        "good_score": round(float(good_score), 3),
        "bad_score": round(float(bad_score), 3)
    }
