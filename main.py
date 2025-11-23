from monitors.entailmentMonitor import monitorEntailment
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

app = FastAPI()

class CallbackUrls(BaseModel):
    language: str
    semantics: str
    entailment: str
    consistency: str
    legibility_coverage: str
    reproducibility: str
    surprisal: str
    adversarial: str
    #factcheck: str

class BigBrotherReponse(BaseModel):
    prompt: str
    reasoning: str
    answer: str
    message_id: str
    callback_urls: CallbackUrls

@app.post("/api/monitor")
async def create_item(bigBrotherResponse: BigBrotherReponse):
    prompt = bigBrotherResponse.prompt
    reasoning = bigBrotherResponse.reasoning
    answer = bigBrotherResponse.answer
    message_id = bigBrotherResponse.message_id
    callback_urls = bigBrotherResponse.callback_urls

    asyncio.create_task(monitorEntailment(message_id, callback_urls.entailment, prompt, reasoning, answer))

    return {"message": "Lil Brother Is Watching!"}

@app.get("/")
def hello():
    return {"message": "hi!"}
